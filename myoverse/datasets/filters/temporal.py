from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.fft import irfft, rfft, rfftfreq
from scipy.signal import savgol_filter, sosfilt, sosfiltfilt

from myoverse.datasets.filters._template import FilterBaseClass
from myoverse.datasets.filters.generic import (
    ApplyFunctionFilter,
    _get_windows_with_shift,
)


class SOSFrequencyFilter(FilterBaseClass):
    """Filter that applies a second-order-section filter to the input array.

    Parameters
    ----------
    input_is_chunked : bool
        Whether the input is chunked or not.
    is_output : bool
        Whether the filter is an output filter. If True, the resulting signal will be outputted by and dataset pipeline.
    name : str | None
        Name of the filter, by default None.
    run_checks : bool
        Whether to run the checks when filtering. By default, True. If False can potentially speed up performance.

        .. warning:: If False, the user is responsible for ensuring that the input array is valid.

    sos_filter_coefficients : tuple[np.ndarray, np.ndarray | float, np.ndarray]
        The second-order-section filter coefficients, typically from scipy.signal.butter with output="sos".
    forwards_and_backwards : bool
        Whether to apply the filter forwards and backwards or only forwards.
    overlap : int or None
        If input_is_chunked=True, this determines how many samples to overlap when filtering each chunk
        to reduce boundary effects. If None, a default overlap based on filter order is used.
    use_continuous_approach : bool
        If True (default), the filter will be applied to a reconstructed continuous signal first before
        reshaping back to chunks, which provides better handling of boundaries.
    real_time_mode : bool
        If True, enables real-time processing with a rolling buffer of chunks.
        This mode maintains filter state between calls and only uses past data.
        Note: When enabled, forwards_and_backwards must be False and use_continuous_approach must be False.

    Methods
    -------
    __call__(input_array: np.ndarray) -> np.ndarray
        Filters the input array. Input shape is determined by whether the allowed_input_type
        is "both", "chunked" or "not chunked".
    reset_state()
        Resets the internal filter state. Only relevant when real_time_mode=True.
    """

    def __init__(
        self,
        input_is_chunked: bool,
        is_output: bool = False,
        name: str | None = None,
        run_checks: bool = True,
        *,
        sos_filter_coefficients: tuple[np.ndarray, np.ndarray | float, np.ndarray],
        forwards_and_backwards: bool = True,
        overlap: int = None,
        use_continuous_approach: bool = True,
        real_time_mode: bool = False,
    ):
        super().__init__(
            input_is_chunked=input_is_chunked,
            allowed_input_type="both",
            is_output=is_output,
            name=name,
            run_checks=run_checks,
        )

        self.sos_filter_coefficients = sos_filter_coefficients
        self.forwards_and_backwards = forwards_and_backwards
        self.use_continuous_approach = use_continuous_approach
        self.real_time_mode = real_time_mode

        # Determine filter order based on SOS shape
        # SOS matrix is Nx6 where N is the number of second-order sections
        self.filter_order = self.sos_filter_coefficients.shape[0] * 2

        # Set default overlap as filter_order*4 or use provided value
        self.overlap = self.filter_order * 4 if overlap is None else overlap

        # Parameter validation for real-time mode
        if self.real_time_mode:
            if self.forwards_and_backwards:
                raise ValueError(
                    "In real-time mode, forwards_and_backwards must be False as future data is not available."
                )
            if self.use_continuous_approach:
                raise ValueError(
                    "In real-time mode, use_continuous_approach must be False."
                )

            # Initialize real-time filter state and history buffer
            self.reset_state()

        self._filtering_method = sosfiltfilt if self.forwards_and_backwards else sosfilt

    def reset_state(self):
        """Reset the filter state for real-time processing.

        This method should be called when starting a new signal or when
        there's a discontinuity in the input data.

        Notes
        -----
        This method only has an effect when real_time_mode=True.
        It resets the internal filter state (_zi) and history buffer to None.
        """
        if self.real_time_mode:
            # Initialize filter state
            self._zi = None
            # Initialize history buffer for overlap
            self._history_buffer = None

    def _filter(self, input_array: np.ndarray) -> np.ndarray:
        if not self.input_is_chunked:
            # Simply apply the filter to the non-chunked data
            return self._filtering_method(self.sos_filter_coefficients, input_array)

        # Handle real-time mode (specifically designed for streaming data)
        if self.real_time_mode:
            return self._filter_real_time(input_array)

        # Handle chunked data in batch mode
        original_shape = input_array.shape

        if self.use_continuous_approach and original_shape[0] > 1:
            # Approach 1: Reconstruct continuous signal, filter it, then reshape back to chunks

            # First, handle multi-dimensional case properly
            num_chunks = original_shape[0]
            chunk_size = original_shape[-1]

            # Save original intermediate dimensions
            intermediate_dims = original_shape[1:-1] if len(original_shape) > 2 else ()

            # Reshape to handle each dimension separately if there are intermediate dimensions
            if intermediate_dims:
                # Reshape to (num_chunks, -1, chunk_size) where -1 combines all intermediate dimensions
                flat_intermediate_size = np.prod(intermediate_dims)
                reshaped_array = input_array.reshape(
                    num_chunks, flat_intermediate_size, chunk_size
                )

                # Initialize output array to hold filtered data
                filtered_result = np.zeros_like(reshaped_array)

                # Loop through each intermediate dimension
                for dim_idx in range(flat_intermediate_size):
                    # Extract data for this dimension and concatenate all chunks
                    dim_data = reshaped_array[:, dim_idx, :].reshape(-1)

                    # Apply filter to continuous data
                    filtered_dim_data = self._filtering_method(
                        self.sos_filter_coefficients, dim_data
                    )

                    # Reshape back to chunks and store
                    filtered_result[:, dim_idx, :] = filtered_dim_data.reshape(
                        num_chunks, chunk_size
                    )

                # Reshape back to original dimensions
                return filtered_result.reshape(original_shape)
            else:
                # Simple case: just one dimension, so we can flatten and reshape directly
                # Flatten the chunked dimension to get a continuous signal
                flattened = input_array.reshape(-1)

                # Apply the filter to the continuous data
                filtered = self._filtering_method(
                    self.sos_filter_coefficients, flattened
                )

                # Reshape back to the original chunked shape
                return filtered.reshape(original_shape)

        else:
            # Approach 2: Use overlap between chunks if continuous approach not possible
            # If overlap is zero or signal is too short, use the original method
            if self.overlap <= 0 or original_shape[-1] <= self.overlap * 2:
                return self._filtering_method(self.sos_filter_coefficients, input_array)

            # Get number of chunks and chunk size
            num_chunks, chunk_size = original_shape[0], original_shape[-1]

            # Initialize output array
            output_array = np.zeros_like(input_array)

            # Handle first chunk (no left padding needed)
            chunk = input_array[0]
            # Pad right with data from next chunk if available
            if num_chunks > 1:
                padded_chunk = np.concatenate(
                    [chunk, input_array[1, : self.overlap]], axis=-1
                )
            else:
                padded_chunk = chunk  # No next chunk, use as is

            # Filter and trim
            filtered_chunk = self._filtering_method(
                self.sos_filter_coefficients, padded_chunk
            )
            output_array[0] = (
                filtered_chunk[:chunk_size] if num_chunks > 1 else filtered_chunk
            )

            # Process middle chunks (needs both left and right padding)
            for i in range(1, num_chunks - 1):
                # Get current chunk with overlap from previous and next chunks
                left_pad = input_array[i - 1, -self.overlap :]
                current = input_array[i]
                right_pad = input_array[i + 1, : self.overlap]

                padded_chunk = np.concatenate([left_pad, current, right_pad], axis=-1)

                # Filter and trim off the padding
                filtered_chunk = self._filtering_method(
                    self.sos_filter_coefficients, padded_chunk
                )
                output_array[i] = filtered_chunk[self.overlap : -self.overlap]

            # Handle last chunk (if more than one chunk exists)
            if num_chunks > 1:
                # Pad left with data from previous chunk
                padded_chunk = np.concatenate(
                    [input_array[-2, -self.overlap :], input_array[-1]], axis=-1
                )

                # Filter and trim
                filtered_chunk = self._filtering_method(
                    self.sos_filter_coefficients, padded_chunk
                )
                output_array[-1] = filtered_chunk[self.overlap :]

            return output_array

    def _filter_real_time(self, input_array: np.ndarray) -> np.ndarray:
        """Process input in real-time mode with state preservation between calls.

        Parameters
        ----------
        input_array : numpy.ndarray
            The newest chunk(s) of data to process.

        Returns
        -------
        numpy.ndarray
            Filtered data of the same shape as input.

        Notes
        -----
        This method is designed for streaming applications and maintains filter
        state between calls using the internal filter state variables.
        """
        original_shape = input_array.shape

        # Extract dimensions and handle multi-dimensional case
        have_multiple_chunks = len(original_shape) > 1 and original_shape[0] > 1

        if have_multiple_chunks:
            # Process each chunk sequentially to maintain state
            output_array = np.zeros_like(input_array)
            for i in range(original_shape[0]):
                chunk = input_array[i]
                filtered_chunk = self._filter_real_time_single_chunk(chunk)
                output_array[i] = filtered_chunk
            return output_array
        else:
            # Single chunk case
            if len(original_shape) > 1:  # Single chunk but has batch dimension
                chunk = input_array[0]
                filtered = self._filter_real_time_single_chunk(chunk)
                return filtered.reshape(1, *filtered.shape)
            else:  # Just raw data without batch dimension
                return self._filter_real_time_single_chunk(input_array)

    def _filter_real_time_single_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Process a single chunk in real-time mode, maintaining filter state between calls.

        Parameters
        ----------
        chunk : numpy.ndarray
            A single chunk of data to process.

        Returns
        -------
        numpy.ndarray
            Filtered chunk with the same shape as input.

        Notes
        -----
        This method handles both 1D and multi-dimensional arrays by reshaping
        as needed and applying the filter while preserving state.
        """
        # Handle multi-dimensional data
        original_shape = chunk.shape
        chunk_size = original_shape[-1]

        # Handle multi-dimensional arrays (reshape to 2D: (channels, samples))
        if len(original_shape) > 1:
            # Flatten all dimensions except the last one (time)
            flat_channels = np.prod(original_shape[:-1])
            reshaped_chunk = chunk.reshape(flat_channels, chunk_size)

            # Process each channel separately
            filtered_result = np.zeros_like(reshaped_chunk)

            for ch in range(flat_channels):
                # Initialize filter state for this channel if needed
                if self._zi is None:
                    self._zi = np.zeros(
                        (len(self.sos_filter_coefficients), 2, flat_channels)
                    )

                # Apply filter with state
                filtered_data, self._zi[:, :, ch] = sosfilt(
                    self.sos_filter_coefficients,
                    reshaped_chunk[ch],
                    zi=self._zi[:, :, ch],
                )

                filtered_result[ch] = filtered_data

            # Reshape back to original dimensions
            return filtered_result.reshape(original_shape)
        else:
            # Simple 1D case
            if self._zi is None:
                self._zi = np.zeros((len(self.sos_filter_coefficients), 2))

            # Apply filter with state
            filtered_data, self._zi = sosfilt(
                self.sos_filter_coefficients, chunk, zi=self._zi
            )

            return filtered_data


class RectifyFilter(ApplyFunctionFilter):
    """Filter that rectifies the input array.

    Parameters
    ----------
    input_is_chunked : bool
        Whether the input is chunked or not.
    is_output : bool
        Whether the filter is an output filter. If True, the resulting signal will be outputted by and dataset pipeline.
    name : str | None
        Name of the filter, by default None.
    run_checks : bool
        Whether to run the checks when filtering. By default, True. If False can potentially speed up performance.

        .. warning:: If False, the user is responsible for ensuring that the input array is valid.
    """

    def __init__(
        self,
        input_is_chunked: bool,
        is_output: bool = False,
        name: str | None = None,
        run_checks: bool = True,
    ):
        super().__init__(
            input_is_chunked=input_is_chunked,
            is_output=is_output,
            name=name,
            run_checks=run_checks,
            function=np.abs,
        )


class WindowedFunctionFilter(ApplyFunctionFilter):
    """Base class for filters that apply a function to windowed data.

    This filter creates windows using _get_windows_with_shift and then applies
    a specified function to each window.

    Parameters
    ----------
    input_is_chunked : bool
        Whether the input is chunked or not.
    is_output : bool
        Whether the filter is an output filter.
    name : str | None
        Name of the filter, by default None.
    run_checks : bool
        Whether to run the checks when filtering. By default, True. If False can potentially speed up performance.

        .. warning:: If False, the user is responsible for ensuring that the input array is valid.

    window_size : int
        The window size to use.
    shift : int
        The shift to use. Default is 1.
    window_function : callable
        Function to apply to each window (along the last axis).

        .. note:: The function should take two arguments: the window and the axis to apply the function along.
    """

    def __init__(
        self,
        input_is_chunked: bool,
        is_output: bool = False,
        name: str | None = None,
        run_checks: bool = True,
        *,
        window_size: int,
        shift: int = 1,
        window_function: callable,
    ):
        # Validate parameters
        if window_size < 1:
            raise ValueError("window_size must be greater than 0.")
        if shift < 1:
            raise ValueError("shift must be greater than 0.")

        # Define the windowed function application
        def apply_window_function(x, window_size, shift, window_function):
            # Get windows if not already chunked
            windowed_array = _get_windows_with_shift(x, window_size, shift)
            # Apply the function to each window
            return np.transpose(
                np.squeeze(window_function(windowed_array, axis=-1), axis=-1), (1, 2, 0)
            )

        # Initialize parent with the windowed function
        super().__init__(
            input_is_chunked=input_is_chunked,
            is_output=is_output,
            name=name,
            run_checks=run_checks,
            function=apply_window_function,
            window_size=window_size,
            shift=shift,
            window_function=window_function,
        )

        # Store parameters for reference
        self.window_size = window_size
        self.shift = shift


class RMSFilter(WindowedFunctionFilter):
    """Filter that computes the root mean squared value [1]_ of the input array.

    Root mean squared value is the square root of the mean of the squared values of the input array.
    It is a measure of the magnitude of the signal.

    .. math::
        \\text{RMS} = \\sqrt{\\frac{1}{N} \\sum_{i=1}^{N} x_i^2}

    Parameters
    ----------
    input_is_chunked : bool
        Whether the input is chunked or not.
    is_output : bool
        Whether the filter is an output filter.
    name : str | None
        Name of the filter, by default None.
    run_checks : bool
        Whether to run the checks when filtering. By default, True. If False can potentially speed up performance.

        .. warning:: If False, the user is responsible for ensuring that the input array is valid.

    window_size : int
        The window size to use.
    shift : int
        The shift to use. Default is 1.
    stabilization_factor : float
        A small value to add to the squared values before taking the mean to stabilize the computation.
        By default, this is the machine epsilon for float values. See numpy.finfo(float).eps.

    References
    ----------
    .. [1] https://doi.org/10.1080/10255842.2023.2165068
    """

    def __init__(
        self,
        input_is_chunked: bool,
        is_output: bool = False,
        name: str | None = None,
        run_checks: bool = True,
        *,
        window_size: int,
        shift: int = 1,
        stabilization_factor: float = np.finfo(float).eps,
    ):
        super().__init__(
            input_is_chunked=input_is_chunked,
            is_output=is_output,
            name=name,
            run_checks=run_checks,
            window_size=window_size,
            shift=shift,
            window_function=lambda x, axis: np.sqrt(
                np.mean(np.square(x), axis=axis, keepdims=True) + stabilization_factor
            ),
        )


class VARFilter(WindowedFunctionFilter):
    """Filter that computes the variance [1]_ of the input array.

    Variance is the average of the squared differences from the mean.
    It is a measure of the spread of the signal.

    .. math::
        \\text{VAR} = \\frac{1}{N} \\sum_{i=1}^{N} (x_i - \\mu)^2

    Parameters
    ----------
    input_is_chunked : bool
        Whether the input is chunked or not.
    is_output : bool
        Whether the filter is an output filter.
    name : str | None
        Name of the filter, by default None.
    run_checks : bool
        Whether to run the checks when filtering. By default, True. If False can potentially speed up performance.

        .. warning:: If False, the user is responsible for ensuring that the input array is valid.

    window_size : int
        The window size to use.
    shift : int
        The shift to use. Default is 1.

    References
    ----------
    .. [1] https://doi.org/10.1080/10255842.2023.2165068
    """

    def __init__(
        self,
        input_is_chunked: bool,
        is_output: bool = False,
        name: str | None = None,
        run_checks: bool = True,
        *,
        window_size: int,
        shift: int = 1,
    ):
        # Initialize with variance function
        super().__init__(
            input_is_chunked=input_is_chunked,
            is_output=is_output,
            name=name,
            run_checks=run_checks,
            window_size=window_size,
            shift=shift,
            window_function=np.var,
        )


class MAVFilter(WindowedFunctionFilter):
    """Filter that computes the mean absolute value [1]_ of the input array.

    Mean absolute value is the average of the absolute values of the input array.
    It is a measure of the average magnitude of the signal.

    .. math::
        \\text{MAV} = \\frac{1}{N} \\sum_{i=1}^{N} |x_i|

    Parameters
    ----------
    input_is_chunked : bool
        Whether the input is chunked or not.
    is_output : bool
        Whether the filter is an output filter. If True, the resulting signal will be outputted by and dataset pipeline.
    name : str | None
        Name of the filter, by default None.
    run_checks : bool
        Whether to run the checks when filtering. By default, True. If False can potentially speed up performance.

        .. warning:: If False, the user is responsible for ensuring that the input array is valid.

    window_size : int
        The window size to use.
    shift : int
        The shift to use. Default is 1.

    References
    ----------
    .. [1] https://doi.org/10.1080/10255842.2023.2165068
    """

    def __init__(
        self,
        input_is_chunked: bool,
        is_output: bool = False,
        name: str | None = None,
        run_checks: bool = True,
        *,
        window_size: int,
        shift: int = 1,
    ):
        super().__init__(
            input_is_chunked=input_is_chunked,
            is_output=is_output,
            name=name,
            run_checks=run_checks,
            window_size=window_size,
            shift=shift,
            window_function=lambda x, axis: np.mean(
                np.abs(x), axis=axis, keepdims=True
            ),
        )


class IAVFilter(WindowedFunctionFilter):
    """Filter that computes the integrated absolute value [1]_ of the input array.

    Integrated absolute value is the sum of the absolute values of the input array.
    It is a measure of the total magnitude of the signal.

    .. math::
        \\text{IAV} = \\sum_{i=1}^{N} |x_i|

    Parameters
    ----------
    input_is_chunked : bool
        Whether the input is chunked or not.
    is_output : bool
        Whether the filter is an output filter. If True, the resulting signal will be outputted by and dataset pipeline.
    name : str | None
        Name of the filter, by default None.
    run_checks : bool
        Whether to run the checks when filtering. By default, True. If False can potentially speed up performance.

        .. warning:: If False, the user is responsible for ensuring that the input array is valid.

    window_size : int
        The window size to use.
    shift : int
        The shift to use. Default is 1.

    References
    ----------
    .. [1] https://doi.org/10.1080/10255842.2023.2165068
    """

    def __init__(
        self,
        input_is_chunked: bool,
        is_output: bool = False,
        name: str | None = None,
        run_checks: bool = True,
        *,
        window_size: int,
        shift: int = 1,
    ):
        super().__init__(
            input_is_chunked=input_is_chunked,
            is_output=is_output,
            name=name,
            run_checks=run_checks,
            window_size=window_size,
            shift=shift,
            window_function=lambda x, axis: np.sum(np.abs(x), axis=axis, keepdims=True),
        )


class WFLFilter(WindowedFunctionFilter):
    """Filter that computes the waveform length [1]_ of the input array.

    Waveform length is the sum of the absolute differences between consecutive samples.
    It is a measure of the total magnitude of the signal.

    .. math::
        \\text{WFL} = \\sum_{i=1}^{N} |x_i - x_{i-1}|

    Parameters
    ----------
    input_is_chunked : bool
        Whether the input is chunked or not.
    is_output : bool
        Whether the filter is an output filter. If True, the resulting signal will be outputted by and dataset pipeline.
    name : str | None
        Name of the filter, by default None.
    run_checks : bool
        Whether to run the checks when filtering. By default, True. If False can potentially speed up performance.

        .. warning:: If False, the user is responsible for ensuring that the input array is valid.

    window_size : int
        The window size to use.
    shift : int
        The shift to use. Default is 1.

    References
    ----------
    .. [1] https://doi.org/10.1080/10255842.2023.2165068
    """

    def __init__(
        self,
        input_is_chunked: bool,
        is_output: bool = False,
        name: str | None = None,
        run_checks: bool = True,
        *,
        window_size: int,
        shift: int = 1,
    ):
        super().__init__(
            input_is_chunked=input_is_chunked,
            is_output=is_output,
            name=name,
            run_checks=run_checks,
            window_size=window_size,
            shift=shift,
            window_function=lambda x, axis: np.sum(
                np.abs(np.diff(x, axis=axis)), axis=axis, keepdims=True
            ),
        )


class ZCFilter(WindowedFunctionFilter):
    """Computes the zero crossings [1]_ of the input array.

    Zero crossings are the number of times the signal crosses the zero axis.
    It is a measure of the number of times the signal changes sign.

    .. math::
        \\text{ZC} = \\sum_{i=1}^{N} \\frac{1}{2} |\\text{sign}(x_i) - \\text{sign}(x_{i-1})|

    Parameters
    ----------
    input_is_chunked : bool
        Whether the input is chunked or not.
    is_output : bool
        Whether the filter is an output filter.
    name : str | None
        Name of the filter, by default None.
    run_checks : bool
        Whether to run the checks when filtering. By default, True. If False can potentially speed up performance.

        .. warning:: If False, the user is responsible for ensuring that the input array is valid.

    window_size : int
        The window size to use.
    shift : int
        The shift to use. Default is 1.

    References
    ----------
    .. [1] https://doi.org/10.1080/10255842.2023.2165068
    """

    def __init__(
        self,
        input_is_chunked: bool,
        is_output: bool = False,
        name: str | None = None,
        run_checks: bool = True,
        *,
        window_size: int,
        shift: int = 1,
    ):
        # Define Zero Crossing function
        def zc_function(windowed_array, axis=-1):
            # 1. Calculate sign of all elements in each window
            signs = np.sign(windowed_array)

            # 2. Calculate differences of signs to find changes
            sign_changes = np.diff(signs, axis=axis)

            # 3. Count absolute changes (divided by 2 since each crossing counts twice)
            return np.sum(np.abs(sign_changes) // 2, axis=axis, keepdims=True)

        super().__init__(
            input_is_chunked=input_is_chunked,
            is_output=is_output,
            name=name,
            run_checks=run_checks,
            window_size=window_size,
            shift=shift,
            window_function=zc_function,
        )


class SSCFilter(WindowedFunctionFilter):
    """Computes the slope sign change [1]_ of the input array.

    Slope sign change is the number of times the slope of the signal changes sign.
    It is a measure of the number of times the signal changes direction.

    .. math::
        \\text{SSC} = \\sum_{i=1}^{N} \\frac{1}{2} |\\text{sign}(x_i - x_{i-1}) - \\text{sign}(x_{i-1} - x_{i-2})|

    Parameters
    ----------
    input_is_chunked : bool
        Whether the input is chunked or not.
    is_output : bool
        Whether the filter is an output filter.
    name : str | None
        Name of the filter, by default None.
    run_checks : bool
        Whether to run the checks when filtering. By default, True. If False can potentially speed up performance.

        .. warning:: If False, the user is responsible for ensuring that the input array is valid.

    window_size : int
        The window size to use.
    shift : int
        The shift to use. Default is 1.

    References
    ----------
    .. [1] https://doi.org/10.1080/10255842.2023.2165068
    """

    def __init__(
        self,
        input_is_chunked: bool,
        is_output: bool = False,
        name: str | None = None,
        run_checks: bool = True,
        *,
        window_size: int,
        shift: int = 1,
    ):
        # Define Slope Sign Change function
        def ssc_function(windowed_array, axis=-1):
            # 1. Calculate differences (first derivative)
            diffs = np.diff(windowed_array, axis=axis)

            # 2. Calculate sign of differences
            sign_diffs = np.sign(diffs)

            # 3. Calculate sign changes of the sign differences (second derivative sign changes)
            sign_changes = np.diff(sign_diffs, axis=axis)

            # 4. Count number of sign changes in each window
            return np.sum(np.abs(sign_changes) // 2, axis=axis, keepdims=True)

        super().__init__(
            input_is_chunked=input_is_chunked,
            is_output=is_output,
            name=name,
            run_checks=run_checks,
            window_size=window_size,
            shift=shift,
            window_function=ssc_function,
        )


class SpectralInterpolationFilter(FilterBaseClass):
    """Filter that removes certain frequency bands from the signal and interpolates the gaps.

    This is ideal for removing power line interference or other narrowband noise.
    The filter works by:
    1. Computing FFT of the signal
    2. Setting the magnitude of the specified frequency bands to interpolated values
    3. Preserving the phase information
    4. Converting back to time domain

    .. warning:: When used for real-time applications, performance depends on chunk size.
                 Smaller chunks reduce latency but may decrease frequency resolution and
                 interpolation quality, especially for narrow frequency bands.


    Parameters
    ----------
    input_is_chunked : bool
        Whether the input is chunked or not.
    is_output : bool
        Whether the filter is an output filter.
    name : str | None
        Name of the filter, by default None.
    run_checks : bool
        Whether to run the checks when filtering. By default, True. If False can potentially speed up performance.

        .. warning:: If False, the user is responsible for ensuring that the input array is valid.

    bandwidth : tuple[float, float]
        Frequency band to remove (min_freq, max_freq) in Hz, by default (47.5, 52.5)
    number_of_harmonics : int
        Number of harmonics to remove, by default 3
    sampling_frequency : float
        Sampling frequency in Hz, by default 2000
    interpolation_window : int
        Window size for interpolation, must be an odd number, by default 15
    interpolation_poly_order : int
        Polynomial order for interpolation, must be less than interpolation_window, by default 3
    remove_dc : bool
        Whether to remove the DC component (set FFT[0] to 0), by default True
    """

    def __init__(
        self,
        input_is_chunked: bool,
        is_output: bool = False,
        name: str | None = None,
        run_checks: bool = True,
        *,
        bandwidth: tuple[float, float] = (47.5, 52.5),
        number_of_harmonics: int = 3,
        sampling_frequency: float = 2000,
        interpolation_window: int = 15,
        interpolation_poly_order: int = 3,
        remove_dc: bool = True,
    ):
        super().__init__(
            input_is_chunked=input_is_chunked,
            allowed_input_type="both",
            is_output=is_output,
            name=name,
            run_checks=run_checks,
        )
        self.bandwidth = bandwidth
        self.number_of_harmonics = number_of_harmonics
        self.sampling_frequency = sampling_frequency

        # Validate interpolation parameters
        if interpolation_window % 2 == 0:
            raise ValueError("interpolation_window must be an odd number")
        if interpolation_poly_order >= interpolation_window:
            raise ValueError(
                "interpolation_poly_order must be less than interpolation_window"
            )

        self.interpolation_window = interpolation_window
        self.interpolation_poly_order = interpolation_poly_order
        self.remove_dc = remove_dc

        # Pre-compute harmonic frequencies
        center_freq = (bandwidth[0] + bandwidth[1]) / 2
        self.harmonic_freqs = [
            (i * center_freq, i * bandwidth[0], i * bandwidth[1])
            for i in range(1, number_of_harmonics + 1)
        ]

    def _get_indices_to_interpolate(self, freqs):
        """Get the indices of the frequency bins to interpolate.

        Parameters
        ----------
        freqs : numpy.ndarray
            Frequency bins from rfftfreq

        Returns
        -------
        list
            List of arrays, each containing indices for a harmonic frequency band
        """
        indices_list = []
        for _, min_freq, max_freq in self.harmonic_freqs:
            indices = np.where((freqs >= min_freq) & (freqs <= max_freq))[0]
            indices_list.append(indices)
        return indices_list

    def _filter(self, input_array):
        """Apply the filter to the input array.

        Parameters
        ----------
        input_array : numpy.ndarray
            Input array to filter

        Returns
        -------
        numpy.ndarray
            Filtered array
        """
        # Save original shape
        original_shape = input_array.shape

        # For multidimensional arrays, reshape to 2D for easier processing
        if len(original_shape) > 1:
            reshaped_array = input_array.reshape(-1, original_shape[-1])

            # Process each signal
            for j in range(reshaped_array.shape[0]):
                # Compute the FFT
                signal_fft = rfft(reshaped_array[j], axis=-1)

                # Set the DC component to zero (optional)
                if self.remove_dc:
                    signal_fft[0] = 0

                # Calculate frequency bins
                freqs = rfftfreq(original_shape[-1], d=1 / self.sampling_frequency)

                # Get interpolation indices for each harmonic's frequency band
                for indices in self._get_indices_to_interpolate(freqs):
                    if len(indices) > 0:
                        # Get magnitude and phase
                        magnitude = np.abs(signal_fft)
                        phase = np.angle(signal_fft)

                        # Determine appropriate window size based on available data
                        window_size = min(self.interpolation_window, len(magnitude) - 2)
                        if window_size % 2 == 0:
                            window_size -= 1  # Ensure it's odd

                        # Ensure polynomial order is appropriate for window size
                        poly_order = min(self.interpolation_poly_order, window_size - 1)

                        # Use 'nearest' mode if data is too short for 'interp'
                        filter_mode = (
                            "nearest" if window_size >= len(magnitude) else "interp"
                        )

                        # Apply savgol_filter to get a smooth interpolation, with adjusted parameters
                        if (
                            window_size >= 3
                        ):  # Minimum window size for Savitzky-Golay filter
                            smooth_magnitude = savgol_filter(
                                magnitude,
                                window_size,
                                poly_order,
                                axis=-1,
                                mode=filter_mode,
                            )

                            # Replace the magnitude in the specified indices while preserving phase
                            signal_fft[indices] = smooth_magnitude[indices] * np.exp(
                                1j * phase[indices]
                            )

                # Convert back to time domain
                reshaped_array[j] = irfft(signal_fft, n=original_shape[-1], axis=-1)

            # Reshape back to original dimensions
            output_array = reshaped_array.reshape(original_shape)
        else:
            # Simple case: just one dimension
            freqs = rfftfreq(original_shape[-1], d=1 / self.sampling_frequency)

            # Compute the FFT
            signal_fft = rfft(input_array, axis=-1)

            # Set the DC component to zero (optional)
            if self.remove_dc:
                signal_fft[0] = 0

            # Get interpolation indices for each harmonic's frequency band
            for indices in self._get_indices_to_interpolate(freqs):
                if len(indices) > 0:
                    # Get magnitude and phase
                    magnitude = np.abs(signal_fft)
                    phase = np.angle(signal_fft)

                    # Determine appropriate window size based on available data
                    window_size = min(self.interpolation_window, len(magnitude) - 2)
                    if window_size % 2 == 0:
                        window_size -= 1  # Ensure it's odd

                    # Ensure polynomial order is appropriate for window size
                    poly_order = min(self.interpolation_poly_order, window_size - 1)

                    # Use 'nearest' mode if data is too short for 'interp'
                    filter_mode = (
                        "nearest" if window_size >= len(magnitude) else "interp"
                    )

                    # Apply savgol_filter to get a smooth interpolation, with adjusted parameters
                    if (
                        window_size >= 3
                    ):  # Minimum window size for Savitzky-Golay filter
                        smooth_magnitude = savgol_filter(
                            magnitude,
                            window_size,
                            poly_order,
                            axis=-1,
                            mode=filter_mode,
                        )

                        # Replace the magnitude in the specified indices while preserving phase
                        signal_fft[indices] = smooth_magnitude[indices] * np.exp(
                            1j * phase[indices]
                        )

            # Convert back to time domain
            output_array = irfft(signal_fft, n=original_shape[-1], axis=-1)

        return output_array
