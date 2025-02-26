from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.fft import irfft, rfft, rfftfreq
from scipy.signal import savgol_filter, sosfilt, sosfiltfilt

from myoverse.datasets.filters._template import FilterBaseClass
from myoverse.datasets.filters.generic import ApplyFunctionFilter


def _get_windows_with_shift(
    input_array: np.ndarray, window_size: int, shift: int
) -> np.ndarray:
    """Create windows of specified size and shift from input array using strided operations.

    Parameters
    ----------
    input_array : numpy.ndarray
        The input array to window.
    window_size : int
        Size of each window.
    shift : int
        Number of samples to shift between consecutive windows.

    Returns
    -------
    numpy.ndarray
        Array of windows with shape (*input_array.shape[:-1], n_windows, window_size)
        where n_windows = (input_array.shape[-1] - window_size) // shift + 1

    Notes
    -----
    This function uses numpy's as_strided function for efficient windowing without
    creating copies of the data. The returned array is read-only (writeable=False).
    """
    # Calculate how many windows we'll have
    n_windows = (input_array.shape[-1] - window_size) // shift + 1

    # Calculate new shape with windows
    # Original dimensions (except the last) + number of windows + window size
    window_shape = (*input_array.shape[:-1], n_windows, window_size)

    # Calculate strides for the windowed view
    # Original strides + stride for windows + original stride for last dimension
    original_strides = input_array.strides
    window_strides = (
        *original_strides[:-1],
        shift * original_strides[-1],  # Step between windows
        original_strides[-1],
    )  # Step within a window

    # Create windowed view using as_strided
    return as_strided(
        input_array, shape=window_shape, strides=window_strides, writeable=False
    )


class SOSFrequencyFilter(FilterBaseClass):
    """Filter that applies a second-order-section filter to the input array.

    Parameters
    ----------
    sos_filter_coefficients : np.ndarray
        The second-order-section filter coefficients, typically from scipy.signal.butter with output="sos".
    forwards_and_backwards : bool
        Whether to apply the filter forwards and backwards or only forwards.
    input_is_chunked : bool
        Whether the input is chunked or not.
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
    is_output : bool
        Whether the filter is an output filter. If True, the resulting signal will be outputted by and dataset pipeline.

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
        sos_filter_coefficients: np.ndarray,
        forwards_and_backwards: bool = True,
        input_is_chunked: bool = None,
        overlap: int = None,
        use_continuous_approach: bool = True,
        real_time_mode: bool = False,
        is_output: bool = False,
        name: str = None,
    ):
        super().__init__(
            input_is_chunked=input_is_chunked,
            allowed_input_type="both",
            is_output=is_output,
            name=name,
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
    """

    def __init__(self, input_is_chunked: bool = None, is_output: bool = False):
        super().__init__(
            input_is_chunked=input_is_chunked, function=np.abs, is_output=is_output
        )


class RMSFilter(FilterBaseClass):
    """Filter that computes the root mean squared value [1]_ of the input array.

    Parameters
    ----------
    window_size : int
        The window size to use.
    shift : int
        The shift to use.
    input_is_chunked : bool
        Whether the input is chunked or not.

    Methods
    -------
    __call__(input_array: np.ndarray) -> np.ndarray
        Filters the input array. Input shape is determined by whether the allowed_input_type
        is "both", "chunked" or "not chunked".

    References
    ----------
    .. [1] https://doi.org/10.1080/10255842.2023.2165068
    """

    def __init__(
        self,
        window_size: int,
        shift: int = 1,
        input_is_chunked: bool = None,
        is_output: bool = False,
        name: str = None,
    ):
        super().__init__(
            input_is_chunked=input_is_chunked,
            allowed_input_type="both",
            is_output=is_output,
            name=name,
        )

        self.window_size = window_size
        self.shift = shift

        if self.window_size < 1:
            raise ValueError("window_size must be greater than 0.")
        if self.shift < 1:
            raise ValueError("shift must be greater than 0.")

    def _filter(self, input_array: np.ndarray) -> np.ndarray:
        windowed_array = _get_windows_with_shift(
            input_array, self.window_size, self.shift
        )

        # Calculate RMS for each window
        return np.sqrt(np.mean(np.square(windowed_array), axis=-1))


class VARFilter(FilterBaseClass):
    """Computes the Variance with given window length and window shift over the input signal.

    Parameters
    ----------
    window_size : int
        The window size to use.
    shift : int, optional
        The shift to use, by default 1.
    input_is_chunked : bool, optional
        Whether the input is chunked or not, by default True.
    is_output : bool, optional
        Whether the filter is an output filter. If True, the resulting signal will be outputted by the dataset pipeline, by default False.
    name : str, optional
        The name of the filter, by default None.

    References
    ----------
    .. [1] https://doi.org/10.1080/10255842.2023.2165068
    """

    def __init__(
        self,
        window_size: int,
        shift: int = 1,
        input_is_chunked: bool = True,
        is_output: bool = False,
        name: str = None,
    ):
        super().__init__(
            input_is_chunked=input_is_chunked,
            allowed_input_type="both",
            is_output=is_output,
            name=name,
        )
        self.window_size = window_size
        self.shift = shift

        if self.window_size < 1:
            raise ValueError("window_size must be greater than 0.")
        if self.shift < 1:
            raise ValueError("shift must be greater than 0.")

    def _filter(self, input_array: np.ndarray) -> np.ndarray:
        # Use the optimized window function instead of list comprehension
        windowed_array = _get_windows_with_shift(
            input_array, self.window_size, self.shift
        )

        # Calculate variance for each window
        return np.var(windowed_array, axis=-1)


class MAVFilter(FilterBaseClass):
    """Computes the Mean Absolute Value with given window length and window shift over the input signal. See formula in
    the following paper: https://doi.org/10.1080/10255842.2023.2165068.

    Parameters
    ----------
    window_size : int
        The window size to use.
    shift : int
        The shift to use.
    input_is_chunked : bool
        Whether the input is chunked or not.
    is_output : bool
        Whether the filter is an output filter. If True, the resulting signal will be outputted by and dataset pipeline.
    name : str
        The name of the filter.
    """

    def __init__(
        self,
        window_size: int,
        shift: int = 1,
        input_is_chunked: bool = True,
        is_output: bool = False,
        name: str = None,
    ):
        super().__init__(
            input_is_chunked=input_is_chunked,
            allowed_input_type="both",
            is_output=is_output,
            name=name,
        )
        self.window_size = window_size
        self.shift = shift

        if self.window_size < 1:
            raise ValueError("window_size must be greater than 0.")
        if self.shift < 1:
            raise ValueError("shift must be greater than 0.")

    def _filter(self, input_array: np.ndarray) -> np.ndarray:
        # Use the optimized window function instead of list comprehension
        windowed_array = _get_windows_with_shift(
            input_array, self.window_size, self.shift
        )

        # Calculate mean absolute value for each window
        return np.mean(np.abs(windowed_array), axis=-1)


class IAVFilter(FilterBaseClass):
    """Computes the Integrated Absolute Value with given window length and window shift over the input signal. See
    formula in the following paper: https://doi.org/10.1080/10255842.2023.2165068.

    Parameters
    ----------
    window_size : int
        The window size to use.
    shift : int
        The shift to use.
    input_is_chunked : bool
        Whether the input is chunked or not.
    is_output : bool
        Whether the filter is an output filter. If True, the resulting signal will be outputted by and dataset pipeline.
    name : str
        The name of the filter.
    """

    def __init__(
        self,
        window_size: int,
        shift: int = 1,
        input_is_chunked: bool = True,
        is_output: bool = False,
        name: str = None,
    ):
        super().__init__(
            input_is_chunked=input_is_chunked,
            allowed_input_type="both",
            is_output=is_output,
            name=name,
        )
        self.window_size = window_size
        self.shift = shift

        if self.window_size < 1:
            raise ValueError("window_size must be greater than 0.")
        if self.shift < 1:
            raise ValueError("shift must be greater than 0.")

    def _filter(self, input_array: np.ndarray) -> np.ndarray:
        # Use the optimized window function instead of list comprehension
        windowed_array = _get_windows_with_shift(
            input_array, self.window_size, self.shift
        )

        # Calculate integrated absolute value (sum of absolute values) for each window
        return np.sum(np.abs(windowed_array), axis=-1)


class WFLFilter(FilterBaseClass):
    """Computes the Waveform Length with given window length and window shift over the input signal. See
    formula in the following paper: https://doi.org/10.1080/10255842.2023.2165068.

    Parameters
    ----------
    window_size : int
        The window size to use.
    shift : int
        The shift to use.
    input_is_chunked : bool
        Whether the input is chunked or not.
    is_output : bool
        Whether the filter is an output filter. If True, the resulting signal will be outputted by and dataset pipeline.
    name : str
        The name of the filter.
    """

    def __init__(
        self,
        window_size: int,
        shift: int = 1,
        input_is_chunked: bool = True,
        is_output: bool = False,
        name: str = None,
    ):
        super().__init__(
            input_is_chunked=input_is_chunked,
            allowed_input_type="both",
            is_output=is_output,
            name=name,
        )
        self.window_size = window_size
        self.shift = shift

        if self.window_size < 1:
            raise ValueError("window_size must be greater than 0.")
        if self.shift < 1:
            raise ValueError("shift must be greater than 0.")

    def _filter(self, input_array: np.ndarray) -> np.ndarray:
        # Use the optimized window function
        windowed_array = _get_windows_with_shift(
            input_array, self.window_size, self.shift
        )

        # Calculate waveform length (sum of absolute differences) for each window
        # First compute differences between consecutive samples in each window
        diffs = np.diff(windowed_array, axis=-1)

        # Then sum the absolute differences for each window
        return np.sum(np.abs(diffs), axis=-1)


class ZCFilter(FilterBaseClass):
    """Computes the Zero Crossings with given window length and window shift over the input signal. See formula in the
    following paper: https://doi.org/10.1080/10255842.2023.2165068.

    Parameters
    ----------
    window_size : int
        The window size to use.
    shift : int
        The shift to use.
    input_is_chunked : bool
        Whether the input is chunked or not.
    is_output : bool
        Whether the filter is an output filter. If True, the resulting signal will be outputted by and dataset pipeline.
    name : str
        The name of the filter.
    """

    def __init__(
        self,
        window_size: int,
        shift: int = 1,
        input_is_chunked: bool = True,
        is_output: bool = False,
        name: str = None,
    ):
        super().__init__(
            input_is_chunked=input_is_chunked,
            allowed_input_type="both",
            is_output=is_output,
            name=name,
        )
        self.window_size = window_size
        self.shift = shift

        if self.window_size < 1:
            raise ValueError("window_size must be greater than 0.")
        if self.shift < 1:
            raise ValueError("shift must be greater than 0.")

    def _filter(self, input_array: np.ndarray) -> np.ndarray:
        # Use the optimized window function
        windowed_array = _get_windows_with_shift(
            input_array, self.window_size, self.shift
        )

        # Calculate zero crossings for each window
        # 1. Calculate sign of all elements in each window
        signs = np.sign(windowed_array)

        # 2. Calculate differences of signs to find changes
        sign_changes = np.diff(signs, axis=-1)

        # 3. Count absolute changes (divided by 2 since each crossing counts twice)
        return np.sum(np.abs(sign_changes) // 2, axis=-1)


class SSCFilter(FilterBaseClass):
    """Computes the Slope Sign Change with given window length and window shift over the input signal. See formula in
    the following paper: https://doi.org/10.1080/10255842.2023.2165068.

    Parameters
    ----------
    window_size : int
        The window size to use.
    shift : int
        The shift to use.
    input_is_chunked : bool
        Whether the input is chunked or not.
    is_output : bool
        Whether the filter is an output filter. If True, the resulting signal will be outputted by and dataset pipeline.
    name : str
        The name of the filter.
    """

    def __init__(
        self,
        window_size: int,
        shift: int = 1,
        input_is_chunked: bool = True,
        is_output: bool = False,
        name: str = None,
    ):
        super().__init__(
            input_is_chunked=input_is_chunked,
            allowed_input_type="both",
            is_output=is_output,
            name=name,
        )
        self.window_size = window_size
        self.shift = shift

        if self.window_size < 1:
            raise ValueError("window_size must be greater than 0.")
        if self.shift < 1:
            raise ValueError("shift must be greater than 0.")

    def _filter(self, input_array: np.ndarray) -> np.ndarray:
        # Use the optimized window function
        windowed_array = _get_windows_with_shift(
            input_array, self.window_size, self.shift
        )

        # Calculate slope sign changes for each window
        # 1. Calculate differences (first derivative)
        diffs = np.diff(windowed_array, axis=-1)

        # 2. Calculate sign of differences
        sign_diffs = np.sign(diffs)

        # 3. Calculate sign changes of the sign differences (second derivative sign changes)
        sign_changes = np.diff(sign_diffs, axis=-1)

        # 4. Count number of sign changes in each window
        return np.sum(np.abs(sign_changes) // 2, axis=-1)


class SpectralInterpolationFilter(FilterBaseClass):
    """Filter that removes certain frequency bands from the signal and interpolates the gaps.

    This is ideal for removing power line interference or other narrowband noise.
    The filter works by:
    1. Computing FFT of the signal
    2. Setting the magnitude of the specified frequency bands to interpolated values
    3. Preserving the phase information
    4. Converting back to time domain

    Parameters
    ----------
    bandwidth : tuple, optional
        Frequency band to remove (min_freq, max_freq) in Hz, by default (47.5, 52.5)
    number_of_harmonics : int, optional
        Number of harmonics to remove, by default 3
    sampling_frequency : float, optional
        Sampling frequency in Hz, by default 2000
    interpolation_window : int, optional
        Window size for interpolation, must be an odd number, by default 15
    interpolation_poly_order : int, optional
        Polynomial order for interpolation, must be less than interpolation_window, by default 3
    input_is_chunked : bool, optional
        Whether the input is chunked, by default True
    representations_to_filter : str, optional
        Representations to filter, by default "all"
    """

    def __init__(
        self,
        bandwidth=(47.5, 52.5),
        number_of_harmonics=3,
        sampling_frequency=2000,
        interpolation_window=15,
        interpolation_poly_order=3,
        input_is_chunked=True,
        representations_to_filter="all",
    ):
        """Initialize the filter.

        Parameters
        ----------
        bandwidth : tuple, optional
            Frequency band to remove (min_freq, max_freq) in Hz, by default (47.5, 52.5)
        number_of_harmonics : int, optional
            Number of harmonics to remove, by default 3
        sampling_frequency : float, optional
            Sampling frequency in Hz, by default 2000
        interpolation_window : int, optional
            Window size for interpolation, must be an odd number, by default 15
        interpolation_poly_order : int, optional
            Polynomial order for interpolation, must be less than interpolation_window, by default 3
        input_is_chunked : bool, optional
            Whether the input is chunked, by default True
        representations_to_filter : str, optional
            Representations to filter, by default "all"
        """
        super().__init__(
            input_is_chunked=input_is_chunked,
            allowed_input_type="both",
            is_output=False,
            name="Spectral Interpolation Filter",
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
