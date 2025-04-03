from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.random.mtrand import Sequence
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
    continuous_approach : tuple[int, int] | None
        If input_is_chunked is True and a tuple of window size and shift is provided, the filter will first unchunk the data, filter, then re-chunk the data.

        .. warning:: Some data **will** be lost at the end of the signal if the window shift is not equal to the window size.

    real_time_support_chunks: np.ndarray | None
        If not None and continuous_approach is not None, the support chunks will be prepended to the input array.
        That way boundary

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
        continuous_approach: tuple[int, int] | None = None,
        real_time_support_chunks: np.ndarray | None = None,
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
        self.continuous_approach = continuous_approach
        self.real_time_support_chunks = real_time_support_chunks

        self._filtering_method = sosfiltfilt if self.forwards_and_backwards else sosfilt


    def _filter(self, input_array: np.ndarray, **kwargs) -> np.ndarray:
        """Apply the filter to the input array.

        Parameters
        ----------
        input_array : numpy.ndarray
            Input array to filter
        **kwargs
            Additional keyword arguments from the Data object

        Returns
        -------
        numpy.ndarray
            Filtered array
        """
        if not self.input_is_chunked:
            # Simply apply the filter to the non-chunked data
            return self._filtering_method(self.sos_filter_coefficients, input_array)

        if self.continuous_approach:
            temp = input_array

            if self.real_time_support_chunks is not None:
                # Prepend the support chunks to the input array
                temp = np.concatenate(
                    (self.real_time_support_chunks, temp), axis=0
                )

            temp = self._filtering_method(self.sos_filter_coefficients, np.concatenate(temp[..., : self.continuous_approach[1]], axis=-1))

            # Reshape the filtered data back to the original shape
            temp = _get_windows_with_shift(temp, *self.continuous_approach)

            if self.real_time_support_chunks is not None:
                # Remove the prepended support chunks
                temp = temp[self.real_time_support_chunks.shape[0] :]

            return temp

        return self._filtering_method(self.sos_filter_coefficients, input_array)



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
            func_result = np.squeeze(window_function(windowed_array, axis=-1), axis=-1)

            return np.transpose(func_result, (*list(range(func_result.ndim))[1:], 0))

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
            window_function=lambda x, axis: np.var(x, axis=axis, keepdims=True),
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

    def _filter(self, input_array, **kwargs):
        """Apply the filter to the input array.

        Parameters
        ----------
        input_array : numpy.ndarray
            Input array to filter
        **kwargs
            Additional keyword arguments from the Data object

        Returns
        -------
        numpy.ndarray
            Filtered array
        """
        # Use sampling_frequency from kwargs if available, otherwise use the instance attribute
        sampling_frequency = kwargs.get("sampling_frequency", self.sampling_frequency)

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
                freqs = rfftfreq(original_shape[-1], d=1 / sampling_frequency)

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
            freqs = rfftfreq(original_shape[-1], d=1 / sampling_frequency)

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
