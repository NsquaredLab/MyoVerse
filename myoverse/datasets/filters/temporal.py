from __future__ import annotations

from typing import List, Literal, Sequence, Tuple, Union

import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.fft import irfft, rfft, rfftfreq
from scipy.signal import savgol_filter, sosfilt, sosfiltfilt
from statsmodels.tsa.ar_model import AutoReg

from myoverse.datasets.filters._template import FilterBaseClass
from myoverse.datasets.filters.generic import ApplyFunctionFilter

def _get_windows_with_shift(
    input_array: np.ndarray, window_size: int, shift: int
) -> np.ndarray:
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
                raise ValueError("In real-time mode, forwards_and_backwards must be False as future data is not available.")
            if self.use_continuous_approach:
                raise ValueError("In real-time mode, use_continuous_approach must be False.")
            
            # Initialize real-time filter state and history buffer
            self.reset_state()
        
        self._filtering_method = sosfiltfilt if self.forwards_and_backwards else sosfilt

    def reset_state(self):
        """Reset the filter state for real-time processing.
        This should be called when starting a new signal or when there's a discontinuity in the input."""
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
                reshaped_array = input_array.reshape(num_chunks, flat_intermediate_size, chunk_size)
                
                # Initialize output array to hold filtered data
                filtered_result = np.zeros_like(reshaped_array)
                
                # Loop through each intermediate dimension
                for dim_idx in range(flat_intermediate_size):
                    # Extract data for this dimension and concatenate all chunks
                    dim_data = reshaped_array[:, dim_idx, :].reshape(-1)
                    
                    # Apply filter to continuous data
                    filtered_dim_data = self._filtering_method(self.sos_filter_coefficients, dim_data)
                    
                    # Reshape back to chunks and store
                    filtered_result[:, dim_idx, :] = filtered_dim_data.reshape(num_chunks, chunk_size)
                
                # Reshape back to original dimensions
                return filtered_result.reshape(original_shape)
            else:
                # Simple case: just one dimension, so we can flatten and reshape directly
                # Flatten the chunked dimension to get a continuous signal
                flattened = input_array.reshape(-1)
                
                # Apply the filter to the continuous data
                filtered = self._filtering_method(self.sos_filter_coefficients, flattened)
                
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
                padded_chunk = np.concatenate([chunk, input_array[1, :self.overlap]], axis=-1)
            else:
                padded_chunk = chunk  # No next chunk, use as is
                
            # Filter and trim
            filtered_chunk = self._filtering_method(self.sos_filter_coefficients, padded_chunk)
            output_array[0] = filtered_chunk[:chunk_size] if num_chunks > 1 else filtered_chunk
            
            # Process middle chunks (needs both left and right padding)
            for i in range(1, num_chunks-1):
                # Get current chunk with overlap from previous and next chunks
                left_pad = input_array[i-1, -self.overlap:]
                current = input_array[i]
                right_pad = input_array[i+1, :self.overlap]
                
                padded_chunk = np.concatenate([left_pad, current, right_pad], axis=-1)
                
                # Filter and trim off the padding
                filtered_chunk = self._filtering_method(self.sos_filter_coefficients, padded_chunk)
                output_array[i] = filtered_chunk[self.overlap:-self.overlap]
            
            # Handle last chunk (if more than one chunk exists)
            if num_chunks > 1:
                # Pad left with data from previous chunk
                padded_chunk = np.concatenate([input_array[-2, -self.overlap:], input_array[-1]], axis=-1)
                
                # Filter and trim
                filtered_chunk = self._filtering_method(self.sos_filter_coefficients, padded_chunk)
                output_array[-1] = filtered_chunk[self.overlap:]
            
            return output_array
            
    def _filter_real_time(self, input_array: np.ndarray) -> np.ndarray:
        """Process input in real-time mode with state preservation between calls.
        Expects input_array to be the newest chunk(s) of data to process.
        Returns filtered data of the same shape as input.
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
        """Process a single chunk in real-time mode, maintaining filter state between calls."""
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
                    self._zi = np.zeros((len(self.sos_filter_coefficients), 2, flat_channels))
                
                # Apply filter with state
                filtered_data, self._zi[:, :, ch] = sosfilt(
                    self.sos_filter_coefficients,
                    reshaped_chunk[ch],
                    zi=self._zi[:, :, ch]
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
                self.sos_filter_coefficients,
                chunk,
                zi=self._zi
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


class ARFilter(FilterBaseClass):
    """Filter that computes n autoregressive coefficients for each window of the input array."""

    def __init__(
        self,
        n_coefficients: int = 4,
        input_is_chunked: bool = True,
        representations_to_filter: Union[Literal["all"], Sequence[int]] = (0,),
    ):
        super().__init__(
            input_is_chunked=input_is_chunked,
            representations_to_filter=representations_to_filter,
        )
        self.n_coefficients = n_coefficients

    def _filter(
        self, input_array: np.ndarray, representations_to_filter_indices: np.ndarray
    ) -> np.ndarray:
        output_array = []
        for i in range(len(representations_to_filter_indices)):
            for j in range(input_array.shape[1]):
                for k in range(input_array.shape[2]):
                    output_array.append(
                        AutoReg(
                            input_array[representations_to_filter_indices[i], j, k],
                            lags=self.n_coefficients - 1,
                        )
                        .fit()
                        .params
                    )

        return output_array


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
        windowed_array = _get_windows_with_shift(input_array, self.window_size, self.shift)

        # Calculate RMS for each window
        return np.sqrt(np.mean(np.square(windowed_array), axis=-1))


class VARFilter(FilterBaseClass):
    """Computes the Variance with given window length and window shift over the input signal."""

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
        windowed_array = _get_windows_with_shift(input_array, self.window_size, self.shift)
        
        # Calculate variance for each window
        return np.var(windowed_array, axis=-1)


# TODO: Check if this is correct
class HISTFilter(FilterBaseClass):
    """Computes the Histogram with given window length and window shift over the input signal."""

    def __init__(
        self,
        window_size: int,
        shift: int = 1,
        bins: int = 10,
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
        self.bins = bins

    def _filter(self, input_array: np.ndarray) -> np.ndarray:
        output_array = []

        for i in range(0, input_array.shape[-1] - self.window_size + 1, self.shift):
            input_segment = input_array[..., i : i + self.window_size]
            histograms = np.zeros(
                (input_segment.shape[0], input_segment.shape[1], self.bins)
            )

            for j in range(histograms.shape[0]):
                for k in range(histograms.shape[1]):
                    histograms[j, k] = np.histogram(
                        input_segment[j, k], bins=self.bins
                    )[0]

            output_array.append(
                histograms.reshape((histograms.shape[0], -1, 1), order="F")
            )

        return np.concatenate(output_array, axis=-1)


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
        windowed_array = _get_windows_with_shift(input_array, self.window_size, self.shift)
        
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
        windowed_array = _get_windows_with_shift(input_array, self.window_size, self.shift)
        
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
        windowed_array = _get_windows_with_shift(input_array, self.window_size, self.shift)
        
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
        windowed_array = _get_windows_with_shift(input_array, self.window_size, self.shift)
        
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
        windowed_array = _get_windows_with_shift(input_array, self.window_size, self.shift)
        
        # Calculate slope sign changes for each window
        # 1. Calculate differences (first derivative)
        diffs = np.diff(windowed_array, axis=-1)
        
        # 2. Calculate sign of differences
        sign_diffs = np.sign(diffs)
        
        # 3. Calculate sign changes of the sign differences (second derivative sign changes)
        sign_changes = np.diff(sign_diffs, axis=-1)
        
        # 4. Count number of sign changes in each window
        return np.sum(np.abs(sign_changes) // 2, axis=-1)

# TODO
class SpectralInterpolationFilter(FilterBaseClass):
    def __init__(
        self,
        bandwidth: Tuple[float, float] = (47.5, 50.75),
        number_of_harmonics: int = 5,
        emg_frequency: float = 2044,
        input_is_chunked: bool = True,
        representations_to_filter: Union[Literal["all"], Sequence[int]] = "all",
    ):
        super().__init__(
            input_is_chunked=input_is_chunked,
            representations_to_filter=representations_to_filter,
        )
        self.bandwidth = bandwidth
        self.number_of_harmonics = number_of_harmonics
        self.emg_frequency = emg_frequency

        self._indices_to_interpolate = (
            np.repeat(np.array([bandwidth]), self.number_of_harmonics, axis=0)
            * np.arange(1, self.number_of_harmonics + 1)[..., None]
        )

    def _get_indices_to_interpolate(self, rfft_freqs: np.ndarray) -> List[np.ndarray]:
        mean_diff = np.mean(np.diff(rfft_freqs)) / 2

        return [
            np.argwhere(
                np.logical_and(
                    frequency_to_interpolate[0] - mean_diff <= rfft_freqs,
                    rfft_freqs <= frequency_to_interpolate[1] + mean_diff,
                )
            ).flatten()
            for frequency_to_interpolate in self._indices_to_interpolate
        ]

    def _filter(
        self, input_array: np.ndarray, representations_to_filter_indices: np.ndarray
    ) -> np.ndarray:
        output_array = np.copy(input_array)

        fourier = rfft(input_array[representations_to_filter_indices], axis=-1)
        fourier[..., 0] = 0

        smooth_fourier = savgol_filter(np.abs(fourier), 15, 3, axis=-1)

        for i, indices in enumerate(
            self._get_indices_to_interpolate(
                rfftfreq(input_array.shape[-1], d=1 / self.emg_frequency)
            )
        ):
            fourier[..., indices] = smooth_fourier[..., indices]

        output_array[representations_to_filter_indices] = irfft(fourier, axis=-1)

        return output_array
