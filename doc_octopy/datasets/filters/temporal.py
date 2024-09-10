from __future__ import annotations

from typing import List, Literal, Sequence, Tuple, Union

import numpy as np
from scipy.fft import irfft, rfft, rfftfreq
from scipy.signal import savgol_filter, sosfilt, sosfiltfilt
from statsmodels.tsa.ar_model import AutoReg

from doc_octopy.datasets.filters._template import FilterBaseClass
from doc_octopy.datasets.filters.generic import ApplyFunctionFilter


class SOSFrequencyFilter(FilterBaseClass):
    """Filter that applies a second-order-section filter to the input array.

    Parameters
    ----------
    sos_filter_coefficients : tuple[np.ndarray, np.ndarray | float, np.ndarray]
        The second-order-section filter coefficients. This is a tuple of the form (sos, gain, delay).
    forwards_and_backwards : bool
        Whether to apply the filter forwards and backwards or only forwards.
    input_is_chunked : bool
        Whether the input is chunked or not.
    is_output : bool
        Whether the filter is an output filter. If True, the resulting signal will be outputted by and dataset pipeline.

    Methods
    -------
    __call__(input_array: np.ndarray) -> np.ndarray
        Filters the input array. Input shape is determined by whether the allowed_input_type
        is "both", "chunked" or "not chunked".
    """

    def __init__(
        self,
        sos_filter_coefficients: tuple[
            np.ndarray, Union[np.ndarray, float], np.ndarray
        ],
        forwards_and_backwards: bool = True,
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

        self.sos_filter_coefficients = sos_filter_coefficients
        self.forwards_and_backwards = forwards_and_backwards

        self._filtering_method = sosfiltfilt if self.forwards_and_backwards else sosfilt

    def _filter(self, input_array: np.ndarray) -> np.ndarray:
        return self._filtering_method(self.sos_filter_coefficients, input_array)


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

    def __init__(self, window_size: int, shift: int = 1, input_is_chunked: bool = None):
        super().__init__(
            input_is_chunked=input_is_chunked,
        )

        self.window_size = window_size
        self.shift = shift

        if self.window_size < 1:
            raise ValueError("window_size must be greater than 0.")
        if self.shift < 1:
            raise ValueError("shift must be greater than 0.")

    def _filter(self, input_array: np.ndarray) -> np.ndarray:
        if self.input_is_chunked:
            output_array = []

            for i in range(0, input_array.shape[-1] - self.window_size + 1, self.shift):
                output_array.append(
                    np.sqrt(
                        np.mean(
                            input_array[..., i : i + self.window_size] ** 2, axis=-1
                        )
                    )
                )

            return np.concatenate(output_array, axis=-1)

        return np.sqrt(np.mean(input_array**2, axis=-1))


# TODO
class VARFilter(FilterBaseClass):
    """Computes the Variance with given window length and window shift over the input signal."""

    def __init__(
        self,
        window_size: int,
        shift: int = 1,
        input_is_chunked: bool = True,
        representations_to_filter: Union[Literal["all"], Sequence[int]] = "all",
    ):
        super().__init__(
            input_is_chunked=input_is_chunked,
            representations_to_filter=representations_to_filter,
            changes_filtered_dimension=True,
        )
        self.window_size = window_size
        self.shift = shift

    def _filter(
        self, input_array: np.ndarray, representations_to_filter_indices: np.ndarray
    ) -> np.ndarray:
        output_array = []

        for i in range(0, input_array.shape[-1] - self.window_size + 1, self.shift):
            output_array.append(
                np.var(
                    (
                        input_array[
                            representations_to_filter_indices,
                            ...,
                            i : i + self.window_size,
                        ]
                    ),
                    axis=-1,
                    keepdims=True,
                )
            )

        return np.concatenate(output_array, axis=-1)


# TODO
class HISTFilter(FilterBaseClass):
    """Computes the Histogram with given window length and window shift over the input signal."""

    def __init__(
        self,
        window_size: int,
        shift: int = 1,
        bins: int = 10,
        input_is_chunked: bool = True,
        representations_to_filter: Union[Literal["all"], Sequence[int]] = "all",
    ):
        super().__init__(
            input_is_chunked=input_is_chunked,
            representations_to_filter=representations_to_filter,
            changes_filtered_dimension=True,
        )
        self.window_size = window_size
        self.shift = shift
        self.bins = bins

    def _filter(
        self, input_array: np.ndarray, representations_to_filter_indices: np.ndarray
    ) -> np.ndarray:
        output_array = []

        for i in range(0, input_array.shape[-1] - self.window_size + 1, self.shift):
            input_segment = input_array[
                representations_to_filter_indices, ..., i : i + self.window_size
            ]
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


# TODO
class MAVFilter(FilterBaseClass):
    """Computes the Mean Absolute Value with given window length and window shift over the input signal. See formula in
    the following paper: https://doi.org/10.1080/10255842.2023.2165068.
    """

    def __init__(
        self,
        window_size: int,
        shift: int = 1,
        input_is_chunked: bool = True,
        representations_to_filter: Union[Literal["all"], Sequence[int]] = "all",
    ):
        super().__init__(
            input_is_chunked=input_is_chunked,
            representations_to_filter=representations_to_filter,
            changes_filtered_dimension=True,
        )
        self.window_size = window_size
        self.shift = shift

    def _filter(
        self, input_array: np.ndarray, representations_to_filter_indices: np.ndarray
    ) -> np.ndarray:
        output_array = []

        for i in range(0, input_array.shape[-1] - self.window_size + 1, self.shift):
            output_array.append(
                np.mean(
                    np.abs(
                        input_array[
                            representations_to_filter_indices,
                            ...,
                            i : i + self.window_size,
                        ]
                    ),
                    axis=-1,
                    keepdims=True,
                )
            )

        return np.concatenate(output_array, axis=-1)


# TODO
class IAVFilter(FilterBaseClass):
    """Computes the Integrated Absolute Value with given window length and window shift over the input signal. See
    formula in the following paper: https://doi.org/10.1080/10255842.2023.2165068.
    """

    def __init__(
        self,
        window_size: int,
        shift: int = 1,
        input_is_chunked: bool = True,
        representations_to_filter: Union[Literal["all"], Sequence[int]] = "all",
    ):
        super().__init__(
            input_is_chunked=input_is_chunked,
            representations_to_filter=representations_to_filter,
            changes_filtered_dimension=True,
        )
        self.window_size = window_size
        self.shift = shift

    def _filter(
        self, input_array: np.ndarray, representations_to_filter_indices: np.ndarray
    ) -> np.ndarray:
        output_array = []

        for i in range(0, input_array.shape[-1] - self.window_size + 1, self.shift):
            output_array.append(
                np.sum(
                    np.abs(
                        input_array[
                            representations_to_filter_indices,
                            ...,
                            i : i + self.window_size,
                        ]
                    ),
                    axis=-1,
                    keepdims=True,
                )
            )

        return np.concatenate(output_array, axis=-1)


# TODO
class WFLFilter(FilterBaseClass):
    """Computes the Waveform Length with given window length and window shift over the input signal. See
    formula in the following paper: https://doi.org/10.1080/10255842.2023.2165068.
    """

    def __init__(
        self,
        window_size: int,
        shift: int = 1,
        input_is_chunked: bool = True,
        representations_to_filter: Union[Literal["all"], Sequence[int]] = "all",
    ):
        super().__init__(
            input_is_chunked=input_is_chunked,
            representations_to_filter=representations_to_filter,
            changes_filtered_dimension=True,
        )
        self.window_size = window_size
        self.shift = shift

    def _filter(
        self, input_array: np.ndarray, representations_to_filter_indices: np.ndarray
    ) -> np.ndarray:
        output_array = []

        for i in range(0, input_array.shape[-1] - self.window_size + 1, self.shift):
            output_array.append(
                np.sum(
                    np.abs(
                        np.diff(
                            input_array[
                                representations_to_filter_indices,
                                ...,
                                i : i + self.window_size,
                            ]
                        )
                    ),
                    axis=-1,
                    keepdims=True,
                )
            )

        return np.concatenate(output_array, axis=-1)


# TODO
class ZCFilter(FilterBaseClass):
    """Computes the Zero Crossings with given window length and window shift over the input signal. See formula in the
    following paper: https://doi.org/10.1080/10255842.2023.2165068.
    """

    def __init__(
        self,
        window_size: int,
        shift: int = 1,
        input_is_chunked: bool = True,
        representations_to_filter: Union[Literal["all"], Sequence[int]] = "all",
    ):
        super().__init__(
            input_is_chunked=input_is_chunked,
            representations_to_filter=representations_to_filter,
            changes_filtered_dimension=True,
        )
        self.window_size = window_size
        self.shift = shift

    def _filter(
        self, input_array: np.ndarray, representations_to_filter_indices: np.ndarray
    ) -> np.ndarray:
        output_array = []

        for i in range(0, input_array.shape[-1] - self.window_size + 1, self.shift):
            output_array.append(
                np.sum(
                    np.abs(
                        np.diff(
                            np.sign(
                                input_array[
                                    representations_to_filter_indices,
                                    ...,
                                    i : i + self.window_size,
                                ]
                            )
                        )
                    )
                    // 2,
                    axis=-1,
                    keepdims=True,
                )
            )

        return np.concatenate(output_array, axis=-1)


# TODO
class SSCFilter(FilterBaseClass):
    """Computes the Slope Sign Change with given window length and window shift over the input signal. See formula in
    the following paper: https://doi.org/10.1080/10255842.2023.2165068.
    """

    def __init__(
        self,
        window_size: int,
        shift: int = 1,
        input_is_chunked: bool = True,
        representations_to_filter: Union[Literal["all"], Sequence[int]] = "all",
    ):
        super().__init__(
            input_is_chunked=input_is_chunked,
            representations_to_filter=representations_to_filter,
            changes_filtered_dimension=True,
        )
        self.window_size = window_size
        self.shift = shift

    def _filter(
        self, input_array: np.ndarray, representations_to_filter_indices: np.ndarray
    ) -> np.ndarray:
        output_array = []

        for i in range(0, input_array.shape[-1] - self.window_size + 1, self.shift):
            output_array.append(
                np.sum(
                    np.abs(
                        np.diff(
                            np.sign(
                                np.diff(
                                    input_array[
                                        representations_to_filter_indices,
                                        ...,
                                        i : i + self.window_size,
                                    ]
                                )
                            )
                        )
                    )
                    // 2,
                    axis=-1,
                    keepdims=True,
                )
            )

        return np.concatenate(output_array, axis=-1)


# TODO
class GaileyFeature2(FilterBaseClass):
    """Computes the second EMG feature from the Gailey et al. paper  with given window length and window shift over the
    input signal. See formula in the following paper: https://doi.org/10.3389/fneur.2017.00007.
    """

    def __init__(
        self,
        window_size: int,
        shift: int = 1,
        input_is_chunked: bool = True,
        representations_to_filter: Union[Literal["all"], Sequence[int]] = "all",
    ):
        super().__init__(
            input_is_chunked=input_is_chunked,
            representations_to_filter=representations_to_filter,
            changes_filtered_dimension=True,
        )
        self.window_size = window_size
        self.shift = shift

    def _filter(
        self, input_array: np.ndarray, representations_to_filter_indices: np.ndarray
    ) -> np.ndarray:
        output_array = []

        for i in range(0, input_array.shape[-1] - self.window_size + 1, self.shift):
            output_array.append(
                np.log(
                    (
                        np.sum(
                            input_array[
                                representations_to_filter_indices,
                                ...,
                                i : i + self.window_size,
                            ]
                            ** 2,
                            axis=-1,
                            keepdims=True,
                        )
                        + np.finfo(float).eps
                    )
                    / self.window_size
                )
            )

        return np.concatenate(output_array, axis=-1)


# TODO
class GaileyFeature3(FilterBaseClass):
    """Computes the third EMG feature from the Gailey et al. paper  with given window length and window shift over the
    input signal. See formula in the following paper: https://doi.org/10.3389/fneur.2017.00007.
    """

    def __init__(
        self,
        window_size: int,
        shift: int = 1,
        input_is_chunked: bool = True,
        representations_to_filter: Union[Literal["all"], Sequence[int]] = "all",
    ):
        super().__init__(
            input_is_chunked=input_is_chunked,
            representations_to_filter=representations_to_filter,
            changes_filtered_dimension=True,
        )
        self.window_size = window_size
        self.shift = shift

    def _filter(
        self, input_array: np.ndarray, representations_to_filter_indices: np.ndarray
    ) -> np.ndarray:
        output_array = []

        for i in range(0, input_array.shape[-1] - self.window_size + 1, self.shift):
            segment = input_array[
                representations_to_filter_indices, ..., i : i + self.window_size
            ]
            m0 = np.sum(segment**2, axis=-1, keepdims=True)
            m2 = np.sum(
                np.diff(segment, axis=-1, prepend=segment[..., [0]]) ** 2,
                axis=-1,
                keepdims=True,
            )
            m4 = np.sum(
                np.diff(
                    np.diff(segment, axis=-1, prepend=segment[..., [0]]),
                    prepend=segment[..., [0]],
                )
                ** 2,
                axis=-1,
                keepdims=True,
            )

            IF = np.sqrt(
                m2**2 / (m0 + np.finfo(float).eps) / (m4 + np.finfo(float).eps)
            )
            WL = np.sum(
                np.abs(np.diff(segment, axis=-1, prepend=segment[..., [0]])),
                axis=-1,
                keepdims=True,
            )

            output_array.append(
                np.log((IF + np.finfo(float).eps) / (WL + np.finfo(float).eps))
            )

        return np.concatenate(output_array, axis=-1)


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
