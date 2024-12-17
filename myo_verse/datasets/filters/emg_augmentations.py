import numba
import numpy as np
import pywt
from scipy import interpolate

from myo_verse.datasets.filters._template import EMGAugmentation


@numba.njit(nogil=True, fastmath=True)
def _gaussian_noise(input_array: np.ndarray, target_snr__db: float) -> np.ndarray:
    squared_chunk = np.square(input_array.astype(np.float64))
    mean_squared_chunk = np.zeros((input_array.shape[0]))
    for i in range(input_array.shape[1]):
        mean_squared_chunk += squared_chunk[:, i]
    mean_squared_chunk /= input_array.shape[1]

    noise_avg_per_channel_and_electrode__watts = np.sqrt(
        np.power(
            10, ((10 * np.log10(mean_squared_chunk + 1e-15)) - target_snr__db) / 10
        )
    )

    for electrode_index in range(input_array.shape[0]):
        input_array[electrode_index] += np.random.normal(
            loc=0.0,
            scale=noise_avg_per_channel_and_electrode__watts[electrode_index],
            size=input_array.shape[1],
        ).astype(np.int16)

    return input_array


class GaussianNoise(EMGAugmentation):
    """Adds Gaussian noise to the input EMG data. This augmentation is based on the paper [1]_

    Parameters
    ----------
    target_snr__db : float, optional
        The target signal-to-noise ratio in decibels, by default 5.0.
    input_is_chunked : bool
        Whether the input is chunked or not.
    is_output : bool
        Whether the filter is an output filter. If True, the resulting signal will be outputted by and dataset pipeline.

    Methods
    -------
    __call__(chunk: numpy.ndarray) -> numpy.ndarray
        Augments the chunk.

    Notes
    -----
    .. [1] Tsinganos, P., Cornelis, B., Cornelis, J., Jansen, B., Skodras, A., 2020. Data Augmentation of Surface Electromyography for Hand Gesture Recognition. Sensors 20, 4892. https://doi.org/10/grc7ph
    """

    def __init__(
        self,
        target_snr__db: float = 5.0,
        input_is_chunked: bool = None,
        is_output: bool = False,
    ):
        super().__init__(input_is_chunked=input_is_chunked, is_output=is_output)
        self.target_snr__db = target_snr__db

    def _filter(self, input_array: np.ndarray) -> np.ndarray:
        return _gaussian_noise(
            input_array=input_array, target_snr__db=self.target_snr__db
        )


class MagnitudeWarping(EMGAugmentation):
    """Magnitude warping augmentation. This augmentation is based on the paper [2]_

    Parameters
    ----------
    nr_of_point_for_spline : int, optional
        The number of points to use for the spline, by default 6.
    gaussian_mean : float, optional
        The mean of the Gaussian distribution, by default 1.0.
    gaussian_std : float, optional
        The standard deviation of the Gaussian distribution, by default 0.35.
    nr_of_grids : int, optional
        The number of grids to use, by default 5.
    input_is_chunked : bool
        Whether the input is chunked or not.
    is_output : bool
        Whether the filter is an output filter. If True, the resulting signal will be outputted by and dataset pipeline.

    Methods
    -------
    __call__(chunk: numpy.ndarray) -> numpy.ndarray
        Augments the chunk.

    Notes
    -----
    .. [2] Tsinganos, P., Cornelis, B., Cornelis, J., Jansen, B., Skodras, A., 2020. Data Augmentation of Surface Electromyography for Hand Gesture Recognition. Sensors 20, 4892. https://doi.org/10/grc7ph
    """

    def __init__(
        self,
        nr_of_point_for_spline: int = 6,
        gaussian_mean: float = 1.0,
        gaussian_std: float = 0.35,
        nr_of_grids: int = None,
        input_is_chunked: bool = None,
        is_output: bool = False,
    ):
        super().__init__(input_is_chunked=input_is_chunked, is_output=is_output)
        self.nr_of_point_for_spline = nr_of_point_for_spline
        self.gaussian_mean = gaussian_mean
        self.gaussian_std = gaussian_std
        self.nr_of_grids = nr_of_grids

        if self.nr_of_grids is None:
            raise ValueError("nr_of_grids must be specified.")

    def _filter(self, input_array: np.ndarray) -> np.ndarray:
        random_gens = [np.random.default_rng() for _ in range(self.nr_of_grids)]

        return np.multiply(
            input_array.astype(np.float64),
            np.repeat(
                np.array(
                    [
                        interpolate.interp1d(
                            np.linspace(
                                start=0,
                                stop=input_array.shape[-1],
                                num=self.nr_of_point_for_spline,
                            ),
                            random_gens[i].normal(
                                loc=self.gaussian_mean,
                                scale=self.gaussian_std,
                                size=self.nr_of_point_for_spline,
                            ),
                            kind="cubic",
                        )(np.arange(input_array.shape[-1]))
                        for i in range(self.nr_of_grids)
                    ]
                ),
                repeats=input_array.shape[0] // self.nr_of_grids,
                axis=0,
            ),
        )


class WaveletDecomposition(EMGAugmentation):
    """Wavelet decomposition augmentation. This augmentation is based on the paper [3]_

    Parameters
    ----------
    b : float, optional
        The scaling factor, by default 0.25.
    wavelet : str, optional
        The wavelet to use, by default "db7".
    level : int, optional
        The level of decomposition, by default 5.
    nr_of_grids : int, optional
        The number of grids to use, by default 5.
    input_is_chunked : bool
        Whether the input is chunked or not.
    is_output : bool
        Whether the filter is an output filter. If True, the resulting signal will be outputted by and dataset pipeline.

    Methods
    -------
    __call__(chunk: numpy.ndarray) -> numpy.ndarray
        Augments the chunk.

    Notes
    -----
    .. [3] Tsinganos, P., Cornelis, B., Cornelis, J., Jansen, B., Skodras, A., 2020. Data Augmentation of Surface Electromyography for Hand Gesture Recognition. Sensors 20, 4892. https://doi.org/10/grc7ph
    """

    def __init__(
        self,
        b: float = 0.25,
        wavelet: str = "db7",
        level: int = 5,
        nr_of_grids: int = None,
        input_is_chunked: bool = None,
        is_output: bool = False,
    ):
        super().__init__(input_is_chunked=input_is_chunked, is_output=is_output)
        self.b = b
        self.wavelet = wavelet
        self.level = level
        self.nr_of_grids = nr_of_grids

        if self.nr_of_grids is None:
            raise ValueError("nr_of_grids must be specified.")

    def _filter(self, input_array: np.ndarray) -> np.ndarray:
        coefficients_per_grid = [
            pywt.wavedec(
                grid.astype(np.float64),
                wavelet=self.wavelet,
                level=self.level,
                mode="reflect",
                axis=1,
            )
            for grid in np.array_split(input_array, self.nr_of_grids, axis=0)
        ]
        return np.concatenate(
            [
                pywt.waverec(
                    [coefficients[0]]
                    + [coefficients[i] * self.b for i in range(1, self.level + 1)],
                    wavelet=self.wavelet,
                    axis=1,
                )
                for coefficients in coefficients_per_grid
            ],
            axis=0,
        )
