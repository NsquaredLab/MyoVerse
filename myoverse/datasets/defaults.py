from pathlib import Path
from typing import Sequence, Optional

import numpy as np
from scipy.signal import butter

from myoverse.datasets.filters.emg_augmentations import (
    GaussianNoise,
    MagnitudeWarping,
    WaveletDecomposition,
)
from myoverse.datasets.filters.generic import (
    ApplyFunctionFilter,
    IndexDataFilter,
    IdentityFilter,
)
from myoverse.datasets.filters.temporal import RMSFilter, SOSFrequencyFilter
from myoverse.datasets.supervised import EMGDataset


class EMBCDataset:
    """Official dataset maker for the EMBC paper [1].

    Parameters
    ----------
    emg_data_path : Path
        The path to the pickle file containing the EMG data.
        This should be a dictionary with the keys as the tasks in tasks_to_use and the values as the EMG data.
        The EMG data should be of shape (320, samples).
    ground_truth_data_path : Path
        The path to the pickle file containing the ground truth data.
        This should be a dictionary with the keys as the tasks in tasks_to_use and the values as the ground truth data.
        The ground truth data should be of shape (21, 3, samples).
    save_path : Path
        The path to save the dataset to. This should be a zarr file.
    emg_data : dict[str, np.ndarray], optional
        Optional dictionary containing EMG data if not loading from a file.
    ground_truth_data : dict[str, np.ndarray], optional
        Optional dictionary containing ground truth data if not loading from a file.
    tasks_to_use : Sequence[str], optional
        The tasks to use.
    debug_level : int, optional
        Debug level (0-2). Default is 0 (no debugging).
    silence_zarr_warnings : bool, optional
        Whether to silence all Zarr-related warnings. Default is False.

    Methods
    -------
    create_dataset()
        Creates the dataset.

    References
    ----------
    [1] Sîmpetru, R.C., Osswald, M., Braun, D.I., Souza de Oliveira, D., Cakici, A.L., Del Vecchio, A., 2022.
    Accurate Continuous Prediction of 14 Degrees of Freedom of the Hand from Myoelectrical Signals through
    Convolutive Deep Learning, in: Proceedings of the 2022 44th Annual International Conference of
    the IEEE Engineering in Medicine & Biology Society (EMBC) pp. 702–706. https://doi.org/10/gq2f47
    """

    def __init__(
        self,
        emg_data_path: Path,
        ground_truth_data_path: Path,
        save_path: Path,
        emg_data: dict[str, np.ndarray] = {},
        ground_truth_data: dict[str, np.ndarray] = {},
        tasks_to_use: Sequence[str] = ("Change Me",),
        debug_level: int = 0,
        silence_zarr_warnings: bool = False,
    ):
        self.emg_data_path = emg_data_path
        self.emg_data = emg_data
        self.ground_truth_data_path = ground_truth_data_path
        self.ground_truth_data = ground_truth_data
        self.tasks_to_use = tasks_to_use
        self.save_path = save_path
        self.debug_level = debug_level
        self.silence_zarr_warnings = silence_zarr_warnings

    def create_dataset(self):
        # EMBC default settings
        dataset = EMGDataset(
            emg_data_path=self.emg_data_path,
            emg_data=self.emg_data,
            ground_truth_data_path=self.ground_truth_data_path,
            ground_truth_data=self.ground_truth_data,
            ground_truth_data_type="kinematics",
            sampling_frequency=2048.0,
            tasks_to_use=self.tasks_to_use,
            save_path=self.save_path,
            chunk_size=192,
            chunk_shift=64,
            testing_split_ratio=0.2,
            validation_split_ratio=0.2,
            debug_level=self.debug_level,
            silence_zarr_warnings=self.silence_zarr_warnings,
            # EMBC-specific filter pipelines and augmentations
            emg_filter_pipeline_after_chunking=[
                [
                    IdentityFilter(is_output=True, name="raw", input_is_chunked=True),
                    SOSFrequencyFilter(
                        sos_filter_coefficients=butter(
                            4, 20, "lowpass", output="sos", fs=2048
                        ),
                        is_output=True,
                        input_is_chunked=True,
                    ),
                ]
            ],
            emg_representations_to_filter_after_chunking=[["Last"]],
            ground_truth_filter_pipeline_before_chunking=[
                [
                    ApplyFunctionFilter(
                        function=np.reshape,
                        name="Reshape",
                        newshape=(63, -1),
                        input_is_chunked=False,
                    ),
                    IndexDataFilter(indices=(slice(3, 63),), input_is_chunked=False),
                ]
            ],
            ground_truth_representations_to_filter_before_chunking=[["Input"]],
            ground_truth_filter_pipeline_after_chunking=[
                [
                    ApplyFunctionFilter(
                        function=np.mean,
                        name="Mean",
                        axis=-1,
                        is_output=True,
                        input_is_chunked=True,
                    )
                ]
            ],
            ground_truth_representations_to_filter_after_chunking=[["Last"]],
            augmentation_pipelines=[
                [GaussianNoise(is_output=True, input_is_chunked=False)],
                [
                    MagnitudeWarping(
                        is_output=True, nr_of_grids=5, input_is_chunked=False
                    )
                ],
                [
                    WaveletDecomposition(
                        level=3, is_output=True, nr_of_grids=5, input_is_chunked=False
                    )
                ],
            ],
            amount_of_chunks_to_augment_at_once=500,
        ).create_dataset()


class CastelliniDataset:
    """Dataset maker made after the Castellini paper [1].
    This is not the official dataset maker used but our own version made after the paper.

    Parameters
    ----------
    emg_data_path : Path
        The path to the pickle file containing the EMG data.
        This should be a dictionary with the keys as the tasks in tasks_to_use and the values as the EMG data.
        The EMG data should be of shape (320, samples).
    ground_truth_data_path : Path
        The path to the pickle file containing the ground truth data.
        This should be a dictionary with the keys as the tasks in tasks_to_use and the values as the ground truth data.
        The ground truth data should be of shape (21, 3, samples).
    save_path : Path
        The path to save the dataset to. This should be a zarr file.
    emg_data : dict[str, np.ndarray], optional
        Optional dictionary containing EMG data if not loading from a file.
    ground_truth_data : dict[str, np.ndarray], optional
        Optional dictionary containing ground truth data if not loading from a file.
    tasks_to_use : Sequence[str], optional
        The tasks to use.
    debug_level : int, optional
        Debug level (0-2). Default is 0 (no debugging).
    silence_zarr_warnings : bool, optional
        Whether to silence all Zarr-related warnings. Default is False.

    Methods
    -------
    create_dataset()
        Creates the dataset.

    References
    ----------
    [1] Nowak, M., Vujaklija, I., Sturma, A., Castellini, C., Farina, D., 2023.
    Simultaneous and Proportional Real-Time Myocontrol of Up to Three Degrees of Freedom of the Wrist and Hand.
    IEEE Transactions on Biomedical Engineering 70, 459–469. https://doi.org/10/grc7qf
    """

    def __init__(
        self,
        emg_data_path: Path,
        ground_truth_data_path: Path,
        save_path: Path,
        emg_data: dict[str, np.ndarray] = {},
        ground_truth_data: dict[str, np.ndarray] = {},
        tasks_to_use: Sequence[str] = ("Change Me",),
        debug_level: int = 0,
        silence_zarr_warnings: bool = False,
    ):
        self.emg_data_path = emg_data_path
        self.emg_data = emg_data
        self.ground_truth_data_path = ground_truth_data_path
        self.ground_truth_data = ground_truth_data
        self.save_path = save_path
        self.tasks_to_use = tasks_to_use
        self.debug_level = debug_level
        self.silence_zarr_warnings = silence_zarr_warnings

    def create_dataset(self):
        dataset = EMGDataset(
            emg_data_path=self.emg_data_path,
            emg_data=self.emg_data,
            ground_truth_data_path=self.ground_truth_data_path,
            ground_truth_data=self.ground_truth_data,
            ground_truth_data_type="kinematics",
            sampling_frequency=2048,
            tasks_to_use=self.tasks_to_use,
            save_path=self.save_path,
            debug_level=self.debug_level,
            silence_zarr_warnings=self.silence_zarr_warnings,
            # Castellini-specific filter pipelines
            emg_filter_pipeline_before_chunking=[
                [
                    SOSFrequencyFilter(
                        sos_filter_coefficients=butter(
                            5,
                            (20, 500),
                            "bandpass",
                            output="sos",
                            fs=2048,
                        ),
                        name="Bandpass 20-500 Hz",
                        input_is_chunked=False,
                    ),
                    SOSFrequencyFilter(
                        sos_filter_coefficients=butter(
                            5, (45, 55), "bandstop", output="sos", fs=2048
                        ),
                        name="Bandstop 45-55 Hz",
                        input_is_chunked=False,
                    ),
                    RMSFilter(
                        window_size=204,
                        shift=20,
                        name=f"RMS {204 / 2048 * 1000} ms",
                        input_is_chunked=False,
                    ),
                ]
            ],
            emg_representations_to_filter_before_chunking=[["Input"]],
            ground_truth_filter_pipeline_before_chunking=[
                [
                    ApplyFunctionFilter(
                        function=np.reshape,
                        newshape=(63, -1),
                        name="Reshape",
                        input_is_chunked=False,
                    ),
                    IndexDataFilter(
                        indices=(slice(3, 63),),
                        name="Indexing (Remove Wrist)",
                        input_is_chunked=False,
                    ),
                ]
            ],
            ground_truth_representations_to_filter_before_chunking=[["Input"]],
            ground_truth_filter_pipeline_after_chunking=[
                [
                    ApplyFunctionFilter(
                        function=np.mean,
                        axis=-1,
                        is_output=True,
                        name="Mean",
                        input_is_chunked=True,
                    )
                ]
            ],
            ground_truth_representations_to_filter_after_chunking=[["Last"]],
            augmentation_pipelines=[
                [GaussianNoise(is_output=True, input_is_chunked=False)],
                [
                    MagnitudeWarping(
                        is_output=True, input_is_chunked=False, nr_of_grids=5
                    )
                ],
                [
                    WaveletDecomposition(
                        level=3, is_output=True, input_is_chunked=False, nr_of_grids=5
                    )
                ],
            ],
            amount_of_chunks_to_augment_at_once=500,
        )
        dataset.create_dataset()
