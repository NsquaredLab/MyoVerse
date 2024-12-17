from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.signal import butter

from myoverse.datasets.filters.emg_augmentations import (
    GaussianNoise,
    MagnitudeWarping,
    WaveletDecomposition,
)
from myoverse.datasets.filters.generic import ApplyFunctionFilter, IndexDataFilter, IdentityFilter
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
    tasks_to_use : Sequence[str], optional
        The tasks to use. The default is EXPERIMENTS_TO_USE.

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
        tasks_to_use: Sequence[str] = ("Change Me",),
        debug: bool = False,
    ):
        self.emg_data_path = emg_data_path
        self.ground_truth_data_path = ground_truth_data_path
        self.save_path = save_path
        self.tasks_to_use = tasks_to_use
        self.debug = debug

    def create_dataset(self):
        EMGDataset(
            emg_data_path=self.emg_data_path,
            ground_truth_data_path=self.ground_truth_data_path,
            sampling_frequency=2048,
            tasks_to_use=self.tasks_to_use,
            save_path=self.save_path,
            emg_filter_pipeline_after_chunking=[
                [
                    IdentityFilter(is_output=True),
                    SOSFrequencyFilter(
                        sos_filter_coefficients=butter(
                            4, 20, "lowpass", output="sos", fs=2048
                        ),
                        is_output=True,
                    )
                ]
            ],
            emg_representations_to_filter_after_chunking=["Last"],
            ground_truth_filter_pipeline_before_chunking=[
                [
                    ApplyFunctionFilter(function=np.reshape, name="Reshape", newshape=(63, -1)),
                    IndexDataFilter(indices=(slice(3, 63),)),
                ]
            ],
            ground_truth_representations_to_filter_before_chunking=["Input"],
            ground_truth_filter_pipeline_after_chunking=[
                [ApplyFunctionFilter(function=np.mean, name="Mean", axis=-1, is_output=True)]
            ],
            ground_truth_representations_to_filter_after_chunking=["Last"],
            augmentation_pipelines=[
                [GaussianNoise(is_output=True)],
                [MagnitudeWarping(is_output=True, nr_of_grids=5)],
                [WaveletDecomposition(level=3, is_output=True, nr_of_grids=5)],
            ],
            amount_of_chunks_to_augment_at_once=500,
            debug=self.debug,
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
        tasks_to_use: Sequence[str] = ("Change Me",),
    ):
        self.emg_data_path = emg_data_path
        self.ground_truth_data_path = ground_truth_data_path
        self.save_path = save_path
        self.tasks_to_use = tasks_to_use

    def create_dataset(self):
        EMGDataset(
            emg_data_path=self.emg_data_path,
            ground_truth_data_path=self.ground_truth_data_path,
            tasks_to_use=self.tasks_to_use,
            save_path=self.save_path,
            emg_filter_pipeline_before_chunking=[
                [
                    SOSFrequencyFilter(
                        sos_filter_coefficients=butter(
                            5, (20, 500), "bandpass", output="sos", fs=2048
                        ),
                    ),
                    SOSFrequencyFilter(
                        sos_filter_coefficients=butter(
                            5, (45, 55), "bandpass", output="sos", fs=2048
                        ),
                    ),
                    RMSFilter(window_size=204, shift=20),
                ]
            ],
            ground_truth_filter_pipeline_before_chunking=[
                [
                    ApplyFunctionFilter(function=np.reshape, newshape=(63, -1)),
                    IndexDataFilter(indices=(slice(3, 63),)),
                ]
            ],
            ground_truth_filter_pipeline_after_chunking=[
                [ApplyFunctionFilter(function=np.mean, axis=-1, is_output=True)]
            ],
            augmentation_pipelines=[
                [GaussianNoise(is_output=True)],
                [MagnitudeWarping(is_output=True)],
                [WaveletDecomposition(level=3, is_output=True)],
            ],
            amount_of_chunks_to_augment_at_once=500,
        ).create_dataset()
