"""
Creating a dataset
===========================

This example shows how to create a dataset for training a deep learning model.

# sphinx_gallery_defer_exec
"""

# %%
# In this example we will create a dataset that was used in our real-time paper [1]_.
#
# .. [1] Sîmpetru, R.C., März, M., Del Vecchio, A., 2023. Proportional and Simultaneous Real-Time Control of the Full Human Hand From High-Density Electromyography. IEEE TNSRE 31, 3118–3131. https://doi.org/10/gsgk4s
from functools import partial
from pathlib import Path

import numpy as np
from scipy.signal import butter

from myoverse.datasets.filters.emg_augmentations import WaveletDecomposition
from myoverse.datasets.filters.generic import (
    ApplyFunctionFilter,
    IndexDataFilter,
    IdentityFilter,
)
from myoverse.datasets.filters.temporal import SOSFrequencyFilter, RMSFilter
from myoverse.datasets.supervised import EMGDataset

# Example 1: Creating a dataset with specific filter pipelines using Zarr 3
dataset = EMGDataset(
    emg_data_path=Path(r"../data/emg.pkl").resolve(),
    ground_truth_data_path=Path(r"../data/kinematics.pkl").resolve(),
    ground_truth_data_type="kinematics",
    sampling_frequency=2044.0,
    tasks_to_use=["1", "2"],
    save_path=Path(r"../data/dataset.zarr").resolve(),
    emg_filter_pipeline_after_chunking=[
        [
            SOSFrequencyFilter(
                sos_filter_coefficients=butter(
                    4, [47.5, 52.5], "bandstop", output="sos", fs=2044
                ),
                is_output=True,
                name="Raw No Powerline (Bandstop 50 Hz)",
                input_is_chunked=True,
            ),
            SOSFrequencyFilter(
                sos_filter_coefficients=butter(4, 20, "lowpass", output="sos", fs=2044),
                is_output=True,
                name="Raw No High Freq (Lowpass 20 Hz)",
                input_is_chunked=True,
            ),
        ]
    ],
    emg_representations_to_filter_after_chunking=[["Last"]],
    ground_truth_filter_pipeline_before_chunking=[
        [
            ApplyFunctionFilter(
                function=np.reshape, newshape=(63, -1), input_is_chunked=False
            ),
            IndexDataFilter(indices=(slice(3, 63),), input_is_chunked=False),
        ]
    ],
    ground_truth_representations_to_filter_before_chunking=[["Input"]],
    ground_truth_filter_pipeline_after_chunking=[
        [
            ApplyFunctionFilter(
                function=partial(np.mean, axis=-1),
                is_output=True,
                name="Mean Kinematics per EMG Chunk",
                input_is_chunked=True,
            ),
        ]
    ],
    ground_truth_representations_to_filter_after_chunking=[["Last"]],
    chunk_size=192,
    chunk_shift=64,
    testing_split_ratio=0.3,
    validation_split_ratio=0.1,
    augmentation_pipelines=[
        [
            WaveletDecomposition(
                nr_of_grids=5, is_output=True, level=2, input_is_chunked=False
            )
        ]
    ],
    debug_level=1,  # Disable debug output
    silence_zarr_warnings=True,  # Silence zarr codec warnings
)

# Create the dataset
dataset.create_dataset()

# %%
# Default dataset are also available. Here is an example of how to use the EMBCDataset used in [2]_.
#
# .. [2] Sîmpetru, R.C., Osswald, M., Braun, D.I., Souza de Oliveira, D., Cakici, A.L., Del Vecchio, A., 2022. Accurate continuous prediction of 14 degrees of freedom of the hand from myoelectrical signals through convolutive deep learning, in: Proceedings of the 2022 44th Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC). Presented at the 2022 44th Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC), pp. 702–706. https://doi.org/10/gq2f47
from myoverse.datasets.defaults import EMBCDataset

# Using the default EMBCDataset with Zarr 3
dataset = EMBCDataset(
    emg_data_path=Path(r"../data/emg.pkl").resolve(),
    ground_truth_data_path=Path(r"../data/kinematics.pkl").resolve(),
    save_path=Path(r"../data/dataset_embc.zarr").resolve(),
    tasks_to_use=["1", "2"],
    debug_level=1,
    silence_zarr_warnings=True,  # Silence zarr codec warnings
)

# Create the EMBC dataset
dataset.create_dataset()
