"""Dataset utilities for MyoVerse.

Direct tensor loading from zarr with GPU support.

Classes
-------
DatasetCreator : Create datasets with any combination of modalities
DataModule : Lightning DataModule with input/target selection at training time
ContinuousDataset : On-the-fly windowing from continuous data
Modality : Configuration for a data modality

Features
--------
- Any number of modalities (EMG, EEG, kinematics, forces, etc.)
- Named tensor dimensions per modality
- Direct loading to GPU via device parameter
- Input/target selection at training time (not storage time)
- Chunked zarr storage for parallel window loading

Example
-------
>>> from myoverse.datasets import DatasetCreator, DataModule, Modality
>>> from myoverse.transforms import Compose, ZScore
>>>
>>> # Create dataset with multiple modalities
>>> creator = DatasetCreator(
...     modalities={
...         "emg": Modality(path="emg.pkl", dims=("channel", "time")),
...         "kinematics": Modality(path="kin.pkl", dims=("joint", "time")),
...     },
...     sampling_frequency=2048.0,
...     save_path="data.zarr",
... )
>>> creator.create()
>>>
>>> # Load directly to GPU with named tensors
>>> dm = DataModule(
...     "data.zarr",
...     inputs=["emg"],
...     targets=["kinematics"],
...     window_size=200,
...     n_windows_per_epoch=10000,
...     device="cuda",  # Direct GPU loading
... )
>>>
>>> inputs, targets = next(iter(dm.train_dataloader()))
>>> inputs.names   # ('batch', 'channel', 'time')
>>> inputs.device  # cuda:0
"""

from myoverse.datasets.loader_v2 import (
    DataModule,
    ContinuousDataset,
    # Backwards compat
    EMGDataModule,
    EMGContinuousDataset,
)
from myoverse.datasets.supervised_v2 import (
    DatasetCreator,
    Modality,
    # Backwards compat
    EMGDatasetCreator,
)
from myoverse.datasets.utils import (
    DataSplitter,
    DatasetFormatter,
)
from myoverse.datasets.defaults import (
    EMBCConfig,
    embc_kinematics_transform,
    embc_train_transform,
    embc_eval_transform,
    embc_target_transform,
)

__all__ = [
    # New API
    "DatasetCreator",
    "DataModule",
    "ContinuousDataset",
    "Modality",
    # Defaults / presets (transforms, not classes)
    "EMBCConfig",
    "embc_kinematics_transform",
    "embc_train_transform",
    "embc_eval_transform",
    "embc_target_transform",
    # Backwards compat
    "EMGDatasetCreator",
    "EMGDataModule",
    "EMGContinuousDataset",
    # Utilities
    "DataSplitter",
    "DatasetFormatter",
]
