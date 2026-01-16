"""Dataset utilities for MyoVerse.

Architecture
------------
This module uses a layered architecture for flexibility:

**Base Layer (Infrastructure)**
    WindowedDataset : Handles zarr I/O, windowing, caching, device management.
    Returns all modalities as a dict.

**Paradigm Layer (Learning Paradigms)**
    SupervisedDataset : Supervised learning (inputs → targets).
    Returns (inputs_dict, targets_dict) tuple.

    Future: ContrastiveDataset, MaskedDataset, etc.

**Integration Layer**
    DataModule : Lightning DataModule for training integration.

**Storage Layer**
    DatasetCreator : Creates zarr datasets from multi-modal data.
    Modality : Configuration for a data modality.

**Presets**
    Pre-configured transforms for published papers (EMBC 2022, etc.).

Example
-------
>>> from myoverse.datasets import DatasetCreator, DataModule, Modality
>>> from myoverse.datasets.presets import embc_train_transform
>>>
>>> # Create dataset
>>> creator = DatasetCreator(
...     modalities={
...         "emg": Modality(path="emg.pkl", dims=("channel", "time")),
...         "kinematics": Modality(path="kin.pkl", dims=("joint", "time")),
...     },
...     sampling_frequency=2048.0,
...     save_path="data.zip",
... )
>>> creator.create()
>>>
>>> # Load for training
>>> dm = DataModule(
...     "data.zip",
...     inputs=["emg"],
...     targets=["kinematics"],
...     window_size=200,
...     n_windows_per_epoch=10000,
...     train_transform=embc_train_transform(),
...     device="cuda",
... )
>>> dm.setup("fit")
>>> inputs, targets = next(iter(dm.train_dataloader()))
"""

# Base infrastructure
from myoverse.datasets.base import WindowedDataset

# Paradigms
from myoverse.datasets.paradigms import SupervisedDataset

# Integration
from myoverse.datasets.datamodule import DataModule, collate_supervised

# Storage
from myoverse.datasets.creator import DatasetCreator
from myoverse.datasets.modality import Modality

# Utilities
from myoverse.datasets.utils import DataSplitter, DatasetFormatter

# Presets (convenience re-exports)
from myoverse.datasets.presets import (
    EMBCConfig,
    embc_eval_transform,
    embc_kinematics_transform,
    embc_target_transform,
    embc_train_transform,
)

__all__ = [
    # Base
    "WindowedDataset",
    # Paradigms
    "SupervisedDataset",
    # Integration
    "DataModule",
    "collate_supervised",
    # Storage
    "DatasetCreator",
    "Modality",
    # Utilities
    "DataSplitter",
    "DatasetFormatter",
    # Presets
    "EMBCConfig",
    "embc_kinematics_transform",
    "embc_train_transform",
    "embc_eval_transform",
    "embc_target_transform",
]
