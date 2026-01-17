"""Data types for MyoVerse.

This module provides data containers for various biosignal types used in
myocontrol research, including EMG, kinematics, and virtual hand data.

Example:
-------
>>> import numpy as np
>>> from myoverse.datatypes import EMGData, create_grid_layout
>>>
>>> # Create sample EMG data (16 channels, 1000 samples)
>>> emg_data = np.random.randn(16, 1000)
>>> sampling_freq = 2000  # 2000 Hz
>>>
>>> # Create a basic EMGData object
>>> emg = EMGData(emg_data, sampling_freq)
>>>
>>> # Create with grid layout
>>> grid = create_grid_layout(4, 4, fill_pattern='row')
>>> emg_with_grid = EMGData(emg_data, sampling_freq, grid_layouts=[grid])

"""

# Types and constants
# Base class
from myoverse.datatypes.base import _Data

# Data types
from myoverse.datatypes.emg import EMGData, create_grid_layout, emg_xarray
from myoverse.datatypes.kinematics import KinematicsData
from myoverse.datatypes.types import (
    DeletedRepresentation,
    InputRepresentationName,
    LastRepresentationName,
    Representation,
)
from myoverse.datatypes.virtual_hand import VirtualHandKinematics

# Re-export emg_tensor from transforms for convenient access
from myoverse.transforms.base import emg_tensor

# Data types map for dynamic lookup
DATA_TYPES_MAP = {
    "emg": EMGData,
    "kinematics": KinematicsData,
    "virtual_hand": VirtualHandKinematics,
}

__all__ = [
    # Types and constants
    "DeletedRepresentation",
    "Representation",
    "InputRepresentationName",
    "LastRepresentationName",
    # Base class
    "_Data",
    # Data types
    "EMGData",
    "KinematicsData",
    "VirtualHandKinematics",
    # Utilities
    "create_grid_layout",
    "DATA_TYPES_MAP",
    # Array/tensor creation
    "emg_xarray",
    "emg_tensor",
]
