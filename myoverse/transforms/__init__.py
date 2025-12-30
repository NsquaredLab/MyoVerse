"""Transform system for MyoVerse.

GPU-accelerated transforms using PyTorch named tensors.
Works on both CPU and GPU - tensors provide dimension awareness everywhere.

Example
-------
>>> import torch
>>> from myoverse.transforms import Pipeline, ZScore, RMS, Bandpass
>>>
>>> # Create named tensor (works on CPU or GPU)
>>> x = torch.randn(64, 2048, names=('channel', 'time'))
>>>
>>> # Pipeline with dimension-aware transforms
>>> pipeline = Pipeline([
...     Bandpass(20, 450, fs=2048, dim='time'),
...     ZScore(dim='time'),
...     RMS(window_size=200, dim='time'),
... ])
>>> y = pipeline(x)
>>>
>>> # Or on GPU
>>> x_gpu = x.cuda()
>>> y_gpu = pipeline(x_gpu)
"""

# Re-export torchvision's Compose
from torchvision.transforms import Compose

# Base classes and utilities
from myoverse.transforms.base import (
    TensorTransform as Transform,
    TensorTransformError as TransformError,
    named_tensor,
    emg_tensor,
    get_dim_index,
    align_tensors,
)

# Temporal / signal processing
from myoverse.transforms.temporal import (
    RMS,
    MAV,
    VAR,
    Rectify,
    Bandpass,
    Highpass,
    Lowpass,
    Notch,
    ZeroCrossings,
    SlopeSignChanges,
    WaveformLength,
    Diff,
)

# Normalization
from myoverse.transforms.normalize import (
    ZScore,
    MinMax,
    Normalize,
    InstanceNorm,
    LayerNorm,
    BatchNorm,
    ClampRange,
    Standardize,
)

# Generic array operations
from myoverse.transforms.generic import (
    Reshape,
    Index,
    Flatten,
    Squeeze,
    Unsqueeze,
    Transpose,
    Mean,
    Sum,
    Stack,
    Concat,
    Lambda,
    Identity,
    Repeat,
    Pad,
)

# Augmentations
from myoverse.transforms.augment import (
    GaussianNoise,
    MagnitudeWarp,
    TimeWarp,
    Dropout,
    ChannelShuffle,
    TimeShift,
    Scale,
    Cutout,
)

# Spatial / grid-aware
from myoverse.transforms.spatial import (
    SpatialFilter,
    NDD,
    LSD,
    TSD,
    IB2,
    SPATIAL_KERNELS,
)

__all__ = [
    # Compose
    "Compose",
    # Base
    "Transform",
    "TransformError",
    "named_tensor",
    "emg_tensor",
    "get_dim_index",
    "align_tensors",
    # Temporal / signal processing
    "RMS",
    "MAV",
    "VAR",
    "Rectify",
    "Bandpass",
    "Highpass",
    "Lowpass",
    "Notch",
    "ZeroCrossings",
    "SlopeSignChanges",
    "WaveformLength",
    "Diff",
    # Normalization
    "ZScore",
    "MinMax",
    "Normalize",
    "InstanceNorm",
    "LayerNorm",
    "BatchNorm",
    "ClampRange",
    "Standardize",
    # Generic array ops
    "Reshape",
    "Index",
    "Flatten",
    "Squeeze",
    "Unsqueeze",
    "Transpose",
    "Mean",
    "Sum",
    "Stack",
    "Concat",
    "Lambda",
    "Identity",
    "Repeat",
    "Pad",
    # Augmentation
    "GaussianNoise",
    "MagnitudeWarp",
    "TimeWarp",
    "Dropout",
    "ChannelShuffle",
    "TimeShift",
    "Scale",
    "Cutout",
    # Spatial / grid-aware
    "SpatialFilter",
    "NDD",
    "LSD",
    "TSD",
    "IB2",
    "SPATIAL_KERNELS",
]
