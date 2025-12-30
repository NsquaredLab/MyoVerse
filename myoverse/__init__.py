"""
MyoVerse - The AI toolkit for myocontrol research

MyoVerse is a cutting-edge research companion for unlocking the secrets hidden within 
biomechanical data. It's specifically designed for exploring the complex interplay 
between electromyography (EMG) signals, kinematics (movement), and kinetics (forces).

Leveraging PyTorch and PyTorch Lightning, MyoVerse provides:
- Data loaders and preprocessing filters tailored for biomechanical signals
- Peer-reviewed AI models and components for analysis and prediction tasks
- Essential utilities to streamline the research workflow

MyoVerse aims to accelerate research in predicting movement from muscle activity, 
analyzing forces during motion, and developing novel AI approaches for biomechanical challenges.

Note: MyoVerse is built for research and is continuously evolving.
"""

from icecream import install, ic
import datetime
import importlib.metadata
import os
import toml

# Initialize zarr with zarrs codec pipeline (must be done before any zarr imports)
from myoverse.io import zarr_io as _zarr_io  # noqa: F401

# Try multiple methods to get the version
try:
    # Method 1: Try to read from pyproject.toml first
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pyproject_path = os.path.join(package_root, "pyproject.toml")
    
    if os.path.exists(pyproject_path):
        pyproject_data = toml.load(pyproject_path)
        __version__ = pyproject_data.get("project", {}).get("version", "unknown")
    
    # Method 2: If that fails or version is still unknown, try importlib.metadata
    if __version__ == "unknown":
        __version__ = importlib.metadata.version("MyoVerse")
        
except Exception:
    # If all methods fail, we at least have a default
    __version__ = "unknown"

install()


# Define a function to generate a prefix with ISO timestamp
def timestamp_prefix():
    timestamp = datetime.datetime.now().isoformat(timespec="milliseconds")
    return f"{timestamp} | MyoVerse {__version__} | "


# Configure IceCream to use the timestamp prefix function
ic.configureOutput(includeContext=True, prefix=timestamp_prefix)


# Top-level API for creating EMG arrays
def emg_xarray(
    data,
    grid_layouts=None,
    fs=2048.0,
    dims=("channel", "time"),
    **attrs,
):
    """Create an EMG DataArray with grid layouts and metadata.

    This is the recommended way to create EMG data for use with transforms.
    Grid layouts are stored in attrs for spatial transforms to use.

    Parameters
    ----------
    data : np.ndarray
        EMG data array.
    grid_layouts : list[np.ndarray] | None
        List of 2D arrays mapping grid positions to channel indices.
        Each array element contains the electrode index (0-based), or -1 for gaps.
    fs : float
        Sampling frequency in Hz.
    dims : tuple[str, ...]
        Dimension names. Default: ("channel", "time").
    **attrs
        Additional attributes to store.

    Returns
    -------
    xr.DataArray
        EMG DataArray with grid_layouts and sampling_frequency in attrs.

    Examples
    --------
    >>> import myoverse
    >>> from myoverse.datatypes import create_grid_layout
    >>>
    >>> # Create grid layouts
    >>> grid1 = create_grid_layout(8, 8)  # 64 electrodes
    >>> grid2 = create_grid_layout(4, 4)  # 16 electrodes
    >>> grid2[grid2 >= 0] += 64  # Offset indices
    >>>
    >>> # Create EMG array with grid info
    >>> emg = myoverse.emg_xarray(
    ...     data,
    ...     grid_layouts=[grid1, grid2],
    ...     fs=2048.0,
    ... )
    >>>
    >>> # Use with transforms
    >>> from myoverse.transforms import NDD, Pipeline, Bandpass
    >>> pipeline = Pipeline([
    ...     Bandpass(20, 450, fs=2048, dim="time"),
    ...     NDD(grids="all"),
    ... ])
    >>> filtered = pipeline(emg)
    """
    import xarray as xr

    all_attrs = {"sampling_frequency": fs, **attrs}
    if grid_layouts is not None:
        all_attrs["grid_layouts"] = grid_layouts

    return xr.DataArray(data, dims=dims, attrs=all_attrs)


def emg_tensor(
    data,
    fs=2048.0,
    grid_layouts=None,
    device=None,
    dtype=None,
):
    """Create an EMG tensor with named dimensions for GPU-accelerated processing.

    This is the entry point for GPU-accelerated EMG processing with named tensors.
    Use this instead of emg_xarray() when you need GPU acceleration.

    Parameters
    ----------
    data : array-like
        EMG data. Shape should be (channels, time) or (batch, channels, time).
    fs : float
        Sampling frequency in Hz.
    grid_layouts : list[np.ndarray] | None
        Grid layouts for spatial transforms.
    device : str | torch.device | None
        Device to place tensor on ('cuda', 'cpu', etc.).
    dtype : torch.dtype | None
        Data type for the tensor. Default: torch.float32.

    Returns
    -------
    torch.Tensor
        Named tensor on the specified device with dimension names.

    Note
    ----
    Metadata (fs, grid_layouts) is stored as tensor attributes.

    Examples
    --------
    >>> import myoverse
    >>> import numpy as np
    >>>
    >>> # Create EMG tensor on GPU
    >>> data = np.random.randn(64, 2048)
    >>> emg = myoverse.emg_tensor(data, fs=2048, device='cuda')
    >>> emg.names  # ('channel', 'time')
    >>> emg.device  # cuda:0
    >>>
    >>> # Use with tensor transforms
    >>> from myoverse.transforms.tensor import Pipeline, ZScore, RMS
    >>> pipeline = Pipeline([ZScore(), RMS(window_size=200)])
    >>> processed = pipeline(emg)
    """
    import torch

    if dtype is None:
        dtype = torch.float32

    # Convert to tensor if needed
    if not isinstance(data, torch.Tensor):
        data = torch.as_tensor(data, dtype=dtype)
    else:
        data = data.to(dtype=dtype)

    # Move to device
    if device is not None:
        data = data.to(device=device)

    # Add dimension names
    if data.ndim == 2:
        names = ('channel', 'time')
    elif data.ndim == 3:
        names = ('batch', 'channel', 'time')
    else:
        names = tuple(f'dim_{i}' for i in range(data.ndim))

    data = data.rename(*names)

    # Store metadata as attributes
    data.fs = fs
    data.sampling_frequency = fs
    if grid_layouts is not None:
        data.grid_layouts = grid_layouts

    return data
