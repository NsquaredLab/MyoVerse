"""Base dataset for windowed multi-modal data loading.

This module provides the paradigm-agnostic infrastructure for loading
windowed data from zarr stores. It handles:
- Zarr I/O with optional GPU Direct Storage (GDS)
- Window sampling (random or deterministic)
- RAM caching for performance
- Device management (CPU/GPU)
- Multiprocessing support

The WindowedDataset class returns all modalities as a dict, without
making assumptions about the learning paradigm (supervised, contrastive, etc.).
Paradigm-specific datasets should subclass WindowedDataset.

Example
-------
>>> from myoverse.datasets.base import WindowedDataset
>>>
>>> # Load all modalities from zarr
>>> ds = WindowedDataset(
...     "data.zip",
...     split="training",
...     modalities=["emg", "kinematics"],
...     window_size=200,
...     n_windows=10000,
...     device="cuda",
... )
>>> data = ds[0]  # dict[str, Tensor] with 'emg' and 'kinematics'
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset
from zarr.storage import ZipStore

# Suppress named tensor experimental warning
warnings.filterwarnings("ignore", category=UserWarning, message=".*Named tensors.*")

# Try to import kvikio for GPU Direct Storage
try:
    import cupy as cp
    from kvikio.zarr import GDSStore
    HAS_KVIKIO = True
except ImportError:
    HAS_KVIKIO = False
    GDSStore = None
    cp = None


class WindowedDataset(Dataset):
    """Base dataset that loads windows from zarr for any modality.

    This is the infrastructure layer - it handles loading, windowing, caching,
    and device management. It returns ALL requested modalities as a dict.

    Subclasses implement paradigm-specific logic (e.g., SupervisedDataset
    splits into inputs/targets, ContrastiveDataset creates augmented views).

    Parameters
    ----------
    zarr_path : Path | str
        Path to the Zarr dataset.
    split : str
        Dataset split ('training', 'validation', 'testing').
    modalities : Sequence[str] | None
        Modality names to load. If None, loads all available modalities.
    window_size : int
        Number of samples per window.
    window_stride : int | None
        Stride between windows. If None, uses random positions.
    n_windows : int | None
        Number of windows per epoch. Required if window_stride is None.
    seed : int | None
        Random seed for reproducible window positions.
    device : torch.device | str | None
        Output device:
        - None: return numpy arrays
        - "cpu": return tensors on CPU
        - "cuda": return tensors on GPU (uses kvikio GDS if available)
    dtype : torch.dtype
        Data type for tensors. Default: torch.float32.
    cache_in_ram : bool
        Cache entire split in RAM for faster access. Default: True.

    Examples
    --------
    >>> # Return numpy arrays
    >>> ds = WindowedDataset("data.zip", modalities=["emg"], device=None)
    >>> data = ds[0]
    >>> type(data["emg"])  # numpy.ndarray
    >>>
    >>> # Return tensors on GPU with named dimensions
    >>> ds = WindowedDataset("data.zip", modalities=["emg"], device="cuda")
    >>> data["emg"].device  # cuda:0
    >>> data["emg"].names   # ('channel', 'time')
    """

    def __init__(
        self,
        zarr_path: Path | str,
        split: str = "training",
        modalities: Sequence[str] | None = None,
        window_size: int = 200,
        window_stride: int | None = None,
        n_windows: int | None = None,
        seed: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        cache_in_ram: bool = True,
    ):
        self.zarr_path = Path(zarr_path)
        self.split = split
        self.window_size = window_size
        self.window_stride = window_stride
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.device = torch.device(device) if device else None
        self.cache_in_ram = cache_in_ram
        self.dtype = dtype

        # Validate path
        if not self.zarr_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.zarr_path}")

        if window_stride is None and n_windows is None:
            raise ValueError("Must specify n_windows when window_stride is None")

        # Use GPU Direct Storage if available and GPU exists
        # Skip GDS if caching in RAM - not worth the complexity for one-time load
        self._use_gds = (
            HAS_KVIKIO
            and torch.cuda.is_available()
            and not cache_in_ram
        )
        self._gds_to_cpu = self._use_gds and (self.device is None or self.device.type == "cpu")

        # Detect if path is a zip file
        self._is_zip = self.zarr_path.suffix.lower() == ".zip"

        # Open zarr store (with GDS if available and not a zip file)
        if self._use_gds and not self._is_zip:
            zarr.config.enable_gpu()
            self._store = zarr.open(GDSStore(str(self.zarr_path)), mode="r")
        elif self._is_zip:
            self._zip_store = ZipStore(self.zarr_path, mode="r")
            self._store = zarr.open(self._zip_store, mode="r")
        else:
            self._store = zarr.open(str(self.zarr_path), mode="r")

        # Get metadata
        self._available_modalities = self._store.attrs.get("modalities", [])
        self._tasks = self._store.attrs.get("tasks", [])
        self._dims_info = self._store.attrs.get("dims", {})

        # Get split group
        if split not in self._store:
            raise FileNotFoundError(f"Split '{split}' not found in {self.zarr_path}")
        self._split_group = self._store[split]

        # Determine modalities to load
        if modalities is None:
            self.modalities = list(self._available_modalities)
        else:
            self.modalities = list(modalities)
            # Validate modalities exist
            missing = set(self.modalities) - set(self._available_modalities)
            if missing:
                raise ValueError(
                    f"Requested modalities {missing} not in dataset. "
                    f"Available: {self._available_modalities}"
                )

        # Cache arrays in RAM if requested
        if self.cache_in_ram:
            print(f"Loading {split} split into RAM...")
            zarr.config.set({'async.concurrency': 32})

            start = time.perf_counter()
            self._ram_cache = {}
            total_size = 0

            for arr_name in self._split_group.keys():
                arr = self._split_group[arr_name]
                if self._use_gds and cp is not None:
                    data = arr[:]
                    self._ram_cache[arr_name] = cp.asnumpy(data)
                else:
                    self._ram_cache[arr_name] = np.asarray(arr[:])
                total_size += self._ram_cache[arr_name].nbytes

            elapsed = time.perf_counter() - start
            print(f"  Loaded {total_size / (1024**3):.2f} GB in {elapsed:.2f}s ({total_size / (1024**2) / elapsed:.1f} MB/s)")

            if self._use_gds:
                zarr.config.reset()
        else:
            self._ram_cache = None

        # Build variable lists for each modality
        self._modality_vars: dict[str, list[str]] = {mod: [] for mod in self.modalities}

        for arr_name in self._split_group.keys():
            for mod in self.modalities:
                if arr_name.startswith(f"{mod}_"):
                    self._modality_vars[mod].append(arr_name)

        # Sort for consistent ordering
        for mod in self._modality_vars:
            self._modality_vars[mod].sort()

        # Get recording lengths from first modality
        first_mod = self.modalities[0]
        self._recording_lengths = []
        self._recording_vars = []

        for var in self._modality_vars[first_mod]:
            arr = self._split_group[var]
            length = arr.shape[-1]  # Time is last dimension
            self._recording_lengths.append(length)
            self._recording_vars.append(var)

        self._total_length = sum(self._recording_lengths)

        # Compute number of windows
        if window_stride is not None:
            self._n_windows = sum(
                max(0, (length - window_size) // window_stride + 1)
                for length in self._recording_lengths
            )
            self._random_mode = False
        else:
            self._n_windows = n_windows
            self._random_mode = True

        self._setup_recording_ranges()

    def __getstate__(self):
        """Prepare state for pickling (used by multiprocessing workers)."""
        state = self.__dict__.copy()
        state['_store'] = None
        state['_split_group'] = None
        state['_rng'] = None
        return state

    def __setstate__(self, state):
        """Restore state after unpickling (in worker processes)."""
        try:
            self.__dict__.update(state)
            self._rng = np.random.default_rng(self.seed)

            try:
                zarr.config.reset()
            except Exception:
                pass

            if self._ram_cache is not None:
                return

            # Reopen store based on file type
            if self._is_zip:
                self._zip_store = ZipStore(self.zarr_path, mode="r")
                self._store = zarr.open(self._zip_store, mode="r")
            else:
                self._store = zarr.open(str(self.zarr_path), mode="r")

            self._split_group = self._store[self.split]
            self._use_gds = False
            self._gds_to_cpu = False
        except Exception as e:
            import sys
            print(f"ERROR in __setstate__: {e}", file=sys.stderr)
            raise

    def get_sample_shape(self, modality: str) -> tuple[int, ...]:
        """Get the shape of a sample for a given modality (without time dimension).

        Parameters
        ----------
        modality : str
            Modality name.

        Returns
        -------
        tuple[int, ...]
            Shape without time dimension.
        """
        var_list = self._modality_vars.get(modality)
        if not var_list:
            raise ValueError(f"Modality '{modality}' not found")

        first_var = var_list[0]
        arr = self._split_group[first_var]
        return arr.shape[:-1]

    def _setup_recording_ranges(self) -> None:
        """Setup valid sampling ranges for each recording."""
        self._valid_ranges = []
        cumsum = 0

        for rec_idx, length in enumerate(self._recording_lengths):
            if length >= self.window_size:
                valid_start = cumsum
                valid_end = cumsum + length - self.window_size
                self._valid_ranges.append((rec_idx, valid_start, valid_end))
            cumsum += length

        if not self._valid_ranges:
            raise ValueError(
                f"No recordings long enough for window_size={self.window_size}"
            )

        self._total_valid = sum(end - start + 1 for _, start, end in self._valid_ranges)

    def _global_to_local(self, global_pos: int) -> tuple[int, int]:
        """Convert global position to (recording_idx, local_position)."""
        cumsum = 0
        for rec_idx, length in enumerate(self._recording_lengths):
            if global_pos < cumsum + length:
                return rec_idx, global_pos - cumsum
            cumsum += length
        raise ValueError(f"Position {global_pos} out of range")

    def _sample_random_position(self) -> tuple[int, int]:
        """Sample a random valid window position."""
        pos = self._rng.integers(0, self._total_valid)

        cumsum = 0
        for rec_idx, start, end in self._valid_ranges:
            range_size = end - start + 1
            if pos < cumsum + range_size:
                global_pos = start + (pos - cumsum)
                return self._global_to_local(global_pos)
            cumsum += range_size

        raise RuntimeError(
            f"Failed to map random position {pos} to valid range "
            f"(total_valid={self._total_valid}, n_ranges={len(self._valid_ranges)})"
        )

    def _get_deterministic_position(self, idx: int) -> tuple[int, int]:
        """Get deterministic window position for given index."""
        cumsum = 0
        for rec_idx, length in enumerate(self._recording_lengths):
            valid_positions = max(
                0, (length - self.window_size) // self.window_stride + 1
            )
            if idx < cumsum + valid_positions:
                local_idx = idx - cumsum
                local_pos = local_idx * self.window_stride
                return rec_idx, local_pos
            cumsum += valid_positions

        raise ValueError(f"Index {idx} out of range")

    def _get_task_for_recording(self, rec_idx: int) -> str:
        """Get the task name for a recording index."""
        var_name = self._recording_vars[rec_idx]
        parts = var_name.split("_", 1)
        return parts[1] if len(parts) > 1 else "default"

    # Default dimension names by modality (fallback when not in metadata)
    _DEFAULT_DIMS: dict[str, tuple[str, ...]] = {
        "emg": ("channel", "time"),
        "kinematics": ("joint", "time"),
        "eeg": ("electrode", "time"),
    }

    def _get_dim_names(self, modality: str) -> tuple[str, ...]:
        """Get dimension names for a modality from metadata."""
        if modality in self._dims_info:
            return tuple(self._dims_info[modality])
        return self._DEFAULT_DIMS.get(modality, ("channel", "time"))

    def _to_tensor(self, data) -> torch.Tensor:
        """Convert data to tensor on target device."""
        if cp is not None and isinstance(data, cp.ndarray):
            if self._gds_to_cpu:
                tensor = torch.from_numpy(cp.asnumpy(data))
            else:
                if not data.flags.c_contiguous:
                    data = cp.ascontiguousarray(data)
                tensor = torch.from_dlpack(data)
        else:
            tensor = torch.from_numpy(np.ascontiguousarray(data))
            if self.device is not None:
                return tensor.to(device=self.device, dtype=self.dtype)

        return tensor.to(dtype=self.dtype)

    def _load_window(self, var_name: str, local_pos: int, modality: str) -> torch.Tensor | np.ndarray:
        """Load a window for a variable and convert to tensor.

        Parameters
        ----------
        var_name : str
            Variable name in zarr (e.g., "emg_task1").
        local_pos : int
            Starting position within the recording.
        modality : str
            Modality name for dimension info.

        Returns
        -------
        torch.Tensor | np.ndarray
            Window data as tensor (if device set) or numpy array.
        """
        if self._ram_cache is not None:
            arr = self._ram_cache[var_name]
        else:
            arr = self._split_group[var_name]

        # Validate window fits within recording
        end_pos = local_pos + self.window_size
        if end_pos > arr.shape[-1]:
            raise ValueError(
                f"Window [{local_pos}:{end_pos}] exceeds recording length {arr.shape[-1]} "
                f"for variable {var_name}"
            )

        data = arr[..., local_pos : end_pos]

        if self.device is None:
            return np.ascontiguousarray(data)
        else:
            tensor = self._to_tensor(data)
            names = self._get_dim_names(modality)
            tensor = tensor.rename(*names)
            return tensor

    def __len__(self) -> int:
        return self._n_windows

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | np.ndarray]:
        """Load windows for all modalities.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        dict[str, torch.Tensor | np.ndarray]
            Dict mapping modality names to data windows.
        """
        # Get window position
        if self._random_mode:
            rec_idx, local_pos = self._sample_random_position()
        else:
            rec_idx, local_pos = self._get_deterministic_position(idx)

        task = self._get_task_for_recording(rec_idx)

        # Extract windows for all modalities
        data = {}
        cache = self._ram_cache or self._split_group
        for mod in self.modalities:
            var_name = f"{mod}_{task}"
            if var_name in cache:
                data[mod] = self._load_window(var_name, local_pos, mod)

        return data

    def reseed(self, seed: int | None = None) -> None:
        """Reseed the random number generator."""
        self._rng = np.random.default_rng(seed)
