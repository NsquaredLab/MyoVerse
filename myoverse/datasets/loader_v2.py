"""Multi-modal data loaders with direct tensor loading.

Loads from zarr directly to GPU tensors with named dimensions.
Optionally uses kvikio for GPU Direct Storage (zero-copy disk -> GPU).

Example
-------
>>> from myoverse.transforms import Compose, ZScore, RMS
>>>
>>> # Load directly to GPU with named tensors
>>> dm = DataModule(
...     "data.zarr",
...     inputs=["emg"],
...     targets=["kinematics"],
...     device="cuda",  # Direct GPU loading (uses kvikio if available)
...     train_transform=Compose([ZScore(), RMS(200)]),
... )
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
import zarr
import lightning as L
from torch.utils.data import DataLoader, Dataset

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


class ContinuousDataset(Dataset):
    """Dataset that loads windows from zarr with configurable output format.

    Output format is controlled by the `device` parameter:
    - `device=None` → numpy arrays (no transforms applied)
    - `device="cpu"` → named tensors on CPU
    - `device="cuda"` → named tensors on GPU (uses kvikio GDS if available)

    Parameters
    ----------
    zarr_path : Path | str
        Path to the Zarr dataset.
    split : str
        Dataset split ('training', 'validation', 'testing').
    inputs : list[str]
        Modality names to use as model inputs.
    targets : list[str]
        Modality names to use as model targets.
    window_size : int
        Number of samples per window.
    window_stride : int | None
        Stride between windows. If None, uses random positions.
    n_windows : int | None
        Number of windows per epoch. Required if window_stride is None.
    transform : callable | None
        Transform to apply to input data (only when device is set).
    target_transform : callable | None
        Transform to apply to target data (only when device is set).
    seed : int | None
        Random seed for reproducible window positions.
    device : str | torch.device | None
        Output format and device:
        - None: return numpy arrays
        - "cpu": return named tensors on CPU
        - "cuda": return named tensors on GPU (uses kvikio if available)
    dtype : torch.dtype
        Data type for tensors. Default: torch.float32.

    Examples
    --------
    >>> # Return numpy arrays (no transforms)
    >>> ds = ContinuousDataset("data.zarr", split="training", device=None)
    >>> inputs, targets = ds[0]
    >>> type(inputs["emg"])  # numpy.ndarray
    >>>
    >>> # Return tensors on CPU
    >>> ds = ContinuousDataset("data.zarr", split="training", device="cpu")
    >>> inputs["emg"].device  # cpu
    >>>
    >>> # Return tensors on GPU (uses kvikio GDS if available)
    >>> ds = ContinuousDataset(
    ...     "data.zarr",
    ...     split="training",
    ...     device="cuda",
    ...     transform=Compose([ZScore(), RMS(50)]),
    ... )
    >>> inputs["emg"].device  # cuda:0
    >>> inputs["emg"].names   # ('channel', 'time')
    """

    def __init__(
        self,
        zarr_path: Path | str,
        split: str = "training",
        inputs: Sequence[str] = ("emg",),
        targets: Sequence[str] = ("kinematics",),
        window_size: int = 200,
        window_stride: int | None = None,
        n_windows: int | None = None,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        seed: int | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        cache_in_ram: bool = True,
    ):
        self.zarr_path = Path(zarr_path)
        self.split = split
        self.inputs = list(inputs)
        self.targets = list(targets)
        self.window_size = window_size
        self.window_stride = window_stride
        self.transform = transform
        self.target_transform = target_transform
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
        # GDS is only beneficial for repeated disk I/O
        self._use_gds = (
            HAS_KVIKIO
            and torch.cuda.is_available()
            and not cache_in_ram  # Don't use GDS if caching - avoids worker issues
        )
        self._gds_to_cpu = self._use_gds and (self.device is None or self.device.type == "cpu")

        # Open zarr store (with GDS if available)
        if self._use_gds:
            # Enable GPU mode in zarr so slicing returns CuPy arrays
            zarr.config.enable_gpu()
            self._store = zarr.open(GDSStore(str(self.zarr_path)), mode="r")
        else:
            self._store = zarr.open(str(self.zarr_path), mode="r")

        # Get metadata
        self._modalities = self._store.attrs.get("modalities", [])
        self._tasks = self._store.attrs.get("tasks", [])
        self._dims_info = self._store.attrs.get("dims", {})

        # Get split group
        if split not in self._store:
            raise FileNotFoundError(f"Split '{split}' not found in {self.zarr_path}")
        self._split_group = self._store[split]

        # Cache arrays in RAM if requested
        if self.cache_in_ram:
            print(f"Loading {split} split into RAM...")
            import time

            # Optimize: Use concurrent I/O for faster loading
            zarr.config.set({'async.concurrency': 32})

            start = time.perf_counter()
            self._ram_cache = {}
            total_size = 0

            for arr_name in self._split_group.keys():
                arr = self._split_group[arr_name]
                # Load to numpy (convert from CuPy if using GDS)
                if self._use_gds and cp is not None:
                    data = arr[:]  # CuPy array
                    self._ram_cache[arr_name] = cp.asnumpy(data)
                else:
                    self._ram_cache[arr_name] = np.asarray(arr[:])
                total_size += self._ram_cache[arr_name].nbytes

            elapsed = time.perf_counter() - start
            print(f"  Loaded {total_size / (1024**3):.2f} GB in {elapsed:.2f}s ({total_size / (1024**2) / elapsed:.1f} MB/s)")

            # Reset zarr config after loading to avoid issues with workers
            # Workers will inherit zarr config, and GPU mode causes crashes
            if self._use_gds:
                zarr.config.reset()
        else:
            self._ram_cache = None

        # Validate inputs/targets exist
        all_requested = set(self.inputs + self.targets)
        available = set(self._modalities)
        missing = all_requested - available
        if missing:
            raise ValueError(
                f"Requested modalities {missing} not in dataset. "
                f"Available: {available}"
            )

        # Build variable lists for each modality
        self._input_vars: dict[str, list[str]] = {mod: [] for mod in self.inputs}
        self._target_vars: dict[str, list[str]] = {mod: [] for mod in self.targets}

        for arr_name in self._split_group.keys():
            for mod in self.inputs:
                if arr_name.startswith(f"{mod}_"):
                    self._input_vars[mod].append(arr_name)
            for mod in self.targets:
                if arr_name.startswith(f"{mod}_"):
                    self._target_vars[mod].append(arr_name)

        # Sort for consistent ordering
        for mod in self._input_vars:
            self._input_vars[mod].sort()
        for mod in self._target_vars:
            self._target_vars[mod].sort()

        # Get recording lengths from first input modality
        first_input = self.inputs[0]
        self._recording_lengths = []
        self._recording_vars = []

        for var in self._input_vars[first_input]:
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
        # Remove unpickleable zarr/GDS objects - workers will reinitialize if needed
        state['_store'] = None
        state['_split_group'] = None
        # Remove RNG - will be recreated in workers using self.seed
        state['_rng'] = None
        return state

    def __setstate__(self, state):
        """Restore state after unpickling (in worker processes)."""
        try:
            self.__dict__.update(state)

            # Recreate RNG in workers using the saved seed
            self._rng = np.random.default_rng(self.seed)

            # Workers must not use GPU/GDS - reset zarr config to defaults
            # Skip this if zarr not imported (shouldn't happen, but be safe)
            try:
                zarr.config.reset()
            except Exception:
                pass  # Ignore zarr config errors

            # If we have RAM cache, we don't need to reinitialize the store
            if self._ram_cache is not None:
                # Workers only access RAM cache, no zarr/GDS needed
                return

            # Otherwise, reinitialize zarr store (but avoid GDS in workers)
            # Workers should use regular zarr to avoid CUDA/GDS issues
            self._store = zarr.open(str(self.zarr_path), mode="r")
            self._split_group = self._store[self.split]
            # Disable GDS for workers to avoid CUDA issues
            self._use_gds = False
            self._gds_to_cpu = False
        except Exception as e:
            # Log any errors during unpickling for debugging
            import sys
            print(f"ERROR in __setstate__: {e}", file=sys.stderr)
            raise

    def get_sample_shape(self, modality: str) -> tuple[int, ...]:
        """Get the shape of a sample for a given modality (without loading data).

        Returns shape without time dimension: (channels,) for emg, (joints, xyz) for kinematics.
        """
        # Get list of variables for this modality
        var_list = self._input_vars.get(modality) or self._target_vars.get(modality)
        if not var_list:
            raise ValueError(f"Modality '{modality}' not found in inputs or targets")

        # Get first variable's array to check shape
        first_var = var_list[0]
        arr = self._split_group[first_var]

        # Return shape without time dimension (last axis)
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

        return self._valid_ranges[0][0], 0

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

    def _get_dim_names(self, modality: str) -> tuple[str, ...]:
        """Get dimension names for a modality from metadata."""
        if modality in self._dims_info:
            return tuple(self._dims_info[modality])

        # Fallback to conventions
        if modality == "emg":
            return ("channel", "time")
        elif modality == "kinematics":
            return ("joint", "time")
        elif modality == "eeg":
            return ("electrode", "time")
        else:
            return ("channel", "time")

    def _to_tensor(self, data) -> torch.Tensor:
        """Convert data to tensor on target device.

        Uses DLPack for zero-copy transfer when using GDS (cupy -> torch).
        """
        # Check if data is actually a CuPy array (not just if GDS is enabled)
        if cp is not None and isinstance(data, cp.ndarray):
            if self._gds_to_cpu:
                # GDS→GPU→CPU path for workers with GDS
                # Convert CuPy to numpy (GPU→CPU copy)
                data = cp.asnumpy(data)
                tensor = torch.from_numpy(data)
                return tensor.to(dtype=self.dtype)
            else:
                # GDS→GPU path for direct GPU loading
                # Use DLPack for zero-copy to torch
                # Ensure contiguous
                if not data.flags.c_contiguous:
                    data = cp.ascontiguousarray(data)
                tensor = torch.from_dlpack(data)
                return tensor.to(dtype=self.dtype)
        else:
            # Regular path: numpy -> torch
            tensor = torch.from_numpy(np.ascontiguousarray(data))
            if self.device is not None:
                tensor = tensor.to(device=self.device, dtype=self.dtype)
            else:
                tensor = tensor.to(dtype=self.dtype)
            return tensor

    def __len__(self) -> int:
        return self._n_windows

    def __getitem__(
        self, idx: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        # Get window position
        if self._random_mode:
            rec_idx, local_pos = self._sample_random_position()
        else:
            rec_idx, local_pos = self._get_deterministic_position(idx)

        task = self._get_task_for_recording(rec_idx)

        # Extract input windows
        inputs = {}
        for mod in self.inputs:
            var_name = f"{mod}_{task}"
            # Check RAM cache first (avoids CUDA/zarr operations in workers)
            exists = var_name in self._ram_cache if self._ram_cache else var_name in self._split_group
            if exists:
                # Read from RAM cache if available, otherwise from zarr
                if self._ram_cache is not None:
                    arr = self._ram_cache[var_name]
                    data = arr[..., local_pos : local_pos + self.window_size]
                else:
                    arr = self._split_group[var_name]
                    data = arr[..., local_pos : local_pos + self.window_size]

                if self.device is None:
                    # Return numpy array (no transforms)
                    inputs[mod] = np.ascontiguousarray(data)
                else:
                    # Convert to named tensor on device
                    tensor = self._to_tensor(data)
                    names = self._get_dim_names(mod)
                    tensor = tensor.rename(*names)
                    # Apply transform
                    if self.transform is not None:
                        tensor = self.transform(tensor)
                    inputs[mod] = tensor

        # Extract target windows
        targets = {}
        for mod in self.targets:
            var_name = f"{mod}_{task}"
            # Check RAM cache first (avoids CUDA/zarr operations in workers)
            exists = var_name in self._ram_cache if self._ram_cache else var_name in self._split_group
            if exists:
                # Read from RAM cache if available, otherwise from zarr
                if self._ram_cache is not None:
                    arr = self._ram_cache[var_name]
                    data = arr[..., local_pos : local_pos + self.window_size]
                else:
                    arr = self._split_group[var_name]
                    data = arr[..., local_pos : local_pos + self.window_size]

                if self.device is None:
                    # Return numpy array (no transforms)
                    targets[mod] = np.ascontiguousarray(data)
                else:
                    # Convert to named tensor on device
                    tensor = self._to_tensor(data)
                    names = self._get_dim_names(mod)
                    tensor = tensor.rename(*names)
                    if self.target_transform is not None:
                        tensor = self.target_transform(tensor)
                    targets[mod] = tensor

        return inputs, targets

    def reseed(self, seed: int | None = None) -> None:
        """Reseed the random number generator."""
        self._rng = np.random.default_rng(seed)


def _collate_dicts(batch):
    """Custom collate function for dict outputs.

    Handles both numpy arrays and tensors.
    Strips named tensor names since most models don't support them.
    """
    inputs_list = [b[0] for b in batch]
    targets_list = [b[1] for b in batch]

    # Check if numpy or tensor
    first_input = list(inputs_list[0].values())[0]
    is_numpy = isinstance(first_input, np.ndarray)

    # Stack each modality
    inputs = {}
    for key in inputs_list[0]:
        items = [b[key] for b in inputs_list]
        if is_numpy:
            # Stack numpy arrays
            inputs[key] = np.stack(items)
        else:
            # Stack tensors (strip names - models don't support them)
            items_unnamed = [t.rename(None) if t.names[0] is not None else t for t in items]
            inputs[key] = torch.stack(items_unnamed)

    targets = {}
    for key in targets_list[0]:
        items = [b[key] for b in targets_list]
        if is_numpy:
            targets[key] = np.stack(items)
        else:
            items_unnamed = [t.rename(None) if t.names[0] is not None else t for t in items]
            targets[key] = torch.stack(items_unnamed)

    # Return directly if single input/target
    if len(inputs) == 1 and len(targets) == 1:
        return list(inputs.values())[0], list(targets.values())[0]

    return inputs, targets


class DataModule(L.LightningDataModule):
    """Lightning DataModule with configurable output format.

    Output format is controlled by the `device` parameter:
    - `device=None` → numpy arrays (no transforms)
    - `device="cpu"` → named tensors on CPU
    - `device="cuda"` → named tensors on GPU (uses kvikio GDS if available)

    Parameters
    ----------
    data_path : Path | str
        Path to the Zarr dataset.
    inputs : list[str]
        Modality names to use as model inputs.
    targets : list[str]
        Modality names to use as model targets.
    batch_size : int
        Batch size for all dataloaders.
    window_size : int
        Window size in samples.
    window_stride : int | None
        Window stride for validation/test.
    n_windows_per_epoch : int | None
        Number of random windows per training epoch.
    num_workers : int
        Number of dataloader workers.
    train_transform : callable | None
        Transform for training inputs (only when device is set).
    val_transform : callable | None
        Transform for validation inputs (only when device is set).
    test_transform : callable | None
        Transform for test inputs (only when device is set).
    target_transform : callable | None
        Transform for targets (only when device is set).
    pin_memory : bool
        Pin memory for faster GPU transfer.
    persistent_workers : bool
        Keep workers alive between epochs.
    device : str | torch.device | None
        Output format and device:
        - None: return numpy arrays
        - "cpu": return named tensors on CPU
        - "cuda": return named tensors on GPU (uses kvikio if available)
    dtype : torch.dtype
        Data type for tensors.

    Examples
    --------
    >>> # Return numpy arrays
    >>> dm = DataModule("data.zarr", device=None)
    >>> inputs, targets = next(iter(dm.train_dataloader()))
    >>> type(inputs)  # numpy.ndarray
    >>>
    >>> # Return tensors on GPU with transforms
    >>> dm = DataModule(
    ...     "data.zarr",
    ...     inputs=["emg"],
    ...     targets=["kinematics"],
    ...     window_size=200,
    ...     n_windows_per_epoch=10000,
    ...     device="cuda",
    ...     train_transform=Compose([ZScore(), RMS(50)]),
    ... )
    >>> inputs, targets = next(iter(dm.train_dataloader()))
    >>> inputs.names  # ('batch', 'channel', 'time')
    >>> inputs.device # cuda:0
    """

    def __init__(
        self,
        data_path: Path | str,
        inputs: Sequence[str] = ("emg",),
        targets: Sequence[str] = ("kinematics",),
        batch_size: int = 32,
        window_size: int = 200,
        window_stride: int | None = None,
        n_windows_per_epoch: int | None = None,
        num_workers: int = 4,
        train_transform: Callable | None = None,
        val_transform: Callable | None = None,
        test_transform: Callable | None = None,
        target_transform: Callable | None = None,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        cache_in_ram: bool = True,
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.inputs = list(inputs)
        self.targets = list(targets)
        self.batch_size = batch_size
        self.window_size = window_size
        self.window_stride = window_stride
        self.n_windows_per_epoch = n_windows_per_epoch
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform or val_transform
        self.target_transform = target_transform
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.device = device
        self.dtype = dtype
        self.cache_in_ram = cache_in_ram

        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")

        if n_windows_per_epoch is None and window_stride is None:
            raise ValueError("Need n_windows_per_epoch or window_stride")

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str | None = None) -> None:
        """Setup datasets for each stage."""
        if stage == "fit" or stage is None:
            self.train_dataset = ContinuousDataset(
                self.data_path,
                split="training",
                inputs=self.inputs,
                targets=self.targets,
                window_size=self.window_size,
                n_windows=self.n_windows_per_epoch,
                transform=self.train_transform,
                target_transform=self.target_transform,
                device=self.device,
                dtype=self.dtype,
                cache_in_ram=self.cache_in_ram,
            )
            self.val_dataset = ContinuousDataset(
                self.data_path,
                split="validation",
                inputs=self.inputs,
                targets=self.targets,
                window_size=self.window_size,
                window_stride=self.window_stride or self.window_size,
                transform=self.val_transform,
                target_transform=self.target_transform,
                device=self.device,
                dtype=self.dtype,
                cache_in_ram=self.cache_in_ram,
            )

        if stage == "test" or stage is None:
            self.test_dataset = ContinuousDataset(
                self.data_path,
                split="testing",
                inputs=self.inputs,
                targets=self.targets,
                window_size=self.window_size,
                window_stride=self.window_stride or self.window_size,
                transform=self.test_transform,
                target_transform=self.target_transform,
                device=self.device,
                dtype=self.dtype,
                cache_in_ram=self.cache_in_ram,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory and self.device is None,
            persistent_workers=self.persistent_workers,
            collate_fn=_collate_dicts,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory and self.device is None,
            persistent_workers=self.persistent_workers,
            collate_fn=_collate_dicts,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory and self.device is None,
            persistent_workers=self.persistent_workers,
            collate_fn=_collate_dicts,
        )


# Backwards compatibility aliases
EMGContinuousDataset = ContinuousDataset
EMGDataModule = DataModule
