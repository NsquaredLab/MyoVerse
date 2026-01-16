"""Batched GDS Dataset for optimal kvikio performance.

Reads entire batches at once to amortize GDS overhead.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset, Sampler

# Suppress named tensor warning
warnings.filterwarnings("ignore", category=UserWarning, message=".*Named tensors.*")

try:
    import cupy as cp
    from kvikio.zarr import GDSStore

    HAS_KVIKIO = True
except ImportError:
    HAS_KVIKIO = False
    GDSStore = None
    cp = None


class BatchedGDSDataset(Dataset):
    """Dataset that reads entire batches at once for GDS efficiency.

    Instead of reading samples individually, reads a large chunk covering
    all samples in a batch, then slices in GPU memory. This amortizes
    GDS overhead and is ~7x faster than individual reads.

    Parameters
    ----------
    zarr_path : Path
        Path to zarr dataset
    split : str
        Split name ('training', 'testing', 'validation')
    inputs : list[str]
        Input modality names
    targets : list[str]
        Target modality names
    window_size : int
        Window size in samples
    batch_size : int
        Batch size (required for batched reading)
    n_windows : int
        Number of windows per epoch
    transform : callable
        Transform to apply to inputs
    dtype : torch.dtype
        Output dtype
    """

    def __init__(
        self,
        zarr_path: Path | str,
        split: str,
        inputs: list[str],
        targets: list[str],
        window_size: int,
        batch_size: int,
        n_windows: int = 1000,
        transform=None,
        dtype: torch.dtype = torch.float32,
    ):
        self.zarr_path = Path(zarr_path)
        self.split = split
        self.inputs = inputs
        self.targets = targets
        self.window_size = window_size
        self.batch_size = batch_size
        self.n_windows = n_windows
        self.transform = transform
        self.dtype = dtype

        if not HAS_KVIKIO:
            raise ImportError("kvikio not available - install with: pip install kvikio")

        # Enable GPU mode and open with GDS
        zarr.config.enable_gpu()
        self._store = zarr.open(GDSStore(str(self.zarr_path)), mode="r")

        # Get split group
        self._split_group = self._store[split]

        # Get recording info
        self._recording_vars = [k for k in self._split_group.keys() if k.startswith(inputs[0])]
        self._recording_lengths = []
        for var in self._recording_vars:
            arr = self._split_group[var]
            self._recording_lengths.append(arr.shape[-1])

        # Build valid ranges (store local positions per recording)
        self._valid_ranges = []
        for rec_idx, length in enumerate(self._recording_lengths):
            if length >= window_size:
                # Max valid starting position in this recording
                max_start = length - window_size
                self._valid_ranges.append((rec_idx, 0, max_start))

        self._total_valid = sum(end - start + 1 for _, start, end in self._valid_ranges)
        self._rng = np.random.default_rng()

    def _sample_batch_positions(self) -> list[tuple[int, int, str]]:
        """Sample batch_size random valid positions.

        Returns list of (rec_idx, local_pos, task)
        """
        positions = []
        for _ in range(self.batch_size):
            pos = self._rng.integers(0, self._total_valid)
            cumsum = 0
            for rec_idx, start, end in self._valid_ranges:
                range_size = end - start + 1
                if pos < cumsum + range_size:
                    # Offset within this recording's valid range
                    offset_in_range = pos - cumsum
                    # Actual position in the recording (start is always 0 now)
                    local_pos = start + offset_in_range
                    # Get task name
                    var_name = self._recording_vars[rec_idx]
                    task = var_name.split("_", 1)[1] if "_" in var_name else "default"
                    positions.append((rec_idx, local_pos, task))
                    break
                cumsum += range_size
        return positions

    def _read_batch_chunked(self, positions: list[tuple[int, int, str]], modality: str) -> list:
        """Read batch by grouping samples from same recording into efficient chunks."""
        # Group by recording+task
        by_recording = {}
        for idx, (rec_idx, local_pos, task) in enumerate(positions):
            key = (rec_idx, task)
            if key not in by_recording:
                by_recording[key] = []
            by_recording[key].append((idx, local_pos))

        # Prepare output list
        batch_data = [None] * len(positions)

        # Read each recording's samples
        for (rec_idx, task), items in by_recording.items():
            full_var_name = f"{modality}_{task}"

            if full_var_name not in self._split_group:
                # Fill with zeros if not found
                ref_var = f"{self.inputs[0]}_{task}"
                if ref_var in self._split_group:
                    ref_shape = self._split_group[ref_var].shape[:-1]
                    for idx, _ in items:
                        batch_data[idx] = cp.zeros(ref_shape + (self.window_size,), dtype=cp.float32)
                continue

            arr = self._split_group[full_var_name]

            # Sort by position
            items.sort(key=lambda x: x[1])

            # Group into efficient chunks (max gap = 3x window_size)
            MAX_GAP = self.window_size * 3
            chunks = []
            current_chunk = [items[0]]

            for item in items[1:]:
                gap = item[1] - current_chunk[-1][1]
                if gap <= MAX_GAP:
                    current_chunk.append(item)
                else:
                    chunks.append(current_chunk)
                    current_chunk = [item]
            chunks.append(current_chunk)

            # Read each chunk
            for chunk in chunks:
                min_pos = chunk[0][1]
                max_pos = chunk[-1][1]
                chunk_size = max_pos - min_pos + self.window_size

                # Read big chunk
                big_chunk = arr[..., min_pos : min_pos + chunk_size]

                # Slice individual samples
                for idx, local_pos in chunk:
                    offset = local_pos - min_pos
                    sample = big_chunk[..., offset : offset + self.window_size]
                    batch_data[idx] = sample

        return batch_data

    def get_batch(self, batch_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get an entire batch at once (for use with custom collate)."""
        # Sample positions
        positions = self._sample_batch_positions()

        # Read inputs
        input_batches = {}
        for mod in self.inputs:
            batch_data = self._read_batch_chunked(positions, mod)
            # Convert to tensors
            tensors = [torch.from_dlpack(d) for d in batch_data]
            # Apply transforms
            if self.transform is not None:
                tensors = [self.transform(t.rename("channel", "time")).rename(None) for t in tensors]
            else:
                tensors = [t for t in tensors]
            # Stack
            input_batches[mod] = torch.stack(tensors).to(dtype=self.dtype)

        # Read targets
        target_batches = {}
        for mod in self.targets:
            batch_data = self._read_batch_chunked(positions, mod)
            tensors = [torch.from_dlpack(d).to(dtype=self.dtype) for d in batch_data]
            target_batches[mod] = torch.stack(tensors)

        # Return single tensors if only one modality each
        if len(input_batches) == 1 and len(target_batches) == 1:
            return list(input_batches.values())[0], list(target_batches.values())[0]

        return input_batches, target_batches

    def __len__(self) -> int:
        # Return number of batches, not samples
        return self.n_windows // self.batch_size

    def __getitem__(self, idx: int):
        """Get a batch (idx is batch index, not sample index)."""
        return self.get_batch(idx)


def batched_collate(batch):
    """Collate function that just returns the pre-batched data."""
    # Each "batch" is already a batched tensor, just return the first item
    return batch[0]
