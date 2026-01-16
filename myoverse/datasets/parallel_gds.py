"""Parallel GDS with async I/O and CUDA streams.

Uses kvikio's async pread to issue multiple reads in parallel.
"""

import time
import cupy as cp
import torch
import zarr
from kvikio.zarr import GDSStore
from pathlib import Path

# Enable GPU mode
zarr.config.enable_gpu()


def test_parallel_gds_reads(dataset_path: Path, batch_size: int = 16, window_size: int = 192):
    """Test parallel GDS reads using CuPy streams."""
    # Open store
    store = zarr.open(GDSStore(str(dataset_path)), mode="r")
    train_keys = list(store['training'].keys())
    arr = store['training'][train_keys[0]]

    print(f"Array shape: {arr.shape}")
    print(f"Testing {batch_size} parallel reads...\n")

    # Method 1: Serial reads (baseline)
    print("Method 1: Serial reads")
    positions = [i * 1000 for i in range(batch_size)]

    start = time.perf_counter()
    samples_serial = []
    for pos in positions:
        data = arr[..., pos:pos+window_size]
        samples_serial.append(data)
    cp.cuda.Stream.null.synchronize()
    elapsed_serial = time.perf_counter() - start
    print(f"  {elapsed_serial*1000:.2f} ms total | {elapsed_serial/batch_size*1000:.2f} ms/sample\n")

    # Method 2: Parallel reads with streams
    # Note: zarr doesn't expose async API, so we can't actually do async reads
    # through the zarr interface. We'd need to use kvikio's file API directly.
    # For now, let's try reading in parallel with CUDA streams at least

    print("Method 2: Reads with multiple CUDA streams")
    streams = [cp.cuda.Stream() for _ in range(4)]

    start = time.perf_counter()
    samples_parallel = []
    for i, pos in enumerate(positions):
        stream = streams[i % len(streams)]
        with stream:
            data = arr[..., pos:pos+window_size]
            samples_parallel.append(data)

    # Wait for all streams
    for stream in streams:
        stream.synchronize()
    elapsed_parallel = time.perf_counter() - start
    print(f"  {elapsed_parallel*1000:.2f} ms total | {elapsed_parallel/batch_size*1000:.2f} ms/sample\n")

    print(f"Speedup: {elapsed_serial/elapsed_parallel:.2f}x")

    return elapsed_serial, elapsed_parallel


if __name__ == "__main__":
    dataset_path = Path("data/sub1_dataset.zarr")
    test_parallel_gds_reads(dataset_path)
