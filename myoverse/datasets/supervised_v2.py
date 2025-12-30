"""Dataset creation with zarr storage for direct tensor loading.

Stores continuous multi-modal data in zarr format that can be loaded
directly to GPU tensors with named dimensions.
"""

from __future__ import annotations

import pickle
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import zarr
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from myoverse.datasets.utils.splitter import DataSplitter


@dataclass
class Modality:
    """Configuration for a data modality.

    Parameters
    ----------
    data : np.ndarray | dict[str, np.ndarray] | None
        Data array or dict of arrays per task.
    path : Path | str | None
        Path to pickle file containing data.
    dims : tuple[str, ...]
        Dimension names (last must be 'time').
    attrs : dict
        Optional attributes to store with the data.
    transform : callable | None
        Transform to apply after loading (receives tensor, returns tensor).

    Examples
    --------
    >>> emg = Modality(
    ...     path="emg.pkl",
    ...     dims=("channel", "time"),
    ... )
    >>> # With preprocessing transform
    >>> from myoverse.transforms import Compose, Flatten, Index
    >>> kinematics = Modality(
    ...     path="kinematics.pkl",
    ...     dims=("dof", "time"),
    ...     transform=Compose([
    ...         Flatten(0, 1),  # (21, 3, time) -> (63, time)
    ...         Index(slice(3, None), dim="channel"),  # Remove wrist -> (60, time)
    ...     ]),
    ... )
    """

    data: np.ndarray | dict[str, np.ndarray] | None = None
    path: Path | str | None = None
    dims: tuple[str, ...] = ("channel", "time")
    attrs: dict = field(default_factory=dict)
    transform: Any = None

    def __post_init__(self):
        if self.path is not None:
            self.path = Path(self.path)
        if self.dims[-1] != "time":
            raise ValueError(f"Last dimension must be 'time', got {self.dims}")
        if self.data is None and (self.path is None or not self.path.exists()):
            raise ValueError("Must provide data or valid path")

    def load(self) -> dict[str, np.ndarray]:
        """Load data from path or return data dict, applying transform if set."""
        if self.data is not None:
            if isinstance(self.data, np.ndarray):
                data = {"default": self.data}
            else:
                data = self.data
        else:
            with open(self.path, "rb") as f:
                data = pickle.load(f)

        # Apply transform (converts to tensor, applies transform, back to numpy)
        if self.transform is not None:
            transformed = {}
            for task, arr in data.items():
                tensor = torch.from_numpy(arr.astype(np.float32))
                result = self.transform(tensor)
                if hasattr(result, 'rename'):
                    result = result.rename(None)
                transformed[task] = result.numpy()
            data = transformed

        return data


class DatasetCreator:
    """Creates datasets stored in zarr for direct tensor loading.

    Data is stored in zarr format with metadata for dimension names,
    enabling direct loading to GPU tensors with named dimensions.

    Parameters
    ----------
    modalities : dict[str, Modality]
        Dictionary mapping modality names to Modality configs.
    sampling_frequency : float
        Sampling frequency in Hz.
    tasks_to_use : Sequence[str]
        Task keys to include (empty = all).
    save_path : Path
        Output Zarr path.
    test_ratio : float
        Ratio for test split (0.0-1.0).
    val_ratio : float
        Ratio for validation split (0.0-1.0).
    time_chunk_size : int
        Chunk size along time dimension for zarr storage. Default 256.
        Set close to your typical window_size for optimal read performance.
    debug_level : int
        Debug output level (0=none, 1=text, 2=text+graphs).

    Examples
    --------
    >>> from myoverse.datasets import DatasetCreator, Modality
    >>>
    >>> creator = DatasetCreator(
    ...     modalities={
    ...         "emg": Modality(path="emg.pkl", dims=("channel", "time")),
    ...         "kinematics": Modality(path="kin.pkl", dims=("joint", "time")),
    ...     },
    ...     sampling_frequency=2048.0,
    ...     save_path=Path("dataset.zarr"),
    ... )
    >>> creator.create()
    >>>
    >>> # Load directly to GPU tensors
    >>> from myoverse.datasets import DataModule
    >>> dm = DataModule("dataset.zarr", device="cuda")
    """

    def __init__(
        self,
        modalities: dict[str, Modality],
        sampling_frequency: float = 2048.0,
        tasks_to_use: Sequence[str] = (),
        save_path: Path | str = Path("dataset.zarr"),
        test_ratio: float = 0.2,
        val_ratio: float = 0.2,
        time_chunk_size: int = 256,
        debug_level: int = 0,
    ):
        self.modalities = modalities
        self.sampling_frequency = sampling_frequency
        self.tasks_to_use = list(tasks_to_use)
        self.save_path = Path(save_path)
        self.time_chunk_size = time_chunk_size
        self.debug_level = debug_level

        self.splitter = DataSplitter(test_ratio=test_ratio, val_ratio=val_ratio)
        self.console = Console()

        # Load all modality data
        self._data: dict[str, dict[str, np.ndarray]] = {}
        for name, modality in self.modalities.items():
            self._data[name] = modality.load()

    def create(self) -> None:
        """Create the dataset."""
        self._print_header()
        self._print_config()

        # Determine tasks from first modality
        if not self.tasks_to_use:
            first_modality = next(iter(self._data.values()))
            self.tasks_to_use = list(first_modality.keys())

        self._print_data_structure()

        # Clear existing dataset
        if self.save_path.exists():
            shutil.rmtree(self.save_path)

        # Create zarr store
        store = zarr.open(str(self.save_path), mode="w")

        # Store metadata
        store.attrs["sampling_frequency"] = self.sampling_frequency
        store.attrs["modalities"] = list(self.modalities.keys())
        store.attrs["tasks"] = self.tasks_to_use

        # Store dimension info for each modality
        dims_info = {name: list(mod.dims) for name, mod in self.modalities.items()}
        store.attrs["dims"] = dims_info

        # Create split groups
        for split in ["training", "validation", "testing"]:
            store.create_group(split)

        # Process tasks
        self._process_all_tasks(store)

        self._print_summary()

    def _print_header(self, title: str = "STARTING DATASET CREATION") -> None:
        if self.debug_level < 1:
            return
        self.console.rule(f"[bold blue]{title}", style="blue")
        self.console.print()

    def _print_config(self) -> None:
        if self.debug_level < 1:
            return

        table = Table(title="Dataset Configuration", show_header=True)
        table.add_column("Parameter", style="dim")
        table.add_column("Value", style="green")

        table.add_row("Modalities", ", ".join(self.modalities.keys()))
        table.add_row("Sampling frequency (Hz)", str(self.sampling_frequency))
        table.add_row("Save path", str(self.save_path))
        table.add_row("Test ratio", str(self.splitter.test_ratio))
        table.add_row("Validation ratio", str(self.splitter.val_ratio))

        self.console.print(table)
        self.console.print()

    def _print_data_structure(self) -> None:
        if self.debug_level < 1:
            return

        self.console.print(
            f"Processing {len(self.tasks_to_use)} tasks: {', '.join(self.tasks_to_use)}"
        )
        self.console.print()

        from rich.tree import Tree

        tree = Tree("Dataset Structure")

        for mod_name, modality in self.modalities.items():
            mod_branch = tree.add(f"[bold]{mod_name}[/bold] dims={modality.dims}")
            for task in self.tasks_to_use:
                if task in self._data[mod_name]:
                    shape = self._data[mod_name][task].shape
                    mod_branch.add(f"Task {task}: {shape}")

        self.console.print(tree)
        self.console.print()

    def _process_all_tasks(self, store: zarr.Group) -> None:
        if self.debug_level < 1:
            for task in self.tasks_to_use:
                self._process_task(task, store)
            return

        self._print_header("PROCESSING TASKS")

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TextColumn("\t"),
            TimeRemainingColumn(),
            console=self.console,
            expand=True,
            transient=False,
        ) as progress:
            task_progress = progress.add_task(
                f"Processing {len(self.tasks_to_use)} tasks...",
                total=len(self.tasks_to_use),
            )

            for task_idx, task in enumerate(self.tasks_to_use):
                progress.update(
                    task_progress,
                    description=f"Processing task {task} ({task_idx + 1}/{len(self.tasks_to_use)})",
                )
                self._process_task(task, store)
                progress.advance(task_progress)

    def _process_task(self, task: str, store: zarr.Group) -> None:
        """Process a single task for all modalities."""
        # Find minimum length across all modalities for this task
        min_len = min(
            self._data[name][task].shape[-1]
            for name in self.modalities
            if task in self._data[name]
        )

        # Process each modality
        for mod_name in self.modalities:
            if task not in self._data[mod_name]:
                continue

            data = self._data[mod_name][task][..., :min_len].astype(np.float32)
            train, test, val = self._split_continuous(data)

            # Store arrays with chunking along time
            array_name = f"{mod_name}_{task}"

            # Training data
            chunks = list(train.shape)
            chunks[-1] = min(self.time_chunk_size, chunks[-1])
            arr = store["training"].create_array(
                array_name,
                shape=train.shape,
                chunks=tuple(chunks),
                dtype=train.dtype,
            )
            arr[:] = train

            # Test data
            if test is not None:
                chunks = list(test.shape)
                chunks[-1] = min(self.time_chunk_size, chunks[-1])
                arr = store["testing"].create_array(
                    array_name,
                    shape=test.shape,
                    chunks=tuple(chunks),
                    dtype=test.dtype,
                )
                arr[:] = test

            # Validation data
            if val is not None:
                chunks = list(val.shape)
                chunks[-1] = min(self.time_chunk_size, chunks[-1])
                arr = store["validation"].create_array(
                    array_name,
                    shape=val.shape,
                    chunks=tuple(chunks),
                    dtype=val.dtype,
                )
                arr[:] = val

    def _split_continuous(
        self, data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Split continuous data along time axis (last dimension)."""
        if self.splitter.test_ratio == 0:
            return data, None, None

        n_samples = data.shape[-1]
        test_amount = int(n_samples * self.splitter.test_ratio / 2)
        middle = n_samples // 2

        test_start = middle - test_amount
        test_end = middle + test_amount
        testing = data[..., test_start:test_end]
        training = np.concatenate(
            [data[..., :test_start], data[..., test_end:]], axis=-1
        )

        if self.splitter.val_ratio > 0:
            val_samples = testing.shape[-1]
            val_amount = int(val_samples * self.splitter.val_ratio / 2)
            val_middle = val_samples // 2
            val_start = val_middle - val_amount
            val_end = val_middle + val_amount

            validation = testing[..., val_start:val_end]
            testing = np.concatenate(
                [testing[..., :val_start], testing[..., val_end:]], axis=-1
            )
        else:
            validation = None

        return training, testing, validation

    def _print_summary(self) -> None:
        """Print dataset summary."""
        if self.debug_level < 1:
            return

        self._print_header("DATASET CREATION COMPLETED")

        total_size = sum(
            f.stat().st_size for f in self.save_path.rglob("*") if f.is_file()
        )

        table = Table(title="Dataset Summary", box=None)
        table.add_column("Split", style="cyan")

        for mod_name in self.modalities:
            table.add_column(mod_name, style="green")
        table.add_column("Size", style="yellow")

        store = zarr.open(str(self.save_path), mode="r")

        for split in ["training", "validation", "testing"]:
            if split not in store:
                continue

            split_group = store[split]
            if len(split_group) == 0:
                continue

            split_path = self.save_path / split
            split_size = sum(
                f.stat().st_size for f in split_path.rglob("*") if f.is_file()
            )

            row = [split]

            for mod_name in self.modalities:
                arrays = [k for k in split_group.keys() if k.startswith(f"{mod_name}_")]
                shapes = []
                for arr_name in sorted(arrays):
                    task = arr_name.split("_", 1)[1]
                    shape = split_group[arr_name].shape
                    shapes.append(f"{task}: {shape}")
                row.append("\n".join(shapes) if shapes else "N/A")

            row.append(f"{split_size / 1024 / 1024:.2f} MB")
            table.add_row(*row)

        self.console.print(table)
        self.console.print(
            f"\n[bold green]Total size: {total_size / 1024 / 1024:.2f} MB[/bold green]"
        )
        self.console.rule(
            "[bold green]Dataset Creation Successfully Completed!", style="green"
        )


# Backwards compatibility alias
EMGDatasetCreator = DatasetCreator
