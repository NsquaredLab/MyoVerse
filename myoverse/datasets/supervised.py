import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import zarr
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    SpinnerColumn,
)
from rich.table import Table
from rich.tree import Tree

from myoverse.datasets.filters.emg_augmentations import EMGAugmentation
from myoverse.datasets.filters.generic import ChunkizeDataFilter, FilterBaseClass
from myoverse.datatypes import _Data, DATA_TYPES_MAP


def _split_data(data: np.ndarray, split_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    split_amount = int(data.shape[0] * split_ratio / 2)
    middle_index = data.shape[0] // 2

    mask = np.ones(data.shape[0], dtype=bool)
    mask[middle_index - split_amount : middle_index + split_amount] = False

    return data[mask], data[~mask]


def _add_to_dataset(group: zarr.Group, data: Optional[np.ndarray], name: str):
    """Add data to a zarr group, handling compatibility with Zarr 3.

    Parameters
    ----------
    group : zarr.Group
        The zarr group to add data to
    data : Optional[np.ndarray]
        The data to add
    name : str
        The name of the dataset
    """
    if data is None or (isinstance(data, np.ndarray) and data.size == 0):
        return

    # Ensure data is a numpy array
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    try:
        if name in group:
            # Don't append empty data
            if data.size == 0:
                return

            # Zarr 3 doesn't have append but we can use setitem to add data
            current_shape = group[name].shape
            new_shape = list(current_shape)
            new_shape[0] += data.shape[0]

            # Resize the dataset
            group[name].resize(new_shape)

            # Insert the new data
            group[name][current_shape[0] :] = data
        else:
            # Don't create empty datasets
            if data.size == 0:
                return

            # Create new dataset with appropriate chunking
            group.create_dataset(
                name, data=data, shape=data.shape, chunks=(1, *data.shape[1:])
            )
    except Exception as e:
        # Handle differences between Zarr 2 and 3
        if "append" in str(e):
            # This is Zarr 2 behavior
            if data.size > 0:  # Only append if there's data
                group[name].append(data)
        else:
            raise


class EMGDataset:
    """
    Class for creating a dataset from EMG and ground truth data.

    Parameters
    ----------
    emg_data_path : pathlib.Path
        Path to the EMG data file. It should be a pickle file containing a dictionary with the keys being the task
        number and the values being a numpy array of shape (n_channels, n_samples).
    emg_data : dict[str, np.ndarray]
        Optional dictionary containing EMG data if not loading from a file
    ground_truth_data_path : pathlib.Path
        Path to the ground truth data file. It should be a pickle file containing a dictionary with the keys being the
        task number and the values being a numpy array of custom shape (..., n_samples). The custom shape can be
        anything, but the last dimension should be the same as the EMG data.
    ground_truth_data : dict[str, np.ndarray]
        Optional dictionary containing ground truth data if not loading from a file
    ground_truth_data_type : str
        Type of ground truth data, e.g. 'kinematics'
    sampling_frequency : float
        Sampling frequency of the data in Hz
    tasks_to_use : Sequence[str]
        Sequence of strings containing the task numbers to use. If empty, all tasks will be used.
    save_path : pathlib.Path
        Path to save the dataset to. It should be a zarr file.
    emg_filter_pipeline_before_chunking : list[list[FilterBaseClass]]
        Sequence of filters to apply to the EMG data before chunking.
    emg_representations_to_filter_before_chunking : list[list[str]]
        Representations of EMG data to filter before chunking.
    emg_filter_pipeline_after_chunking : list[list[FilterBaseClass]]
        Sequence of filters to apply to the EMG data after chunking.
    emg_representations_to_filter_after_chunking : list[list[str]]
        Representations of EMG data to filter after chunking.
    ground_truth_filter_pipeline_before_chunking : list[list[FilterBaseClass]]
        Sequence of filters to apply to the ground truth data before chunking.
    ground_truth_representations_to_filter_before_chunking : list[list[str]]
        Representations of ground truth data to filter before chunking.
    ground_truth_filter_pipeline_after_chunking : list[list[FilterBaseClass]]
        Sequence of filters to apply to the ground truth data after chunking.
    ground_truth_representations_to_filter_after_chunking : list[list[str]]
        Representations of ground truth data to filter after chunking.
    chunk_size : int
        Size of the chunks to create from the data.
    chunk_shift : int
        Shift between the chunks.
    testing_split_ratio : float
        Ratio of the data to use for testing. The data will be split in the middle. The first half will be used for
        training and the second half will be used for testing. If 0, no data will be used for testing.
    validation_split_ratio : float
        Ratio of the data to use for validation. The data will be split in the middle. The first half will be used for
        training and the second half will be used for validation. If 0, no data will be used for validation.
    augmentation_pipelines : list[list[EMGAugmentation]]
        Sequence of augmentation_pipelines to apply to the training data.
    amount_of_chunks_to_augment_at_once : int
        Amount of chunks to augment at once. This is done to speed up the process.
    debug_level : int
        Debug level:
        - 0: No debug output (default)
        - 1: Full text debugging with Rich (configuration, progress, tables, data details)
        - 2: Level 1 plus data visualizations (graphs and plots)
    silence_zarr_warnings : bool
        Whether to silence all Zarr-related warnings, including those from zarr.codecs and zarr.core modules

    Methods
    -------
    create_dataset()
        Creates the dataset.
    """

    def __init__(
        self,
        emg_data_path: Path = Path("REPLACE ME"),
        emg_data: dict[str, np.ndarray] = {},
        ground_truth_data_path: Path = Path("REPLACE ME"),
        ground_truth_data: dict[str, np.ndarray] = {},
        ground_truth_data_type: str = "kinematics",
        sampling_frequency: float = 0.0,
        tasks_to_use: Sequence[str] = (),
        save_path: Path = Path("REPLACE ME"),
        emg_filter_pipeline_before_chunking: list[list[FilterBaseClass]] = (),
        emg_representations_to_filter_before_chunking: list[list[str]] = (),
        emg_filter_pipeline_after_chunking: list[list[FilterBaseClass]] = (),
        emg_representations_to_filter_after_chunking: list[list[str]] = (),
        ground_truth_filter_pipeline_before_chunking: list[list[FilterBaseClass]] = (),
        ground_truth_representations_to_filter_before_chunking: list[list[str]] = (),
        ground_truth_filter_pipeline_after_chunking: list[list[FilterBaseClass]] = (),
        ground_truth_representations_to_filter_after_chunking: list[list[str]] = (),
        chunk_size: int = 192,
        chunk_shift: int = 64,
        testing_split_ratio: float = 0.2,
        validation_split_ratio: float = 0.2,
        augmentation_pipelines: list[list[EMGAugmentation]] = (),
        amount_of_chunks_to_augment_at_once: int = 250,
        debug_level: int = 0,
        silence_zarr_warnings: bool = True,
    ):
        self.emg_data_path = emg_data_path
        self.emg_data = emg_data
        self.ground_truth_data_path = ground_truth_data_path
        self.ground_truth_data = ground_truth_data

        # check if at least one of the data sources is provided
        if not self.emg_data and not self.emg_data_path:
            raise ValueError("At least one of the EMG data sources should be provided.")
        if not self.ground_truth_data and not self.ground_truth_data_path:
            raise ValueError(
                "At least one of the ground truth data sources should be provided."
            )

        self.ground_truth_data_type = ground_truth_data_type
        self.sampling_frequency = sampling_frequency
        self.tasks_to_use = tasks_to_use
        self.save_path = save_path

        self.emg_filter_pipeline_before_chunking = emg_filter_pipeline_before_chunking
        self.emg_representations_to_filter_before_chunking = (
            emg_representations_to_filter_before_chunking
        )
        self.ground_truth_filter_pipeline_before_chunking = (
            ground_truth_filter_pipeline_before_chunking
        )
        self.ground_truth_representations_to_filter_before_chunking = (
            ground_truth_representations_to_filter_before_chunking
        )

        self.emg_filter_pipeline_after_chunking = emg_filter_pipeline_after_chunking
        self.emg_representations_to_filter_after_chunking = (
            emg_representations_to_filter_after_chunking
        )
        self.ground_truth_filter_pipeline_after_chunking = (
            ground_truth_filter_pipeline_after_chunking
        )
        self.ground_truth_representations_to_filter_after_chunking = (
            ground_truth_representations_to_filter_after_chunking
        )

        self.chunk_size = chunk_size
        self.chunk_shift = chunk_shift

        self.testing_split_ratio = testing_split_ratio
        self.validation_split_ratio = validation_split_ratio

        self.augmentation_pipelines = augmentation_pipelines
        self.amount_of_chunks_to_augment_at_once = amount_of_chunks_to_augment_at_once
        self.debug_level = debug_level
        self.silence_zarr_warnings = silence_zarr_warnings

        # Initialize Rich console for all debug levels
        self.console = Console()

        self._tasks_string_length = 0

    def __add_data_to_dataset(
        self, data: _Data, groups: list[zarr.Group]
    ) -> Tuple[list[int], list[int], list[int]]:
        """
        Add data to zarr dataset groups.

        Parameters
        ----------
        data : _Data
            The data object to add to the dataset
        groups : list[zarr.Group]
            List of zarr groups for training, testing, and validation

        Returns
        -------
        tuple[list[int], list[int], list[int]]
            Lists of sizes for training, testing, and validation datasets
        """
        training_data_sizes, testing_data_sizes, validation_data_sizes = [], [], []

        if self.debug_level >= 1:
            self.console.print(
                f"[bold green]Adding data with keys:[/bold green] [cyan]{list(data.output_representations.keys())}[/cyan]"
            )
            self.console.print()  # Add empty line

        for k, v in data.output_representations.items():
            validation_data_from_task = None

            if self.debug_level >= 1:
                self.console.print(
                    f"[bold]Splitting representation:[/bold] [yellow]{k}[/yellow] [dim]with shape {v.shape}[/dim]"
                )

            if self.testing_split_ratio > 0:
                training_data_from_task, testing_data_from_task = _split_data(
                    v, self.testing_split_ratio
                )

                if self.debug_level >= 1:
                    self.console.print(
                        f"  [green]Training shape:[/green] {training_data_from_task.shape}"
                    )
                    self.console.print(
                        f"  [yellow]Testing shape:[/yellow] {testing_data_from_task.shape}"
                    )

                if self.validation_split_ratio > 0:
                    testing_data_from_task, validation_data_from_task = _split_data(
                        testing_data_from_task, self.validation_split_ratio
                    )

                    if self.debug_level >= 1:
                        self.console.print(
                            f"  [yellow]After validation split - Testing shape:[/yellow] {testing_data_from_task.shape}"
                        )
                        self.console.print(
                            f"  [blue]Validation shape:[/blue] {validation_data_from_task.shape}"
                        )

            else:
                training_data_from_task = v
                testing_data_from_task = None

                if self.debug_level >= 1:
                    self.console.print(
                        f"  [green]No testing split, all data for training:[/green] {training_data_from_task.shape}"
                    )

            # Add a space between different splits
            if self.debug_level >= 1:
                self.console.print()  # Add empty line

            for g, data_from_task in zip(
                groups,
                (
                    training_data_from_task,
                    testing_data_from_task,
                    validation_data_from_task,
                ),
            ):
                _add_to_dataset(g, data_from_task, k)

            training_data_sizes.append(training_data_from_task.shape[0])
            testing_data_sizes.append(
                testing_data_from_task.shape[0]
                if testing_data_from_task is not None
                else 0
            )
            validation_data_sizes.append(
                validation_data_from_task.shape[0]
                if validation_data_from_task is not None
                else 0
            )

        if self.debug_level >= 1:
            # Create a table for the dataset sizes
            sizes_table = Table(
                title="Dataset Split Sizes",
                show_header=True,
                header_style="bold magenta",
                box=box.ROUNDED,
                padding=(0, 2),
                width=40,
            )
            sizes_table.add_column("Split", style="cyan")
            sizes_table.add_column("Sizes", style="green")

            sizes_table.add_row("Training", str(training_data_sizes))
            sizes_table.add_row("Testing", str(testing_data_sizes))
            sizes_table.add_row("Validation", str(validation_data_sizes))

            self.console.print(sizes_table)
            self.console.print()  # Add empty line

        return training_data_sizes, testing_data_sizes, validation_data_sizes

    def create_dataset(self):
        """Create a supervised dataset from EMG and ground truth data."""
        # Silence zarr warnings if requested
        if self.silence_zarr_warnings:
            import warnings

            # Silence warnings from zarr.codecs
            warnings.filterwarnings(
                "ignore", category=UserWarning, module="zarr.codecs"
            )
            # Silence warnings from zarr core
            warnings.filterwarnings("ignore", category=UserWarning, module="zarr.core")
            # Silence any other zarr-related warnings
            warnings.filterwarnings("ignore", category=UserWarning, module="zarr")

        # Display configuration when debugging is enabled
        if self.debug_level > 0:
            # Create header for dataset creation
            self.console.rule(
                "[bold blue]STARTING DATASET CREATION", style="blue double"
            )
            self.console.print()  # Add empty line

            # Create a table for configuration
            table = Table(
                title="Dataset Configuration",
                show_header=True,
                header_style="bold magenta",
                box=box.ROUNDED,
                padding=(0, 2),
            )
            table.add_column("Parameter", style="dim", width=30)
            table.add_column("Value", style="green")

            # Add configuration parameters to table
            table.add_row("EMG data path", str(self.emg_data_path))
            table.add_row("Ground truth data path", str(self.ground_truth_data_path))
            table.add_row("Ground truth data type", self.ground_truth_data_type)
            table.add_row("Sampling frequency (Hz)", str(self.sampling_frequency))
            table.add_row("Save path", str(self.save_path))
            table.add_row("Chunk size", str(self.chunk_size))
            table.add_row("Chunk shift", str(self.chunk_shift))
            table.add_row("Testing split ratio", str(self.testing_split_ratio))
            table.add_row("Validation split ratio", str(self.validation_split_ratio))
            table.add_row(
                "Amount of chunks to augment at once",
                str(self.amount_of_chunks_to_augment_at_once),
            )
            table.add_row("Debug level", str(self.debug_level))
            table.add_row("Silence Zarr warnings", str(self.silence_zarr_warnings))

            # Display the table
            self.console.print(table)
            self.console.print()  # Add empty line

        # Load data if not provided
        emg_data = self.emg_data or pickle.load(self.emg_data_path.open("rb"))
        ground_truth_data = self.ground_truth_data or pickle.load(
            self.ground_truth_data_path.open("rb")
        )

        # Use tasks_to_use if provided, otherwise use all tasks
        if not self.tasks_to_use:
            self.tasks_to_use = list(emg_data.keys())

        if self.debug_level > 0:
            self.console.print(
                f"[bold cyan]Processing {len(self.tasks_to_use)} tasks:[/bold cyan] {', '.join(self.tasks_to_use)}"
            )
            self.console.print()  # Add empty line

            # Create a tree for data shapes
            data_tree = Tree("[bold yellow]Dataset Structure")
            emg_branch = data_tree.add("[bold green]EMG Data")
            for i, (k, v) in enumerate(list(emg_data.items())[:5]):
                emg_branch.add(f"Task {k}: Shape {v.shape}")
            if len(emg_data) > 5:
                emg_branch.add(f"... {len(emg_data) - 5} more tasks")

            gt_branch = data_tree.add("[bold green]Ground Truth Data")
            for i, (k, v) in enumerate(list(ground_truth_data.items())[:5]):
                gt_branch.add(f"Task {k}: Shape {v.shape}")
            if len(ground_truth_data) > 5:
                gt_branch.add(f"... {len(ground_truth_data) - 5} more tasks")

            self.console.print(data_tree)
            self.console.print()  # Add empty line

        # Create zarr directory and open store
        self.save_path.mkdir(parents=True, exist_ok=True)

        # Open zarr dataset with specified format
        dataset = zarr.open(str(self.save_path), mode="w", zarr_version=2)

        # Create groups for training, testing, and validation
        training_group = dataset.create_group("training")
        testing_group = dataset.create_group("testing")
        validation_group = dataset.create_group("validation")

        # Set the task string length for labels
        self._tasks_string_length = len(max(self.tasks_to_use, key=len))

        # Process each task
        if self.debug_level > 0:
            self.console.rule("[bold blue]PROCESSING TASKS", style="blue double")
            self.console.print()  # Add empty line

        # Create progress bar for task processing regardless of debug level
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self.console,
            expand=True,
            transient=False,
        ) as progress:
            task_progress = progress.add_task(
                f"[bold]Processing [cyan]{len(self.tasks_to_use)}[/cyan] tasks...",
                total=len(self.tasks_to_use),
            )

            # Process each task with progress tracking
            for task_idx, task in enumerate(self.tasks_to_use):
                # Update the progress bar with current task information
                progress.update(
                    task_progress,
                    description=f"[bold]Processing task [cyan]{task}[/cyan] ([cyan]{task_idx + 1}[/cyan]/[cyan]{len(self.tasks_to_use)}[/cyan])",
                )

                # Process the task
                self._process_task(
                    task,
                    emg_data,
                    ground_truth_data,
                    training_group,
                    testing_group,
                    validation_group,
                )

                # Advance the progress bar
                progress.advance(task_progress)

        # Apply data augmentation if requested
        self._apply_augmentations(dataset, training_group)

        # Print dataset summary at the end
        if self.debug_level > 0:
            self._print_dataset_summary(dataset)

    def _process_task(
        self,
        task: str,
        emg_data: dict[str, np.ndarray],
        ground_truth_data: dict[str, np.ndarray],
        training_group: zarr.Group,
        testing_group: zarr.Group,
        validation_group: zarr.Group,
    ):
        """Process a single task and add its data to the dataset."""
        emg_data_from_task = emg_data[task]
        ground_truth_data_from_task = ground_truth_data[task]

        # Trim data to same length if needed
        min_length = min(
            emg_data_from_task.shape[-1], ground_truth_data_from_task.shape[-1]
        )
        emg_data_from_task = emg_data_from_task[..., :min_length]
        ground_truth_data_from_task = ground_truth_data_from_task[..., :min_length]

        # Create appropriate data objects
        emg_data_from_task = DATA_TYPES_MAP["emg"](
            input_data=emg_data_from_task,
            sampling_frequency=self.sampling_frequency,
        )
        ground_truth_data_from_task = DATA_TYPES_MAP[self.ground_truth_data_type](
            input_data=ground_truth_data_from_task,
            sampling_frequency=self.sampling_frequency,
        )

        # Verify chunking status is the same
        if (
            emg_data_from_task.is_chunked["Input"]
            != ground_truth_data_from_task.is_chunked["Input"]
        ):
            raise ValueError(
                f"The EMG and ground truth data should have the same chunking status. "
                f"EMG data is {'chunked' if emg_data_from_task.is_chunked else 'not chunked'} and "
                f"ground truth data is {'chunked' if ground_truth_data_from_task.is_chunked else 'not chunked'}."
            )

        # Debug output based on debug level
        if self.debug_level >= 1:
            self.console.print(
                "[bold white on blue] Initial Data [/bold white on blue]",
                justify="center",
            )
            self.console.print()  # Add empty line

            emg_panel = Panel.fit(
                str(emg_data_from_task),
                title=f"[bold green]EMG Data Task {task}[/bold green]",
                border_style="green",
                box=box.ROUNDED,
                padding=(0, 2),
            )
            gt_panel = Panel.fit(
                str(ground_truth_data_from_task),
                title=f"[bold blue]Ground Truth Data Task {task}[/bold blue]",
                border_style="blue",
                box=box.ROUNDED,
                padding=(0, 2),
            )
            self.console.print(emg_panel)
            self.console.print(gt_panel)
            self.console.print()  # Add empty line

        # Plot graphs only for debug level 2
        if self.debug_level >= 2:
            self.console.print(
                "[bold yellow]Generating EMG data graph...[/bold yellow]"
            )
            emg_data_from_task.plot_graph(
                title=f"EMG Data - Task: {task} (Pre-Processing)"
            )
            self.console.print(
                "[bold yellow]Generating ground truth data graph...[/bold yellow]"
            )
            ground_truth_data_from_task.plot_graph(
                title=f"Ground Truth Data - Task: {task} (Pre-Processing)"
            )
            self.console.print()  # Add empty line

        # Process unchunked data
        if not emg_data_from_task.is_chunked["Input"]:
            if self.debug_level >= 1:
                self.console.print(
                    "[bold white on magenta] Pre-Chunking Processing [/bold white on magenta]",
                    justify="center",
                )
                self.console.print()  # Add empty line

            # Apply filters before chunking
            if self.emg_filter_pipeline_before_chunking:
                if self.debug_level >= 1:
                    self.console.print(
                        "▶ [bold cyan]Applying EMG filters before chunking...[/bold cyan]"
                    )
                emg_data_from_task.apply_filter_pipeline(
                    filter_pipeline=self.emg_filter_pipeline_before_chunking,
                    representations_to_filter=self.emg_representations_to_filter_before_chunking,
                )

            if self.ground_truth_filter_pipeline_before_chunking:
                if self.debug_level >= 1:
                    self.console.print(
                        "▶ [bold cyan]Applying ground truth filters before chunking...[/bold cyan]"
                    )
                ground_truth_data_from_task.apply_filter_pipeline(
                    filter_pipeline=self.ground_truth_filter_pipeline_before_chunking,
                    representations_to_filter=self.ground_truth_representations_to_filter_before_chunking,
                )

            if self.debug_level >= 1:
                self.console.print()  # Add empty line
                self.console.print(
                    "[bold white on green] Chunking Process [/bold white on green]",
                    justify="center",
                )
                self.console.print()  # Add empty line

            # Apply chunking filters
            if self.debug_level >= 1:
                self.console.print("▶ [bold cyan]Chunking EMG data...[/bold cyan]")
            emg_data_from_task.apply_filter(
                filter=ChunkizeDataFilter(
                    chunk_size=self.chunk_size,
                    chunk_shift=self.chunk_shift,
                    is_output=len(self.emg_filter_pipeline_after_chunking) == 0,
                    name="EMG_Chunkizer",
                    input_is_chunked=False,
                ),
                representations_to_filter=["Last"],
            )
            chunked_emg_data_from_task = emg_data_from_task

            if self.debug_level >= 1:
                self.console.print(
                    "▶ [bold cyan]Chunking ground truth data...[/bold cyan]"
                )
            ground_truth_data_from_task.apply_filter(
                filter=ChunkizeDataFilter(
                    chunk_size=self.chunk_size,
                    chunk_shift=self.chunk_shift,
                    is_output=len(self.ground_truth_filter_pipeline_after_chunking)
                    == 0,
                    input_is_chunked=False,
                ),
                representations_to_filter=["Last"],
            )
            chunked_ground_truth_data_from_task = ground_truth_data_from_task

            # Debug output for chunking
            if self.debug_level >= 1:
                self.console.print()  # Add empty line
                self.console.rule("[bold green]After Chunking", style="green")
                self.console.print()  # Add empty line

                emg_panel = Panel.fit(
                    str(chunked_emg_data_from_task),
                    title="[bold green]Chunked EMG Data[/bold green]",
                    border_style="green",
                    box=box.ROUNDED,
                    padding=(0, 2),
                )
                gt_panel = Panel.fit(
                    str(chunked_ground_truth_data_from_task),
                    title="[bold blue]Chunked Ground Truth Data[/bold blue]",
                    border_style="blue",
                    box=box.ROUNDED,
                    padding=(0, 2),
                )
                self.console.print(emg_panel)
                self.console.print(gt_panel)
                self.console.print()  # Add empty line

            # Plot graphs only for debug level 2
            if self.debug_level >= 2:
                self.console.print(
                    "[bold yellow]Generating chunked EMG data graph...[/bold yellow]"
                )
                chunked_emg_data_from_task.plot_graph(
                    title=f"Chunked EMG Data - Task: {task}"
                )
                self.console.print(
                    "[bold yellow]Generating chunked ground truth data graph...[/bold yellow]"
                )
                chunked_ground_truth_data_from_task.plot_graph(
                    title=f"Chunked Ground Truth Data - Task: {task}"
                )
                self.console.print()  # Add empty line
        else:
            # Data is already chunked
            chunked_emg_data_from_task = emg_data_from_task
            # Process in batches to avoid memory issues
            i = 0
            temp = []
            while (
                i + self.amount_of_chunks_to_augment_at_once
                <= chunked_emg_data_from_task.shape[0]
            ):
                temp.append(
                    np.concatenate(
                        chunked_emg_data_from_task[
                            i : i + self.amount_of_chunks_to_augment_at_once
                        ],
                        axis=-1,
                    )
                )
                i += self.amount_of_chunks_to_augment_at_once
            chunked_emg_data_from_task = np.stack(temp, axis=1)
            chunked_ground_truth_data_from_task = ground_truth_data_from_task

        # Post-chunking processing section
        if self.debug_level >= 1:
            self.console.print(
                "[bold white on magenta] Post-Chunking Processing [/bold white on magenta]",
                justify="center",
            )
            self.console.print()  # Add empty line

        # Apply filters after chunking
        if self.emg_filter_pipeline_after_chunking:
            if self.debug_level >= 1:
                self.console.print(
                    "▶ [bold cyan]Applying EMG filters after chunking...[/bold cyan]"
                )
            chunked_emg_data_from_task.apply_filter_pipeline(
                filter_pipeline=self.emg_filter_pipeline_after_chunking,
                representations_to_filter=self.emg_representations_to_filter_after_chunking,
            )

        if self.ground_truth_filter_pipeline_after_chunking:
            if self.debug_level >= 1:
                self.console.print(
                    "▶ [bold cyan]Applying ground truth filters after chunking...[/bold cyan]"
                )
            chunked_ground_truth_data_from_task.apply_filter_pipeline(
                filter_pipeline=self.ground_truth_filter_pipeline_after_chunking,
                representations_to_filter=self.ground_truth_representations_to_filter_after_chunking,
            )

        # Debug output after filtering chunked data
        if self.debug_level >= 1:
            self.console.print()  # Add empty line
            self.console.rule("[bold green]After Filtering Chunked Data", style="green")
            self.console.print()  # Add empty line

            emg_panel = Panel.fit(
                str(chunked_emg_data_from_task),
                title="[bold green]Filtered Chunked EMG Data[/bold green]",
                border_style="green",
                box=box.ROUNDED,
                padding=(0, 2),
            )
            gt_panel = Panel.fit(
                str(chunked_ground_truth_data_from_task),
                title="[bold blue]Filtered Chunked Ground Truth Data[/bold blue]",
                border_style="blue",
                box=box.ROUNDED,
                padding=(0, 2),
            )
            self.console.print(emg_panel)
            self.console.print(gt_panel)
            self.console.print()  # Add empty line

        # Plot graphs only for debug level 2
        if self.debug_level >= 2:
            self.console.print(
                "[bold yellow]Generating filtered chunked EMG data graph...[/bold yellow]"
            )
            chunked_emg_data_from_task.plot_graph(
                title=f"Filtered Chunked EMG Data - Task: {task}"
            )
            self.console.print(
                "[bold yellow]Generating filtered chunked ground truth data graph...[/bold yellow]"
            )
            chunked_ground_truth_data_from_task.plot_graph(
                title=f"Filtered Chunked Ground Truth Data - Task: {task}"
            )
            self.console.print()  # Add empty line

        # Dataset creation section
        if self.debug_level >= 1:
            self.console.print(
                "[bold white on blue] Dataset Creation [/bold white on blue]",
                justify="center",
            )
            self.console.print(
                "▶ [bold cyan]Adding processed data to dataset...[/bold cyan]"
            )
            self.console.print()  # Add empty line

        for group_name, chunked_data_from_task in zip(
            ["emg", "ground_truth"],
            [chunked_emg_data_from_task, chunked_ground_truth_data_from_task],
        ):
            (
                training_sizes,
                testing_sizes,
                validation_sizes,
            ) = self.__add_data_to_dataset(
                chunked_data_from_task,
                [
                    (
                        g.create_group(group_name)
                        if group_name not in list(g.group_keys())
                        else g[group_name]
                    )
                    for g in (training_group, testing_group, validation_group)
                ],
            )

        # Verify data lengths match
        data_length = list(chunked_emg_data_from_task.output_representations.values())[
            -1
        ].shape[0]

        data_length_ground_truth = list(
            chunked_ground_truth_data_from_task.output_representations.values()
        )[-1].shape[0]

        # Validate sizes
        assert len(set(training_sizes)) == 1, "The training sizes are not the same."
        assert len(set(testing_sizes)) == 1, "The testing sizes are not the same."
        assert len(set(validation_sizes)) == 1, "The validation sizes are not the same."

        assert data_length == data_length_ground_truth, (
            f"The data lengths of the EMG and ground truth data should be the same. "
            f"For task {task}, the EMG data has length {data_length} and the ground "
            f"truth data has length {data_length_ground_truth}."
        )

        # Add labels, class indices, and one-hot encodings
        for g, size in zip(
            (training_group, testing_group, validation_group),
            (training_sizes[0], testing_sizes[0], validation_sizes[0]),
        ):
            # Use consistent unicode string array approach since conversion
            # happens in _add_to_dataset as needed
            label_array = np.array(
                [task] * size, dtype=f"<U{self._tasks_string_length}"
            ).reshape(-1, 1)

            _add_to_dataset(
                g,
                label_array,
                "label",
            )
            _add_to_dataset(
                g,
                np.array([self.tasks_to_use.index(task)] * size, dtype=np.int8).reshape(
                    -1, 1
                ),
                "class",
            )
            _add_to_dataset(
                g,
                np.repeat(
                    np.array(
                        [
                            np.eye(len(self.tasks_to_use), dtype=np.int8)[
                                self.tasks_to_use.index(task)
                            ]
                        ]
                    ),
                    size,
                    axis=0,
                ),
                "one_hot_class",
            )

    def _apply_augmentations(self, dataset: zarr.Group, training_group: zarr.Group):
        """Apply augmentations to the training data."""
        # Start augmentation phase if there are augmentation pipelines
        if self.augmentation_pipelines and len(self.augmentation_pipelines) > 0:
            # Get all available samples in training group
            # Use the first available filter key instead of hardcoding "raw"
            filter_keys = list(training_group["emg"].array_keys())
            if not filter_keys:
                self.console.print(
                    "[bold red]No EMG filters found in training group![/bold red]"
                )
                return

            training_size = training_group["emg"][filter_keys[0]].shape[0]

            if self.debug_level > 0:
                self.console.rule(
                    "[bold blue]APPLYING AUGMENTATIONS", style="blue double"
                )
                self.console.print()  # Add empty line

                # Display augmentation info
                augmentation_info = Table(
                    title="Augmentation Configuration",
                    show_header=True,
                    header_style="bold magenta",
                    box=box.ROUNDED,
                    padding=(0, 2),
                )
                augmentation_info.add_column("Parameter", style="dim", width=30)
                augmentation_info.add_column("Value", style="green")

                augmentation_info.add_row(
                    "Total augmentation pipelines",
                    str(len(self.augmentation_pipelines)),
                )
                pipeline_names = []
                for pipeline in self.augmentation_pipelines:
                    names = [f.name for f in pipeline]
                    pipeline_names.append(" → ".join(names))
                augmentation_info.add_row("Pipelines", "\n".join(pipeline_names))
                augmentation_info.add_row(
                    "Chunks to augment at once",
                    str(self.amount_of_chunks_to_augment_at_once),
                )
                augmentation_info.add_row("Total training samples", str(training_size))

                self.console.print(augmentation_info)
                self.console.print()  # Add empty line

            # Create progress bar for augmentation regardless of debug level
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold green]{task.description}"),
                BarColumn(bar_width=40),
                TaskProgressColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
                console=self.console,
                expand=True,
                transient=False,
            ) as progress:
                # Calculate total batches for all augmentations
                total_batches = int(
                    np.ceil(training_size / self.amount_of_chunks_to_augment_at_once)
                ) * len(self.augmentation_pipelines)

                # Create main progress task
                aug_progress = progress.add_task(
                    f"[bold]Applying [green]{len(self.augmentation_pipelines)}[/green] augmentation pipelines...",
                    total=total_batches,
                )

                # Process each augmentation pipeline
                for aug_idx, augmentation_pipeline in enumerate(
                    self.augmentation_pipelines
                ):
                    pipeline_name = " → ".join([f.name for f in augmentation_pipeline])

                    # Update progress description for this pipeline
                    progress.update(
                        aug_progress,
                        description=f"[bold]Pipeline [green]{aug_idx + 1}/{len(self.augmentation_pipelines)}[/green]: {pipeline_name}",
                    )

                    # Apply augmentation in batches
                    self._apply_augmentation_pipeline(
                        aug_idx,
                        augmentation_pipeline,
                        dataset,
                        training_group,
                        progress,
                        aug_progress,
                    )

    def _apply_augmentation_pipeline(
        self,
        aug_idx: int,
        augmentation_pipeline: list[EMGAugmentation],
        dataset: zarr.Group,
        training_group: zarr.Group,
        progress: Optional[Progress] = None,
        progress_task_id: Optional[int] = None,
    ):
        """Apply a single augmentation pipeline to training data in batches."""
        # Get total samples to process
        # Use the first available filter key instead of hardcoding "raw"
        filter_keys = list(training_group["emg"].array_keys())
        if not filter_keys:
            self.console.print(
                "[bold red]No EMG filters found in training group![/bold red]"
            )
            return

        training_size = training_group["emg"][filter_keys[0]].shape[0]

        # Process in batches
        for start_idx in range(
            0, training_size, self.amount_of_chunks_to_augment_at_once
        ):
            # Calculate end index for current batch
            end_idx = min(
                start_idx + self.amount_of_chunks_to_augment_at_once, training_size
            )

            # Update progress information
            if progress is not None and progress_task_id is not None:
                batch_num = start_idx // self.amount_of_chunks_to_augment_at_once + 1
                total_batches = int(
                    np.ceil(training_size / self.amount_of_chunks_to_augment_at_once)
                )
                progress.update(
                    progress_task_id,
                    description=f"[bold]Pipeline [green]{aug_idx + 1}/{len(self.augmentation_pipelines)}[/green]: Batch [green]{batch_num}/{total_batches}[/green]",
                )

            # Initialize batch accumulators
            emg_to_append = {k: [] for k in dataset["training/emg"]}
            ground_truth_to_append = {k: [] for k in dataset["training/ground_truth"]}
            label_to_append = []
            class_to_append = []
            one_hot_class_to_append = []

            # Process each item in the batch
            for i in range(start_idx, end_idx):
                # Apply augmentation to EMG data
                for k in dataset["training/emg"]:
                    temp = DATA_TYPES_MAP["emg"](
                        input_data=dataset["training/emg"][k][i].astype(np.float32),
                        sampling_frequency=self.sampling_frequency,
                    )
                    temp.apply_filter_pipeline(
                        filter_pipeline=[augmentation_pipeline],
                        representations_to_filter=[["Last"]],
                    )
                    emg_to_append[k].append(temp["Last"])

                # Copy corresponding ground truth data
                for k in dataset["training/ground_truth"]:
                    ground_truth_to_append[k].append(
                        dataset["training/ground_truth"][k][i]
                    )

                # Copy labels and classes
                label_to_append.append(dataset["training/label"][i])
                class_to_append.append(dataset["training/class"][i])
                one_hot_class_to_append.append(dataset["training/one_hot_class"][i])

            # Append the batch to the training group
            self._append_augmented_batch(
                training_group,
                emg_to_append,
                ground_truth_to_append,
                label_to_append,
                class_to_append,
                one_hot_class_to_append,
            )

            # Advance progress if tracking
            if progress is not None and progress_task_id is not None:
                progress.advance(progress_task_id)

    def _append_augmented_batch(
        self,
        training_group: zarr.Group,
        emg_to_append: Dict[str, List[np.ndarray]],
        ground_truth_to_append: Dict[str, List[np.ndarray]],
        label_to_append: List[np.ndarray],
        class_to_append: List[np.ndarray],
        one_hot_class_to_append: List[np.ndarray],
    ):
        """Append a batch of augmented data to the training group."""
        # Debug shapes before appending
        if self.debug_level >= 2:  # Only show shapes in higher debug level
            # Create a table for shapes
            shapes_table = Table(
                title="Augmented Batch Shapes",
                show_header=True,
                header_style="bold magenta",
                box=box.ROUNDED,
                padding=(0, 2),
                width=80,
                title_style="bold magenta",
                title_justify="center",
            )
            shapes_table.add_column("Data Type", style="cyan", width=50)
            shapes_table.add_column("Shape", style="green")

            # Add EMG shapes
            for k, v in emg_to_append.items():
                if v:
                    shapes_table.add_row(f"EMG {k}", str(np.array(v).shape))
                    break

            # Add ground truth shapes
            for k, v in ground_truth_to_append.items():
                if v:
                    shapes_table.add_row(f"Ground Truth {k}", str(np.array(v).shape))
                    break

            # Add label shape
            if label_to_append:
                shapes_table.add_row("Labels", str(np.array(label_to_append).shape))

            self.console.print(shapes_table)

        # Add EMG data
        for k, v in emg_to_append.items():
            if v:  # Check that list is not empty
                _add_to_dataset(training_group["emg"], np.array(v), name=k)

        # Add ground truth data
        for k, v in ground_truth_to_append.items():
            if v:  # Check that list is not empty
                _add_to_dataset(training_group["ground_truth"], np.array(v), name=k)

        # Add labels and classes
        if label_to_append:
            _add_to_dataset(training_group, np.array(label_to_append), name="label")

        if class_to_append:
            _add_to_dataset(training_group, np.array(class_to_append), name="class")
            _add_to_dataset(
                training_group,
                np.array(one_hot_class_to_append),
                name="one_hot_class",
            )

    def _print_dataset_summary(self, dataset: zarr.Group):
        """Print a summary of the created dataset."""
        # Get dataset sizes
        training_emg_sizes = {
            k: dataset["training/emg"][k].shape for k in dataset["training/emg"]
        }
        testing_emg_sizes = (
            {k: dataset["testing/emg"][k].shape for k in dataset["testing/emg"]}
            if "emg" in dataset["testing"]
            else {}
        )
        validation_emg_sizes = (
            {k: dataset["validation/emg"][k].shape for k in dataset["validation/emg"]}
            if "emg" in dataset["validation"]
            else {}
        )

        # Calculate memory usage
        total_size_bytes = 0
        split_sizes = {}
        for split in ["training", "testing", "validation"]:
            split_size_bytes = 0
            for group in ["emg", "ground_truth"]:
                if group in dataset[split]:
                    for k in dataset[split][group]:
                        arr = dataset[split][group][k]
                        item_size = np.dtype(arr.dtype).itemsize
                        arr_size = np.prod(arr.shape) * item_size
                        split_size_bytes += arr_size
                        total_size_bytes += arr_size
            split_sizes[split] = split_size_bytes / (1024 * 1024)  # Convert to MB

        # Total size in MB
        total_size_mb = total_size_bytes / (1024 * 1024)

        # Create a visually appealing summary with Rich
        self.console.rule("[bold blue]DATASET CREATION COMPLETED", style="blue double")
        self.console.print()  # Add empty line

        # Summary table
        summary_table = Table(
            title="Dataset Summary",
            show_header=True,
            header_style="bold magenta",
            box=box.ROUNDED,
            padding=(0, 2),
            width=60,
            title_style="bold magenta",
            title_justify="center",
        )
        summary_table.add_column("Metric", style="dim", width=30)
        summary_table.add_column("Value", style="green")

        # Add summary metrics
        summary_table.add_row("Total tasks", str(len(self.tasks_to_use)))
        summary_table.add_row(
            "Training samples",
            str(
                dataset["training/label"].shape[0]
                if "label" in dataset["training"]
                else 0
            ),
        )
        summary_table.add_row(
            "Testing samples",
            str(
                dataset["testing/label"].shape[0]
                if "label" in dataset["testing"]
                else 0
            ),
        )
        summary_table.add_row(
            "Validation samples",
            str(
                dataset["validation/label"].shape[0]
                if "label" in dataset["validation"]
                else 0
            ),
        )
        summary_table.add_row("Total dataset size", f"{total_size_mb:.2f} MB")

        # Add split sizes
        for split, size_mb in split_sizes.items():
            summary_table.add_row(
                f"{split.capitalize()} split size", f"{size_mb:.2f} MB"
            )

        self.console.print(summary_table)
        self.console.print()  # Add empty line

        # Dataset structure tree
        structure_tree = Tree("[bold yellow]Dataset Structure")

        for split, sizes in [
            ("Training", training_emg_sizes),
            ("Testing", testing_emg_sizes),
            ("Validation", validation_emg_sizes),
        ]:
            if sizes:
                split_branch = structure_tree.add(f"[bold cyan]{split}")

                # EMG representations
                emg_branch = split_branch.add("[bold green]EMG Representations")
                for k, shape in sizes.items():
                    emg_branch.add(f"{k}: {shape}")

        self.console.print(structure_tree)
        self.console.print()  # Add empty line

        self.console.rule(
            "[bold green]Dataset Creation Successfully Completed!", style="green double"
        )
