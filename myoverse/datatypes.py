import copy
import os
import pickle
from abc import abstractmethod
from typing import Dict, Optional, Sequence, TypedDict, Any, Union, List, Tuple

import mplcursors
import networkx
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

from myoverse.datasets.filters._template import FilterBaseClass

Representation = TypedDict(
    "Representation",
    {"data": np.ndarray, "filter_sequence": List[FilterBaseClass]},
)

# Add the standalone function at the module level (before any class definitions)
def create_grid_layout(
    rows: int, 
    cols: int, 
    n_electrodes: int = None, 
    fill_pattern: str = 'row',
    missing_indices: List[Tuple[int, int]] = None
) -> np.ndarray:
    """Creates a grid layout based on specified parameters.
    
    Parameters
    ----------
    rows : int
        Number of rows in the grid.
    cols : int
        Number of columns in the grid.
    n_electrodes : int, optional
        Number of electrodes in the grid. If None, will be set to rows*cols minus 
        the number of missing indices. Default is None.
    fill_pattern : str, optional
        Pattern to fill the grid. Options are 'row' (row-wise) or 'column' (column-wise).
        Default is 'row'.
    missing_indices : List[Tuple[int, int]], optional
        List of (row, col) indices that should be left empty (-1). Default is None.
        
    Returns
    -------
    np.ndarray
        2D array representing the grid layout.
        
    Raises
    ------
    ValueError
        If the parameters are invalid.
        
    Examples
    --------
    >>> import numpy as np
    >>> from myoverse.datatypes import create_grid_layout
    >>> 
    >>> # Create a 4×4 grid with row-wise numbering (0-15)
    >>> grid1 = create_grid_layout(4, 4, fill_pattern='row')
    >>> print(grid1)
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]
     [12 13 14 15]]
    >>> 
    >>> # Create a 4×4 grid with column-wise numbering (0-15)
    >>> grid2 = create_grid_layout(4, 4, fill_pattern='column')
    >>> print(grid2)
    [[ 0  4  8 12]
     [ 1  5  9 13]
     [ 2  6 10 14]
     [ 3  7 11 15]]
    >>> 
    >>> # Create a 3×3 grid with only 8 electrodes (missing bottom-right)
    >>> grid3 = create_grid_layout(3, 3, 8, 'row', 
    ...                           missing_indices=[(2, 2)])
    >>> print(grid3)
    [[ 0  1  2]
     [ 3  4  5]
     [ 6  7 -1]]
    """
    # Initialize grid with -1 (gaps)
    grid = np.full((rows, cols), -1, dtype=int)
    
    # Process missing indices
    if missing_indices is None:
        missing_indices = []
    
    missing_positions = set((r, c) for r, c in missing_indices if 0 <= r < rows and 0 <= c < cols)
    max_electrodes = rows * cols - len(missing_positions)
    
    # Validate n_electrodes
    if n_electrodes is None:
        n_electrodes = max_electrodes
    elif n_electrodes > max_electrodes:
        raise ValueError(
            f"Number of electrodes ({n_electrodes}) exceeds available positions "
            f"({max_electrodes} = {rows}×{cols} - {len(missing_positions)} missing)"
        )
    
    # Fill the grid based on the pattern
    electrode_idx = 0
    if fill_pattern.lower() == 'row':
        for r in range(rows):
            for c in range(cols):
                if (r, c) not in missing_positions and electrode_idx < n_electrodes:
                    grid[r, c] = electrode_idx
                    electrode_idx += 1
    elif fill_pattern.lower() == 'column':
        for c in range(cols):
            for r in range(rows):
                if (r, c) not in missing_positions and electrode_idx < n_electrodes:
                    grid[r, c] = electrode_idx
                    electrode_idx += 1
    else:
        raise ValueError(f"Invalid fill pattern: {fill_pattern}. Use 'row' or 'column'.")
    
    return grid

class _Data:
    """Base class for all data types.
    
    This class provides common functionality for handling different types of data,
    including maintaining original and processed representations, tracking filters
    applied, and managing data flow.
    
    Attributes
    ----------
    sampling_frequency : float
        The sampling frequency of the data.
    input_data : np.ndarray
        The raw input data.
    processed_representations : Dict[str, np.ndarray] 
        Dictionary of all processed representations of the data.
    is_chunked : Dict[str, bool]
        Dictionary indicating whether each representation is chunked or not.
        
    Examples
    --------
    This is an abstract base class and should not be instantiated directly.
    Instead, use one of the concrete subclasses like EMGData or KinematicsData:
    
    >>> import numpy as np
    >>> from myoverse.datatypes import EMGData
    >>> 
    >>> # Create sample data
    >>> data = np.random.randn(16, 1000)
    >>> emg = EMGData(data, 2000)  # 2000 Hz sampling rate
    >>> 
    >>> # Access attributes from the base _Data class
    >>> print(f"Sampling frequency: {emg.sampling_frequency} Hz")
    >>> print(f"Is input data chunked: {emg.is_chunked['Input']}")
    """
    
    def __init__(self, raw_data: np.ndarray, sampling_frequency: float):
        """Initialize the data object.
        
        Parameters
        ----------
        raw_data : np.ndarray
            The raw data to store.
        sampling_frequency : float
            The sampling frequency of the data.
            
        Raises
        ------
        ValueError
            If the sampling frequency is less than or equal to 0.
        """
        self.sampling_frequency = sampling_frequency

        if self.sampling_frequency <= 0:
            raise ValueError("The sampling frequency should be greater than 0.")

        self.__input_representation_name = "Input"
        self.__output_representation_name = "Output"
        self.__last_representation_name = "Last"

        self._data: Dict[str, Union[np.ndarray, str]] = {
            self.__input_representation_name: raw_data,
        }
        self._filters_used: Dict[str, FilterBaseClass] = {}

        self._processed_representations = networkx.DiGraph()
        self._processed_representations.add_node(self.__input_representation_name)
        self._processed_representations.add_node(self.__output_representation_name)

        self.__last_processing_step = self.__input_representation_name

    @property
    def is_chunked(self) -> Dict[str, bool]:
        """Returns whether the data is chunked or not.

        Returns
        -------
        Dict[str, bool]
            A dictionary where the keys are the representations and the values are whether the data is chunked or not.
        """

        output = {}
        for key, value in self._data.items():
            output[key] = self._check_if_chunked(value)

            # check if key is a "Chunkize" filter or comes after it
            if (
                any(
                    [
                        "Chunkize" in x
                        for x in self._processed_representations.predecessors(key)
                    ]
                )
                or "Chunkize" in key
            ):
                output[key] = True

        return output

    @abstractmethod
    def _check_if_chunked(self, data: Union[np.ndarray, str]) -> bool:
        """Checks if the data is chunked or not.

        Parameters
        ----------
        data : Union[np.ndarray, str]
            The data to check.

        Returns
        -------
        bool
            Whether the data is chunked or not.
        """
        raise NotImplementedError(
            "This method should be implemented in the child class."
        )

    @property
    def input_data(self) -> np.ndarray:
        """Returns the input data."""
        return self._data[self.__input_representation_name]

    @input_data.setter
    def input_data(self, value: np.ndarray):
        raise RuntimeError("This property is read-only.")

    @property
    def processed_representations(self) -> Dict[str, np.ndarray]:
        """Returns the processed representations of the data."""
        return self._data

    @processed_representations.setter
    def processed_representations(self, value: Dict[str, Representation]):
        raise RuntimeError("This property is read-only.")

    @property
    def output_representations(self) -> Dict[str, np.ndarray]:
        """Returns the output representations of the data."""

        return {
            key: value
            for key, value in self._data.items()
            if key
            in list(
                self._processed_representations.predecessors(
                    self.__output_representation_name
                )
            )
        }

    @output_representations.setter
    def output_representations(self, value: Dict[str, Representation]):
        raise RuntimeError("This property is read-only.")

    @property
    def _last_processing_step(self) -> str:
        """Returns the last processing step applied to the data.

        Returns
        -------
        str
            The last processing step applied to the data.
        """
        if self.__last_processing_step is None:
            raise ValueError("No processing steps have been applied.")

        return self.__last_processing_step

    @_last_processing_step.setter
    def _last_processing_step(self, value: str):
        """Sets the last processing step applied to the data.

        Parameters
        ----------
        value : str
            The last processing step applied to the data.
        """
        self.__last_processing_step = value

    @abstractmethod
    def plot(self, *_: Any, **__: Any):
        """Plots the data."""
        raise NotImplementedError(
            "This method should be implemented in the child class."
        )

    def plot_graph(self):
        """Draws the graph of the processed representations."""
        pos = nx.spectral_layout(self._processed_representations)

        # if the output node is more to the left than the input node, flip the graph
        if (
            pos[self.__output_representation_name][0]
            < pos[self.__input_representation_name][0]
        ):
            pos = {k: (-v[0], v[1]) for k, v in pos.items()}

        _, ax = plt.subplots()

        # color the raw and output nodes differently
        node_colors = []
        for node in self._processed_representations.nodes:
            if node == self.__input_representation_name:
                node_colors.append("red")
            elif node == self.__output_representation_name:
                node_colors.append("green")
            elif isinstance(self._data[node], str):
                node_colors.append("black")
            else:
                node_colors.append("blue")

        # number the nodes and add the labels
        nx.draw_networkx_labels(
            self._processed_representations,
            pos,
            labels={
                node: str(index)
                for index, node in enumerate(
                    [
                        x
                        for x in self._processed_representations.nodes
                        if x != self.__output_representation_name
                    ]
                )
            },
            ax=ax,
            font_color="white",
            font_size=18,
        )

        nx.draw(
            self._processed_representations,
            pos=pos,
            node_color=node_colors,
            ax=ax,
            alpha=0.5,
            with_labels=False,  # Disable labels here,
            node_size=1000,
        )

        # Add interactive labels with mplcursors
        cursor = mplcursors.cursor(ax.collections[0], hover=True)

        def on_hover(sel):
            hovered_node_name = str(
                list(self._processed_representations.nodes)[sel.index]
            )

            annotation = hovered_node_name
            annotation += "\n\n"

            # add whether the node needs to be recomputed
            need_recompute = False
            if hovered_node_name != self.__output_representation_name:
                if isinstance(self._data[hovered_node_name], str):
                    need_recompute = True
                    annotation += "needs to be\nrecomputed\n\n"

            # add info whether the node is chunked or not
            annotation += "chunked: "
            if hovered_node_name != self.__output_representation_name:
                annotation += str(self.is_chunked[hovered_node_name])
            else:
                annotation += "(see previous node(s))"

            # add shape information to the annotation
            annotation += "\n" + "shape: "
            if hovered_node_name != self.__output_representation_name:
                if not need_recompute:
                    annotation += str(self._data[hovered_node_name].shape)
                else:
                    annotation += self._data[hovered_node_name]
            else:
                annotation += "(see previous node(s))"

            sel.annotation.set_text(annotation)
            sel.annotation.get_bbox_patch().set(
                fc="white", alpha=0.8
            )  # Background color
            sel.annotation.set_fontsize(12)  # Font size
            sel.annotation.set_fontstyle("italic")

        cursor.connect("add", on_hover)

        plt.margins(0.4)
        plt.tight_layout()
        plt.show()

    def apply_filter(
        self,
        filter: FilterBaseClass,
        representation_to_filter: str,
        keep_representation_to_filter: bool = True,
    ) -> str:
        """Applies a filter to the data.

        Parameters
        ----------
        filter : callable
            The filter to apply.
        representation_to_filter : str
            The representation to filter.
        keep_representation_to_filter : bool
            Whether to keep the representation to filter or not.
            If the representation to filter is "raw", this parameter is ignored.

        Returns
        -------
        str
            The name of the representation after applying the filter.
        """
        representation_name = filter.name

        # check if the representation to filter is the last representation
        # if so, we can use the last processing step as the representation to filter
        if representation_to_filter == self.__last_representation_name:
            representation_to_filter = self._last_processing_step

        # check if the representation to filter exists
        # if not add a node to the graph and an edge from the representation to filter to the representation to add
        if representation_to_filter not in self._processed_representations:
            self._processed_representations.add_node(representation_name)
        if not self._processed_representations.has_edge(
            representation_to_filter, representation_name
        ):
            self._processed_representations.add_edge(
                representation_to_filter, representation_name
            )

        # check if the filter is going to be an output
        # if so, add an edge from the representation to add to the output node
        if filter.is_output:
            self._processed_representations.add_edge(
                representation_name, self.__output_representation_name
            )

        # save the used filter
        self._filters_used[representation_name] = filter

        # set the input_is_chunked parameter if it is not set by looking at the representation to filter
        if filter.input_is_chunked is None:
            filter.input_is_chunked = self.is_chunked[representation_to_filter]

        # apply the filter
        self._data[representation_name] = filter(self[representation_to_filter])

        # remove the representation to filter if needed
        if not keep_representation_to_filter:
            self.delete_data(representation_to_filter)

        # set the last processing step
        self._last_processing_step = representation_name

        return representation_name

    def apply_filter_sequence(
        self,
        filter_sequence: List[FilterBaseClass],
        representation_to_filter: str,
        keep_individual_filter_steps: bool = True,
        keep_representation_to_filter: bool = True,
    ):
        """Applies a sequence of filters to the data.

        Parameters
        ----------
        filter_sequence : list[FilterBaseClass]
            The sequence of filters to apply.
        representation_to_filter : str
            The representation to filter.
        keep_individual_filter_steps : bool
            Whether to keep the results of each filter or not.
        keep_representation_to_filter : bool
            Whether to keep the representation to filter or not.
            If the representation to filter is "raw", this parameter is ignored.

        Raises
        ------
        ValueError
            If no filters were provided.
        """
        # check if there are any filters
        if len(filter_sequence) == 0:
            raise ValueError("No filters were provided.")

        what_to_filter = representation_to_filter
        for f in filter_sequence:
            what_to_filter = self.apply_filter(
                filter=f,
                representation_to_filter=what_to_filter,
                keep_representation_to_filter=True,
            )

        # remove the individual filter steps if needed. The last step is always kept
        if not keep_individual_filter_steps:
            for f in filter_sequence[:-1]:
                self.delete_data(f.name)

        # remove the representation to filter if needed
        if not keep_representation_to_filter:
            self.delete_data(representation_to_filter)

    def apply_filter_pipeline(
        self,
        filter_pipeline: List[List[FilterBaseClass]],
        representations_to_filter: List[str],
        keep_individual_filter_steps: bool = True,
        keep_representation_to_filter: bool = True,
    ):
        """Applies a pipeline of filters to the data.

        Parameters
        ----------
        filter_pipeline : list[list[FilterBaseClass]]
            The pipeline of filters to apply.
        representations_to_filter : list[str]
            The representations to filter.
            .. note :: The length of the representations to filter should be the same as the length of the filter pipeline.
        keep_individual_filter_steps : bool
            Whether to keep the results of each filter or not.
        keep_representation_to_filter : bool
            Whether to keep the representation to filter or not.
            If the representation to filter is "raw", this parameter is ignored.

        Raises
        ------
        ValueError
            If the number of filters and representations to filter is different.
        """
        if len(filter_pipeline) == 0:
            return

        if len(filter_pipeline) != len(representations_to_filter):
            raise ValueError(
                "The number of filters and representations to filter should be the same."
            )

        for filter_sequence, representation_to_filter in zip(
            filter_pipeline, representations_to_filter
        ):
            self.apply_filter_sequence(
                filter_sequence=filter_sequence,
                representation_to_filter=representation_to_filter,
                keep_individual_filter_steps=keep_individual_filter_steps,
                keep_representation_to_filter=keep_representation_to_filter,
            )

        # remove the representation to filter if needed
        if not keep_representation_to_filter:
            self.delete_data(representations_to_filter[-1])

        # remove the individual filter steps if needed. The last step is always kept
        if not keep_individual_filter_steps:
            for filter_sequence in filter_pipeline:
                for f in filter_sequence[:-1]:
                    self.delete_data(f.name)

    def get_representation_history(self, representation: str) -> List[str]:
        """Returns the history of a representation.

        Parameters
        ----------
        representation : str
            The representation to get the history of.

        Returns
        -------
        list[str]
            The history of the representation.
        """
        return list(
            nx.shortest_path(
                self._processed_representations,
                self.__input_representation_name,
                representation,
            )
        )

    def __repr__(self) -> str:
        representation = (
            f"{self.__class__.__name__}; "
            f"Sampling frequency: {self.sampling_frequency} Hz; (0) Input {self.input_data.shape}"
        )

        if len(self._processed_representations.nodes) < 3:
            return representation

        representation += "; Filter(s): "

        representation_indices = {
            key: index for index, key in enumerate(self._filters_used.keys())
        }

        for filter_index, (filter_name, filter_representation) in enumerate(
            self._data.items()
        ):
            if filter_name == self.__input_representation_name:
                continue

            history = self.get_representation_history(filter_name)
            history = " -> ".join(
                [str(representation_indices[rep] + 1) for rep in history[1:]]
            )

            representation += (
                f"({filter_index} | " + history + ") "
                f"{'(Output) ' if filter_name in self._processed_representations.predecessors('Output') else ''}"
                f"{filter_name} "
                f"{filter_representation.shape if not isinstance(filter_representation, str) else filter_representation}; "
            )

        representation = representation[:-2]

        return representation

    def __str__(self) -> str:
        return (
            "--\n"
            + self.__repr__()
            .replace("; ", "\n")
            .replace("Filter(s): ", "\nFilter(s):\n")
            + "\n--"
        )

    def __getitem__(self, key: str) -> np.ndarray:
        if key == self.__input_representation_name:
            # Use array.view() for more efficient copying when possible
            data = self.input_data
            return data.view() if data.flags.writeable else data.copy()

        if key == self.__last_representation_name:
            return self[self._last_processing_step]

        if key not in self._processed_representations:
            raise KeyError(f'The representation "{key}" does not exist.')

        data_to_return = self._data[key]

        if isinstance(data_to_return, str):
            print(f'Recomputing representation "{key}"')

            history = self.get_representation_history(key)
            self.apply_filter_sequence(
                filter_sequence=[
                    self._filters_used[filter_name] for filter_name in history[1:]
                ],
                representation_to_filter=history[0],
            )

        # Use view when possible for more efficient memory usage
        data = self._data[key]
        return data.view() if data.flags.writeable else data.copy()

    def __setitem__(self, key: str, value: np.ndarray) -> None:
        raise RuntimeError(
            "This method is not supported. Run apply_filter or apply_filters instead."
        )

    def delete_data(self, representation_to_delete: str):
        """Delete data from a representation while keeping its metadata.

        This replaces the actual numpy array with a string representation of its shape,
        saving memory while allowing regeneration when needed.

        Parameters
        ----------
        representation_to_delete : str
            The representation to delete the data from.
        """
        if representation_to_delete == self.__input_representation_name:
            return
        if representation_to_delete == self.__last_representation_name:
            self.delete_data(self._last_processing_step)
            return

        if representation_to_delete not in self._data:
            raise KeyError(
                f'The representation "{representation_to_delete}" does not exist.'
            )

        if isinstance(self._data[representation_to_delete], np.ndarray):
            self._data[representation_to_delete] = str(
                self._data[representation_to_delete].shape
            )

    def delete_history(self, representation_to_delete: str):
        """Delete the processing history for a representation.

        Parameters
        ----------
        representation_to_delete : str
            The representation to delete the history for.
        """
        if representation_to_delete == self.__input_representation_name:
            return
        if representation_to_delete == self.__last_representation_name:
            self.delete_history(self._last_processing_step)
            return

        if representation_to_delete not in self._processed_representations.nodes:
            raise KeyError(
                f'The representation "{representation_to_delete}" does not exist.'
            )

        self._filters_used.pop(representation_to_delete, None)
        self._processed_representations.remove_node(representation_to_delete)

    def delete(self, representation_to_delete: str):
        """Delete both the data and history for a representation.

        Parameters
        ----------
        representation_to_delete : str
            The representation to delete.
        """
        self.delete_data(representation_to_delete)
        self.delete_history(representation_to_delete)

    def __copy__(self) -> "_Data":
        """Create a shallow copy of the instance.

        Returns
        -------
        _Data
            A shallow copy of the instance.
        """
        new_instance = self.__class__(
            self._data[self.__input_representation_name].copy(), self.sampling_frequency
        )
        # Use a more efficient approach for copying the graph
        new_instance._processed_representations = self._processed_representations.copy()
        # Only deepcopy the data dictionary - full deepcopy is often unnecessary
        new_instance._data = copy.deepcopy(self._data)
        new_instance._last_processing_step = self._last_processing_step
        new_instance._filters_used = copy.deepcopy(self._filters_used)

        return new_instance

    def save(self, filename: str):
        """Save the data to a file.

        Parameters
        ----------
        filename : str
            The name of the file to save the data to.
        """
        # Make sure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename: str) -> "_Data":
        """Load data from a file.

        Parameters
        ----------
        filename : str
            The name of the file to load the data from.

        Returns
        -------
        _Data
            The loaded data.
        """
        with open(filename, "rb") as f:
            return pickle.load(f)

    def memory_usage(self) -> Dict[str, Tuple[str, int]]:
        """Calculate memory usage of each representation.

        Returns
        -------
        Dict[str, Tuple[str, int]]
            Dictionary with representation names as keys and tuples containing
            shape as string and memory usage in bytes as values.
        """
        memory_usage = {}
        for key, value in self._data.items():
            if isinstance(value, np.ndarray):
                memory_usage[key] = (str(value.shape), value.nbytes)
            else:
                memory_usage[key] = (
                    value,
                    0,
                )  # Placeholder shape string uses negligible memory

        return memory_usage


class EMGData(_Data):
    """Class for storing EMG data.

    Attributes
    ----------
    input_data : np.ndarray
        The raw EMG data. The shape of the array should be (n_channels, n_samples) or (n_chunks, n_channels, n_samples).
    sampling_frequency : float
        The sampling frequency of the EMG data.
    grid_layouts : Optional[List[np.ndarray]]
        List of 2D arrays specifying the exact electrode arrangement for each grid.
        Each array element contains the electrode index (0-based) or -1 for gaps/missing electrodes.
        This provides precise control over electrode numbering patterns (row-wise, column-wise, etc.)
        and allows specifying exactly which positions have electrodes.
    processed_data : Dict[str, np.ndarray]
        A dictionary where the keys are the names of filters applied
        to the EMG data and the values are the processed EMG data.

    Parameters
    ----------
    is_chunked : bool
        Whether the EMG data is chunked or not.
        If True, the shape of the raw EMG data should be (n_chunks, n_channels, n_samples).

    Examples
    --------
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
    >>> # Create an EMGData object with grid layouts
    >>> # Define a 4×4 electrode grid with row-wise numbering
    >>> grid = create_grid_layout(4, 4, fill_pattern='row')
    >>> emg_with_grid = EMGData(emg_data, sampling_freq, grid_layouts=[grid])
    """

    def __init__(
        self,
        input_data: np.ndarray,
        sampling_frequency: float,
        grid_layouts: Optional[List[np.ndarray]] = None,
    ):
        """Initializes the EMGData object.

        Parameters
        ----------
        input_data : np.ndarray
            The raw EMG data. The shape of the array should be (n_channels, n_samples)
             or (n_chunks, n_channels, n_samples).
        sampling_frequency : float
            The sampling frequency of the EMG data.
        grid_layouts : Optional[List[np.ndarray]], optional
            List of 2D arrays specifying the exact electrode arrangement for each grid.
            Each array element contains the electrode index (0-based) or -1 for gaps/missing electrodes.
            For example, a 2×3 grid with electrodes numbered column-wise and one gap might be:
            [[0, 2, 4], [1, -1, 5]]
            Default is None.
            
        Examples
        --------
        >>> import numpy as np
        >>> from myoverse.datatypes import EMGData, create_grid_layout
        >>> 
        >>> # Create sample EMG data (64 channels, 1000 samples)
        >>> emg_data = np.random.randn(64, 1000)
        >>> sampling_freq = 2000  # 2000 Hz
        >>> 
        >>> # Create a simple EMGData object
        >>> emg = EMGData(emg_data, sampling_freq)
        >>> 
        >>> # Create EMGData with a custom electrode grid layout
        >>> # 8×8 grid with column-wise numbering
        >>> grid = np.full((8, 8), -1)  # Initialize with gaps
        >>> for c in range(8):
        ...     for r in range(8):
        ...         idx = c * 8 + r
        ...         if idx < 64:  # Only fill up to 64 electrodes
        ...             grid[r, c] = idx
        >>> 
        >>> emg_with_grid = EMGData(emg_data, sampling_freq, grid_layouts=[grid])
        """
        if input_data.ndim != 2 and input_data.ndim != 3:
            raise ValueError(
                "The shape of the raw EMG data should be (n_channels, n_samples) or (n_chunks, n_channels, n_samples)."
            )
        super().__init__(input_data, sampling_frequency)

        self.grid_layouts = None  # Initialize to None first
        
        # Get the number of electrodes from the data
        data_electrodes = input_data.shape[0] if input_data.ndim == 2 else input_data.shape[1]
        
        # Process and validate grid layouts if provided
        if grid_layouts is not None:
            # Check that each layout array is 2D
            for i, layout in enumerate(grid_layouts):
                if not isinstance(layout, np.ndarray) or layout.ndim != 2:
                    raise ValueError(f"Grid layout {i+1} must be a 2D numpy array")
                
                # Count valid electrodes (non-negative values)
                valid_electrodes = np.sum(layout >= 0)
                
                # Check for duplicate electrode indices
                valid_indices = layout[layout >= 0]
                if len(np.unique(valid_indices)) != len(valid_indices):
                    raise ValueError(f"Grid layout {i+1} contains duplicate electrode indices")
                
                # Check if any index is out of bounds
                if np.any(valid_indices >= data_electrodes):
                    raise ValueError(
                        f"Grid layout {i+1} contains electrode indices that exceed the total "
                        f"number of electrodes ({data_electrodes})"
                    )
            
            # Store the validated grid layouts
            self.grid_layouts = grid_layouts

    def _get_grid_dimensions(self):
        """Get dimensions and electrode counts for each grid.
        
        Returns
        -------
        List[Tuple[int, int, int]]
            List of (rows, cols, electrodes) tuples for each grid, or empty list if no grid layouts are available.
        """
        if self.grid_layouts is None:
            return []
        
        return [
            (layout.shape[0], layout.shape[1], np.sum(layout >= 0))
            for layout in self.grid_layouts
        ]

    def _check_if_chunked(self, data: Union[np.ndarray, str]) -> bool:
        """Checks if the data is chunked or not.

        Parameters
        ----------
        data : Union[np.ndarray, str]
            The data to check.

        Returns
        -------
        bool
            Whether the data is chunked or not.
        """
        if isinstance(data, str):
            return len(data.split(",")) == 3
        return data.ndim == 3

    def plot(
        self,
        representation: str,
        nr_of_grids: Optional[int] = None,
        nr_of_electrodes_per_grid: Optional[int] = None,
        scaling_factor: Union[float, List[float]] = 20.0,
        use_grid_layouts: bool = True,
    ):
        """Plots the data for a specific representation.

        Parameters
        ----------
        representation : str
            The representation to plot.
        nr_of_grids : Optional[int], optional
            The number of electrode grids to plot. If None and grid_layouts is provided, 
            will use the number of grids in grid_layouts. Default is None.
        nr_of_electrodes_per_grid : Optional[int], optional
            The number of electrodes per grid to plot. If None, will be determined from data shape
            or grid_layouts if available. Default is None.
        scaling_factor : Union[float, List[float]], optional
            The scaling factor for the data. The default is 20.0.
            If a list is provided, the scaling factor for each grid is used.
        use_grid_layouts : bool, optional
            Whether to use the grid_layouts for plotting. Default is True.
            If False, will use the nr_of_grids and nr_of_electrodes_per_grid parameters.
            
        Examples
        --------
        >>> import numpy as np
        >>> from myoverse.datatypes import EMGData, create_grid_layout
        >>> 
        >>> # Create sample EMG data (64 channels, 1000 samples)
        >>> emg_data = np.random.randn(64, 1000)
        >>> 
        >>> # Create EMGData with two 4×8 grids (32 electrodes each)
        >>> grid1 = create_grid_layout(4, 8, 32, fill_pattern='row')
        >>> grid2 = create_grid_layout(4, 8, 32, fill_pattern='row')
        >>> 
        >>> # Adjust indices for second grid
        >>> grid2[grid2 >= 0] += 32
        >>> 
        >>> emg = EMGData(emg_data, 2000, grid_layouts=[grid1, grid2])
        >>> 
        >>> # Plot the raw data using the grid layouts
        >>> emg.plot('Input')
        >>> 
        >>> # Adjust scaling for better visualization
        >>> emg.plot('Input', scaling_factor=[15.0, 25.0])
        >>> 
        >>> # Plot without using grid layouts (specify manual grid configuration)
        >>> emg.plot('Input', nr_of_grids=2, nr_of_electrodes_per_grid=32, 
        ...         use_grid_layouts=False)
        """
        data = self[representation]

        # Use grid_layouts if available and requested
        if self.grid_layouts is not None and use_grid_layouts:
            grid_dimensions = self._get_grid_dimensions()
            
            if nr_of_grids is not None and nr_of_grids != len(self.grid_layouts):
                print(f"Warning: nr_of_grids ({nr_of_grids}) does not match grid_layouts length "
                      f"({len(self.grid_layouts)}). Using grid_layouts.")
            
            nr_of_grids = len(self.grid_layouts)
            electrodes_per_grid = [dims[2] for dims in grid_dimensions]
        else:
            # Auto-determine nr_of_grids if not provided
            if nr_of_grids is None:
                nr_of_grids = 1
            
            # Auto-determine nr_of_electrodes_per_grid if not provided
            if nr_of_electrodes_per_grid is None:
                if self.is_chunked[representation]:
                    total_electrodes = data.shape[1]
                else:
                    total_electrodes = data.shape[0]
                
                # Try to determine a sensible default
                nr_of_electrodes_per_grid = total_electrodes // nr_of_grids
            
            electrodes_per_grid = [nr_of_electrodes_per_grid] * nr_of_grids

        # Prepare scaling factors
        if isinstance(scaling_factor, float):
            scaling_factor = [scaling_factor] * nr_of_grids

        assert len(scaling_factor) == nr_of_grids, (
            "The number of scaling factors should be equal to the number of grids."
        )

        fig = plt.figure(figsize=(5*nr_of_grids, 6))
        
        # Calculate electrode index offset for each grid
        electrode_offsets = [0]
        for i in range(len(electrodes_per_grid)-1):
            electrode_offsets.append(electrode_offsets[-1] + electrodes_per_grid[i])
        
        # Make a subplot for each grid
        for grid_idx in range(nr_of_grids):
            ax = fig.add_subplot(1, nr_of_grids, grid_idx + 1)
            
            grid_title = f"Grid {grid_idx + 1}"
            if self.grid_layouts is not None and use_grid_layouts:
                rows, cols, _ = grid_dimensions[grid_idx]
                grid_title += f" ({rows}×{cols})"
            ax.set_title(grid_title)
            
            offset = electrode_offsets[grid_idx]
            n_electrodes = electrodes_per_grid[grid_idx]
            
            for electrode_idx in range(n_electrodes):
                data_idx = offset + electrode_idx
                if self.is_chunked[representation]:
                    # Handle chunked data - plot first chunk for visualization
                    ax.plot(
                        data[0, data_idx] + electrode_idx * data[0].mean() * scaling_factor[grid_idx]
                    )
                else:
                    ax.plot(
                        data[data_idx] + electrode_idx * data.mean() * scaling_factor[grid_idx]
                    )

            ax.set_xlabel("Time (samples)")
            ax.set_ylabel("Electrode #")

            # Set the y-axis ticks to the electrode numbers beginning from 1
            mean_val = data[0].mean() if self.is_chunked[representation] else data.mean()
            ax.set_yticks(
                np.arange(0, n_electrodes) * mean_val * scaling_factor[grid_idx],
                np.arange(1, n_electrodes + 1),
            )

            # Only for grid 1 keep the y-axis label
            if grid_idx != 0:
                ax.set_ylabel("")

        plt.tight_layout()
        plt.show()
        
    def plot_grid_layout(self, grid_idx: int = 0, show_indices: bool = True):
        """Plots the 2D layout of a specific electrode grid.
        
        Parameters
        ----------
        grid_idx : int, optional
            The index of the grid to plot. Default is 0.
        show_indices : bool, optional
            Whether to show the electrode indices in the plot. Default is True.
            
        Raises
        ------
        ValueError
            If grid_layouts is not available or the grid_idx is out of range.
            
        Examples
        --------
        >>> import numpy as np
        >>> from myoverse.datatypes import EMGData, create_grid_layout
        >>> 
        >>> # Create sample EMG data (64 channels, 1000 samples)
        >>> emg_data = np.random.randn(64, 1000)
        >>> 
        >>> # Create an 8×8 grid with some missing electrodes
        >>> grid = create_grid_layout(8, 8, 64, fill_pattern='row',
        ...                          missing_indices=[(7, 7), (0, 0)])
        >>> 
        >>> emg = EMGData(emg_data, 2000, grid_layouts=[grid])
        >>> 
        >>> # Visualize the grid layout
        >>> emg.plot_grid_layout(0)
        >>> 
        >>> # Visualize without showing electrode indices
        >>> emg.plot_grid_layout(0, show_indices=False)
        """
        if self.grid_layouts is None:
            raise ValueError("Cannot plot grid layout: grid_layouts not provided.")
            
        if grid_idx < 0 or grid_idx >= len(self.grid_layouts):
            raise ValueError(f"Grid index {grid_idx} out of range (0 to {len(self.grid_layouts)-1}).")
            
        # Get the grid layout
        grid = self.grid_layouts[grid_idx]
        rows, cols = grid.shape
        
        # Get number of electrodes
        n_electrodes = np.sum(grid >= 0)
        grid_title = f"Grid {grid_idx+1} layout ({rows}×{cols}) with {n_electrodes} electrodes"
        
        # Create a masked array for plotting
        masked_grid = np.ma.masked_less(grid, 0)
        
        # Plot the grid
        fig, ax = plt.subplots(figsize=(cols/2 + 3, rows/2 + 1))
        cmap = plt.cm.viridis
        cmap.set_bad('white', 1.0)
        im = ax.imshow(masked_grid, cmap=cmap)
        
        # Add grid lines
        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
        
        # Add electrode numbers
        if show_indices:
            for i in range(rows):
                for j in range(cols):
                    if grid[i, j] >= 0:
                        ax.text(j, i, str(grid[i, j]), ha="center", va="center", 
                                color="w", fontweight='bold')
        
        # Add a title
        plt.title(grid_title)
        
        # Fix the aspect ratio and display
        plt.tight_layout()
        plt.show()


class KinematicsData(_Data):
    """Class for storing kinematics data.

    Attributes
    ----------
    input_data : np.ndarray
        The raw kinematics data. The shape of the array should be (n_joints, 3, n_samples)
        or (n_chunks, n_joints, 3, n_samples).
        The 3 represents the x, y, and z coordinates of the joints.

    sampling_frequency : float
        The sampling frequency of the kinematics data.

    processed_data : Dict[str, np.ndarray]
        A dictionary where the keys are the names of filters applied to the kinematics data and
        the values are the processed kinematics data.

    Parameters
    ----------
    is_chunked : bool
        Whether the kinematics data is chunked or not.
        If True, the shape of the raw kinematics data should be (n_chunks, n_joints, 3, n_samples).
        
    Examples
    --------
    >>> import numpy as np
    >>> from myoverse.datatypes import KinematicsData
    >>> 
    >>> # Create sample kinematics data (16 joints, 3 coordinates, 1000 samples)
    >>> # Each joint has x, y, z coordinates
    >>> joint_data = np.random.randn(16, 3, 1000)
    >>> 
    >>> # Create a KinematicsData object with 100 Hz sampling rate
    >>> kinematics = KinematicsData(joint_data, 100)
    >>> 
    >>> # Access the raw data
    >>> raw_data = kinematics.input_data
    >>> print(f"Data shape: {raw_data.shape}")
    Data shape: (16, 3, 1000)
    """

    def __init__(self, input_data: np.ndarray, sampling_frequency: float):
        """Initializes the KinematicsData object.

        Parameters
        ----------
        input_data : np.ndarray
            The raw kinematics data. The shape of the array should be (n_joints, 3, n_samples)
            or (n_chunks, n_joints, 3, n_samples).
            The 3 represents the x, y, and z coordinates of the joints.
        sampling_frequency : float
            The sampling frequency of the kinematics data.
            
        Examples
        --------
        >>> import numpy as np
        >>> from myoverse.datatypes import KinematicsData
        >>> 
        >>> # Create sample non-chunked data (21 joints, 3 coordinates, 500 samples)
        >>> joint_data = np.random.randn(21, 3, 500)
        >>> kinematics = KinematicsData(joint_data, 120)  # 120 Hz sampling rate
        >>> 
        >>> # Create sample chunked data (10 chunks, 21 joints, 3 coordinates, 100 samples)
        >>> chunked_data = np.random.randn(10, 21, 3, 100)
        >>> chunked_kinematics = KinematicsData(chunked_data, 120)
        >>> 
        >>> # Check if data is chunked
        >>> print(f"Is chunked: {chunked_kinematics.is_chunked['Input']}")
        Is chunked: True
        """
        if input_data.ndim != 3 and input_data.ndim != 4:
            raise ValueError(
                "The shape of the raw kinematics data should be (n_joints, 3, n_samples) "
                "or (n_chunks, n_joints, 3, n_samples)."
            )
        super().__init__(input_data, sampling_frequency)

    def _check_if_chunked(self, data: Union[np.ndarray, str]) -> bool:
        """Checks if the data is chunked or not.

        Parameters
        ----------
        data : Union[np.ndarray, str]
            The data to check.

        Returns
        -------
        bool
            Whether the data is chunked or not.
        """
        if isinstance(data, str):
            return len(data.split(",")) == 4
        return data.ndim == 4

    def plot(
        self, representation: str, nr_of_fingers: int, wrist_included: bool = True
    ):
        """Plots the data.

        Parameters
        ----------
        representation : str
            The representation to plot.
            .. important :: The representation should be a 3D tensor with shape (n_joints, 3, n_samples).
        nr_of_fingers : int
            The number of fingers to plot.
        wrist_included : bool, optional
            Whether the wrist is included in the representation. The default is True.
            .. note :: The wrist is always the first joint in the representation.

        Raises
        ------
        KeyError
            If the representation does not exist.
            
        Examples
        --------
        >>> import numpy as np
        >>> from myoverse.datatypes import KinematicsData
        >>> 
        >>> # Create sample kinematics data for a hand with 5 fingers
        >>> # 16 joints: 1 wrist + 3 joints for each of the 5 fingers
        >>> joint_data = np.random.randn(16, 3, 100)
        >>> kinematics = KinematicsData(joint_data, 100)
        >>> 
        >>> # Plot the kinematics data
        >>> kinematics.plot('Input', nr_of_fingers=5)
        >>> 
        >>> # Plot without wrist
        >>> kinematics.plot('Input', nr_of_fingers=5, wrist_included=False)
        """
        if representation not in self._data:
            raise KeyError(f'The representation "{representation}" does not exist.')

        kinematics = self[representation]

        if not wrist_included:
            kinematics = np.concatenate(
                [np.zeros((1, 3, kinematics.shape[2])), kinematics], axis=0
            )

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # get biggest axis range
        max_range = (
            np.array(
                [
                    kinematics[:, 0].max() - kinematics[:, 0].min(),
                    kinematics[:, 1].max() - kinematics[:, 1].min(),
                    kinematics[:, 2].max() - kinematics[:, 2].min(),
                ]
            ).max()
            / 2.0
        )

        # set axis limits
        ax.set_xlim(
            kinematics[:, 0].mean() - max_range,
            kinematics[:, 0].mean() + max_range,
        )
        ax.set_ylim(
            kinematics[:, 1].mean() - max_range,
            kinematics[:, 1].mean() + max_range,
        )
        ax.set_zlim(
            kinematics[:, 2].mean() - max_range,
            kinematics[:, 2].mean() + max_range,
        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        # create joint and finger plots
        (joints_plot,) = ax.plot(*kinematics[..., 0].T, "o", color="black")

        finger_plots = []
        for finger in range(nr_of_fingers):
            finger_plots.append(
                ax.plot(
                    *kinematics[
                        [0] + list(reversed(range(1 + finger * 4, 5 + finger * 4))),
                        :,
                        0,
                    ].T,
                    color="blue",
                )
            )

        samp = plt.axes([0.25, 0.02, 0.65, 0.03])
        sample_slider = Slider(
            samp,
            label="Sample (a. u.)",
            valmin=0,
            valmax=kinematics.shape[2] - 1,
            valstep=1,
            valinit=0,
        )

        def update(val):
            kinematics_new_sample = kinematics[..., int(val)]

            joints_plot._verts3d = tuple(kinematics_new_sample.T)

            for finger in range(nr_of_fingers):
                finger_plots[finger][0]._verts3d = tuple(
                    kinematics_new_sample[
                        [0] + list(reversed(range(1 + finger * 4, 5 + finger * 4))),
                        :,
                    ].T
                )

            fig.canvas.draw_idle()

        sample_slider.on_changed(update)
        plt.tight_layout()
        plt.show()


class VirtualHandKinematics(_Data):
    """Class for storing virtual hand kinematics data.

    Attributes
    ----------
    input_data : np.ndarray
        The raw kinematics data for a virtual hand. The shape of the array should be (9, n_samples)
        or (n_chunks, 9, n_samples).
        The 9 typically represents the degrees of freedom: wrist flexion/extension,
        wrist pronation/supination, wrist deviation, and the flexion of all 5 fingers.

    sampling_frequency : float
        The sampling frequency of the kinematics data.

    processed_data : Dict[str, np.ndarray]
        A dictionary where the keys are the names of filters applied to the kinematics data and
        the values are the processed kinematics data.

    Parameters
    ----------
    is_chunked : bool
        Whether the kinematics data is chunked or not.
        If True, the shape of the raw kinematics data should be (n_chunks, 9, n_samples).
    """

    def __init__(self, input_data: np.ndarray, sampling_frequency: float):
        """Initializes the VirtualHandKinematics object.

        Parameters
        ----------
        input_data : np.ndarray
            The raw kinematics data. The shape of the array should be (9, n_samples)
            or (n_chunks, 9, n_samples).
        sampling_frequency : float
            The sampling frequency of the kinematics data.
        """
        if input_data.ndim != 2 and input_data.ndim != 3:
            raise ValueError(
                "The shape of the raw kinematics data should be (9, n_samples) "
                "or (n_chunks, 9, n_samples)."
            )
        super().__init__(input_data, sampling_frequency)

    def _check_if_chunked(self, data: Union[np.ndarray, str]) -> bool:
        """Checks if the data is chunked or not.

        Parameters
        ----------
        data : Union[np.ndarray, str]
            The data to check.

        Returns
        -------
        bool
            Whether the data is chunked or not.
        """
        if isinstance(data, str):
            return len(data.split(",")) == 3
        return data.ndim == 3

    def plot(
        self, representation: str, nr_of_fingers: int = 5, visualize_wrist: bool = True
    ):
        """Plots the virtual hand kinematics data.

        Parameters
        ----------
        representation : str
            The representation to plot.
            The representation should be a 2D tensor with shape (9, n_samples)
            or a 3D tensor with shape (n_chunks, 9, n_samples).
        nr_of_fingers : int, optional
            The number of fingers to plot. Default is 5.
        visualize_wrist : bool, optional
            Whether to visualize wrist movements. Default is True.

        Raises
        ------
        KeyError
            If the representation does not exist.
        """
        if representation not in self._data:
            raise KeyError(f'The representation "{representation}" does not exist.')

        data = self[representation]
        is_chunked = self.is_chunked[representation]

        if is_chunked:
            # Use only the first chunk for visualization
            data = data[0]

        # Check if we have the expected number of DOFs
        if data.shape[0] != 9:
            raise ValueError(f"Expected 9 degrees of freedom, but got {data.shape[0]}")

        fig = plt.figure(figsize=(12, 8))

        # Create a separate plot for each DOF
        wrist_ax = fig.add_subplot(2, 1, 1)
        fingers_ax = fig.add_subplot(2, 1, 2)

        # Plot wrist DOFs (first 3 channels)
        if visualize_wrist:
            wrist_ax.set_title("Wrist Kinematics")
            wrist_ax.plot(data[0], label="Wrist Flexion/Extension")
            wrist_ax.plot(data[1], label="Wrist Pronation/Supination")
            wrist_ax.plot(data[2], label="Wrist Deviation")
            wrist_ax.legend()
            wrist_ax.set_xlabel("Time (samples)")
            wrist_ax.set_ylabel("Normalized Position")
            wrist_ax.grid(True)

        # Plot finger DOFs (remaining channels)
        fingers_ax.set_title("Finger Kinematics")
        finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        for i in range(min(nr_of_fingers, 5)):
            fingers_ax.plot(data[i + 3], label=finger_names[i])

        fingers_ax.legend()
        fingers_ax.set_xlabel("Time (samples)")
        fingers_ax.set_ylabel("Normalized Flexion")
        fingers_ax.grid(True)

        plt.tight_layout()
        plt.show()


DATA_TYPES_MAP = {
    "emg": EMGData,
    "kinematics": KinematicsData,
    "virtual_hand": VirtualHandKinematics,
}
