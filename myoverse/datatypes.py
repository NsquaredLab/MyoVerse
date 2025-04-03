import copy
import inspect
import os
import pickle
from abc import abstractmethod
from typing import (
    Dict,
    Optional,
    TypedDict,
    Any,
    Union,
    List,
    Tuple,
    NamedTuple,
    Final,
)

import mplcursors
import networkx
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

from myoverse.datasets.filters._template import FilterBaseClass


class DeletedRepresentation(NamedTuple):
    """Class to hold metadata about deleted representations.

    This stores the shape and dtype of the deleted array. Making it compatible with the numpy array interface.

    Attributes
    ----------
    shape : tuple
        The shape of the deleted array
    dtype : np.dtype
        The data type of the deleted array
    """

    shape: tuple
    dtype: np.dtype

    def __str__(self) -> str:
        """String representation of the deleted data."""
        return str(self.shape)


Representation = TypedDict(
    "Representation",
    {"data": np.ndarray, "filter_sequence": List[FilterBaseClass]},
)

InputRepresentationName: Final[str] = "Input"
OutputRepresentationName: Final[str] = "Output"
LastRepresentationName: Final[str] = "Last"


def create_grid_layout(
    rows: int,
    cols: int,
    n_electrodes: int = None,
    fill_pattern: str = "row",
    missing_indices: List[Tuple[int, int]] = None,
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

    missing_positions = set(
        (r, c) for r, c in missing_indices if 0 <= r < rows and 0 <= c < cols
    )
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
    if fill_pattern.lower() == "row":
        for r in range(rows):
            for c in range(cols):
                if (r, c) not in missing_positions and electrode_idx < n_electrodes:
                    grid[r, c] = electrode_idx
                    electrode_idx += 1
    elif fill_pattern.lower() == "column":
        for c in range(cols):
            for r in range(rows):
                if (r, c) not in missing_positions and electrode_idx < n_electrodes:
                    grid[r, c] = electrode_idx
                    electrode_idx += 1
    else:
        raise ValueError(
            f"Invalid fill pattern: {fill_pattern}. Use 'row' or 'column'."
        )

    return grid


class _Data:
    """Base class for all data types.

    This class provides common functionality for handling different types of data,
    including maintaining original and processed representations, tracking filters
    applied, and managing data flow.

    Parameters
    ----------
    raw_data : np.ndarray
        The raw data to store.
    sampling_frequency : float
        The sampling frequency of the data.

    Attributes
    ----------
    sampling_frequency : float
        The sampling frequency of the data.
    _last_processing_step : str
        The last processing step applied to the data.
    _processed_representations : networkx.DiGraph
        The graph of the processed representations.
    _filters_used : Dict[str, FilterBaseClass]
        Dictionary of all filters used in the data. The keys are the names of the filters and the values are the filters themselves.
    _data : Dict[str, Union[np.ndarray, DeletedRepresentation]]
        Dictionary of all data. The keys are the names of the representations and the values are
        either numpy arrays or DeletedRepresentation objects (for representations that have been
        deleted to save memory but can be regenerated when needed).

    Raises
    ------
    ValueError
        If the sampling frequency is less than or equal to 0.

    Notes
    -----
    Memory Management:
        When representations are deleted with delete_data(), they are replaced with
        DeletedRepresentation objects that store essential metadata (shape, dtype)
        but don't consume memory for the actual data. These representations can be
        automatically recomputed when accessed. The chunking status is determined from
        the shape when needed.

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

    def __init__(
        self,
        raw_data: np.ndarray,
        sampling_frequency: float,
        nr_of_dimensions_when_unchunked: int,
    ):
        self.sampling_frequency: float = sampling_frequency

        self.nr_of_dimensions_when_unchunked: int = nr_of_dimensions_when_unchunked

        if self.sampling_frequency <= 0:
            raise ValueError("The sampling frequency should be greater than 0.")

        self._data: Dict[str, Union[np.ndarray, DeletedRepresentation]] = {
            InputRepresentationName: raw_data,
        }
        self._filters_used: Dict[str, FilterBaseClass] = {}

        self._processed_representations: networkx.DiGraph = networkx.DiGraph()
        self._processed_representations.add_node(InputRepresentationName)
        self._processed_representations.add_node(OutputRepresentationName)

        self.__last_processing_step: str = InputRepresentationName

    @property
    def is_chunked(self) -> Dict[str, bool]:
        """Returns whether the data is chunked or not.

        Returns
        -------
        Dict[str, bool]
            A dictionary where the keys are the representations and the values are whether the data is chunked or not.
        """
        # Create cache if it doesn't exist or if _data might have changed
        if not hasattr(self, "_chunked_cache") or len(self._chunked_cache) != len(
            self._data
        ):
            self._chunked_cache = {
                key: self._check_if_chunked(value) for key, value in self._data.items()
            }

        return self._chunked_cache

    def _check_if_chunked(self, data: Union[np.ndarray, DeletedRepresentation]) -> bool:
        """Checks if the data is chunked or not.

        Parameters
        ----------
        data : Union[np.ndarray, DeletedRepresentation]
            The data to check.

        Returns
        -------
        bool
            Whether the data is chunked or not.
        """
        return len(data.shape) == self.nr_of_dimensions_when_unchunked

    @property
    def input_data(self) -> np.ndarray:
        """Returns the input data."""
        return self._data[InputRepresentationName]

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
        # Convert to set for faster lookups
        output_nodes = set(
            self._processed_representations.predecessors(OutputRepresentationName)
        )
        return {key: value for key, value in self._data.items() if key in output_nodes}

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

    def plot_graph(self, title: Optional[str] = None):
        """Draws the graph of the processed representations.

        Parameters
        ----------
        title : Optional[str], default=None
            Optional title for the graph. If None, no title will be displayed.
        """
        # Use spectral layout but with enhancements for better flow
        G = self._processed_representations

        # Initial layout using spectral positioning
        pos = nx.spectral_layout(G)

        # Always position input node on the left and output node on the right
        min_x = min(p[0] for p in pos.values())
        max_x = max(p[0] for p in pos.values())

        # Normalize x positions to ensure full range is used
        for node in pos:
            pos[node][0] = (
                (pos[node][0] - min_x) / (max_x - min_x) if max_x != min_x else 0.5
            )

        # Force input/output node positions
        pos[InputRepresentationName][0] = 0.0  # Left edge
        pos[OutputRepresentationName][0] = 1.0  # Right edge

        # Use topological sort to improve node positioning
        try:
            # Get topologically sorted nodes (excluding input and output)
            topo_nodes = [
                node
                for node in nx.topological_sort(G)
                if node not in [InputRepresentationName, OutputRepresentationName]
            ]

            # Group nodes by their topological "level" (distance from input)
            node_levels = {}
            for node in topo_nodes:
                # Find all paths from input to this node
                paths = list(nx.all_simple_paths(G, InputRepresentationName, node))
                if paths:
                    # Level is the longest path length (minus 1 for the input node)
                    level = max(len(path) - 1 for path in paths)
                    if level not in node_levels:
                        node_levels[level] = []
                    node_levels[level].append(node)

            # Calculate the total number of levels
            max_level = max(node_levels.keys()) if node_levels else 0

            # Adjust x-positions based on level - without losing the original y-positions from spectral layout
            for level, nodes in node_levels.items():
                # Calculate new x-position (divide evenly between input and output)
                x_pos = level / (max_level + 1) if max_level > 0 else 0.5

                # Preserve the relative y-positions from spectral layout
                for node in nodes:
                    # Update only the x-position
                    pos[node][0] = x_pos
        except nx.NetworkXUnfeasible:
            # If topological sort fails, we'll keep the original spectral layout
            print("Warning: Topological sort failed, using default layout")
            pass
        except Exception as e:
            # Catch other exceptions
            print(f"Warning: Error in layout algorithm: {str(e)}")
            pass

        # Identify related nodes (nodes that share the same filter parent name)
        # This is particularly useful for filters that return multiple outputs
        related_nodes = {}
        for node in G.nodes():
            if node in [InputRepresentationName, OutputRepresentationName]:
                continue

            # Extract base filter name (part before the underscore)
            if "_" in node:
                base_name = node.split("_")[0]
                if base_name not in related_nodes:
                    related_nodes[base_name] = []
                related_nodes[base_name].append(node)

        # Adjust positions for related nodes to prevent overlap
        for base_name, nodes in related_nodes.items():
            if len(nodes) > 1:
                # Find average position for this group
                avg_x = sum(pos[node][0] for node in nodes) / len(nodes)

                # Calculate better vertical spacing
                vertical_spacing = 0.3 / len(nodes)

                # Arrange nodes vertically around their average x-position
                for i, node in enumerate(nodes):
                    # Keep the same x position but adjust y position
                    pos[node][0] = avg_x
                    # Distribute nodes vertically, centered around original y position
                    # Start from -0.15 to +0.15 to ensure good spacing
                    vertical_offset = -0.15 + (i * vertical_spacing)
                    pos[node][1] = pos[node][1] + vertical_offset

        # Apply gentle force-directed adjustments to improve layout
        # without completely changing the spectral positioning
        for _ in range(10):  # Reduced from 20 to 10 iterations
            # Store current positions
            old_pos = {n: p.copy() for n, p in pos.items()}

            for node in G.nodes():
                if node in [InputRepresentationName, OutputRepresentationName]:
                    continue  # Skip fixed nodes

                # Get node neighbors
                neighbors = list(G.neighbors(node))
                if not neighbors:
                    continue

                # Calculate average position of neighbors, weighted by in/out direction
                pred_force = np.zeros(2)
                succ_force = np.zeros(2)

                # Predecessors pull left
                predecessors = list(G.predecessors(node))
                if predecessors:
                    pred_force = (
                        np.mean([old_pos[p] for p in predecessors], axis=0)
                        - old_pos[node]
                    )
                    # Scale down x-force to maintain left-to-right flow
                    pred_force[0] *= 0.05  # Reduced from 0.1 to 0.05

                # Successors pull right
                successors = list(G.successors(node))
                if successors:
                    succ_force = (
                        np.mean([old_pos[s] for s in successors], axis=0)
                        - old_pos[node]
                    )
                    # Scale down x-force to maintain left-to-right flow
                    succ_force[0] *= 0.05  # Reduced from 0.1 to 0.05

                # Apply force (weighted more toward maintaining x position)
                force = pred_force + succ_force
                # Reduce force magnitude to avoid disrupting the topological ordering
                pos[node] += 0.05 * force  # Reduced from 0.1 to 0.05

                # Maintain x position within 0-1 range
                pos[node][0] = max(0.05, min(0.95, pos[node][0]))

        # Final overlap prevention - ensure minimum distance between nodes
        min_distance = 0.1  # Minimum distance between nodes
        for _ in range(3):  # Reduced from 5 to 3 iterations
            overlap_forces = {node: np.zeros(2) for node in G.nodes()}

            # Calculate repulsion forces between every pair of nodes
            node_list = list(G.nodes())
            for i, node1 in enumerate(node_list):
                for node2 in node_list[i + 1 :]:
                    # Skip input/output nodes
                    if node1 in [
                        InputRepresentationName,
                        OutputRepresentationName,
                    ] or node2 in [InputRepresentationName, OutputRepresentationName]:
                        continue

                    # Calculate distance between nodes
                    dist_vec = pos[node1] - pos[node2]
                    dist = np.linalg.norm(dist_vec)

                    # Apply repulsion if nodes are too close
                    if dist < min_distance and dist > 0:
                        # Normalize the vector
                        repulsion = dist_vec / dist
                        # Scale by how much they overlap
                        scale = (min_distance - dist) * 0.4  # Modified from 0.5 to 0.4
                        # Add to both nodes' forces (in opposite directions)
                        overlap_forces[node1] += repulsion * scale
                        overlap_forces[node2] -= repulsion * scale

            # Apply forces
            for node, force in overlap_forces.items():
                if node not in [InputRepresentationName, OutputRepresentationName]:
                    pos[node] += force
                    # Maintain x position closer to its original value
                    # to preserve the topological ordering
                    x_original = pos[node][0]
                    # Make sure nodes stay within bounds
                    pos[node][0] = max(0.05, min(0.95, pos[node][0]))
                    pos[node][1] = max(-0.95, min(0.95, pos[node][1]))
                    # Restore x position with a small adjustment
                    pos[node][0] = 0.9 * x_original + 0.1 * pos[node][0]

        # Create the figure and axis with a larger size for better visualization
        plt.figure(figsize=(16, 12))  # Increased from (14, 10)
        ax = plt.gca()

        # Add title if provided
        if title is not None:
            plt.title(title, fontsize=16, pad=20)

        # Create dictionaries for node attributes
        node_colors = {}
        node_sizes = {}
        node_shapes = {}

        # Set attributes based on node type
        for node in G.nodes():
            if node == InputRepresentationName:
                node_colors[node] = "crimson"
                node_sizes[node] = 1500
                node_shapes[node] = "o"  # Circle
            elif node == OutputRepresentationName:
                node_colors[node] = "forestgreen"
                node_sizes[node] = 1500
                node_shapes[node] = "o"  # Circle
            elif node not in self._data:
                # If the node is not in the data dictionary, it's a dummy node (like a filter name)
                node_colors[node] = "dimgray"
                node_sizes[node] = 1200
                node_shapes[node] = "o"  # Circle
            elif isinstance(self._data[node], DeletedRepresentation):
                node_colors[node] = "dimgray"
                node_sizes[node] = 1200
                node_shapes[node] = "o"  # Square for deleted representations
            else:
                node_colors[node] = "royalblue"
                node_sizes[node] = 1200
                node_shapes[node] = "o"  # Circle

        # Group nodes by shape for drawing
        node_groups = {}
        for shape in set(node_shapes.values()):
            node_groups[shape] = [node for node, s in node_shapes.items() if s == shape]

        # Draw each group of nodes with the correct shape
        drawn_nodes = {}
        for shape, nodes in node_groups.items():
            if not nodes:
                continue

            # Create lists of node properties
            node_list = nodes
            color_list = [node_colors[node] for node in node_list]
            size_list = [node_sizes[node] for node in node_list]

            # Draw nodes with the current shape
            if shape == "o":  # Circle
                drawn_nodes[shape] = nx.draw_networkx_nodes(
                    G,
                    pos,
                    nodelist=node_list,
                    node_color=color_list,
                    node_size=size_list,
                    alpha=0.8,
                    ax=ax,
                )
            elif shape == "s":  # Square
                drawn_nodes[shape] = nx.draw_networkx_nodes(
                    G,
                    pos,
                    nodelist=node_list,
                    node_color=color_list,
                    node_size=size_list,
                    node_shape="s",
                    alpha=0.8,
                    ax=ax,
                )

            # Set z-order for nodes
            if drawn_nodes[shape] is not None:
                drawn_nodes[shape].set_zorder(1)

        # Draw node labels with different colors based on node type
        label_objects = {}

        # Create custom labels: "I" for input, "O" for output, numbers for others starting from 1
        node_labels = {}
        # Filter out input and output nodes for separate labeling
        intermediate_nodes = [
            node
            for node in G.nodes
            if node not in [InputRepresentationName, OutputRepresentationName]
        ]

        # Add labels for input and output nodes
        node_labels[InputRepresentationName] = "I"
        node_labels[OutputRepresentationName] = "O"

        # For intermediate nodes, use sequential numbers (1 to n)
        for i, node in enumerate(intermediate_nodes, 1):
            node_labels[node] = str(i)

        label_objects["nodes"] = nx.draw_networkx_labels(
            G, pos, labels=node_labels, font_size=18, font_color="white", ax=ax
        )

        # Set z-order for all labels
        for label_group in label_objects.values():
            for text in label_group.values():
                text.set_zorder(3)

        # Remove the grid annotations since we're now showing the grid names directly in the nodes
        # Add additional text annotations if needed for extra information (not grid names)
        # This section is kept empty as we're now using the full representation names in the nodes

        # Create edge styles based on connection type
        edge_styles = []
        edge_colors = []
        edge_widths = []

        for u, v in G.edges():
            # Define edge properties based on connection type
            if u == InputRepresentationName:
                edge_colors.append("crimson")  # Input connections
                edge_widths.append(2.0)
                edge_styles.append("solid")
            elif v == OutputRepresentationName:
                edge_colors.append("forestgreen")  # Output connections
                edge_widths.append(2.0)
                edge_styles.append("solid")
            else:
                edge_colors.append("dimgray")  # Intermediate connections
                edge_widths.append(1.5)
                edge_styles.append("solid")

        # Draw all edges with the defined styles
        edges = nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edge_color=edge_colors,
            width=edge_widths,
            arrowstyle="-|>",
            arrowsize=20,
            connectionstyle="arc3,rad=0.2",  # Slightly increased curve for better visibility
            alpha=0.8,
        )

        # Set z-order for edges to be above nodes
        if isinstance(edges, list):
            for edge_collection in edges:
                edge_collection.set_zorder(2)
        elif edges is not None:
            edges.set_zorder(2)

        # Create annotation for hover information (initially invisible)
        annot = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(20, 20),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.9),
            fontsize=12,
            fontweight="normal",
            color="black",
            zorder=5,
        )
        annot.set_visible(False)

        # Add hover functionality for interactive exploration
        # Combine all node collections for the hover effect
        all_node_collections = [
            collection for collection in drawn_nodes.values() if collection is not None
        ]

        if all_node_collections:
            # Initialize the cursor without the hover behavior first
            cursor = mplcursors.cursor(all_node_collections, hover=True)

            # Map to keep track of the nodes for each collection
            node_collection_map = {}
            for shape, collection in drawn_nodes.items():
                if collection is not None:
                    node_collection_map[collection] = node_groups[shape]

            def on_hover(sel):
                try:
                    # Get the artist (the PathCollection) and find its shape
                    artist = sel.artist

                    # Get the target index - this is called 'target.index' in mplcursors
                    if hasattr(sel, "target") and hasattr(sel.target, "index"):
                        idx = sel.target.index
                    else:
                        # Fall back to other possible attribute names
                        idx = getattr(sel, "index", 0)

                    # Look up which nodes correspond to this artist
                    for shape, collection in drawn_nodes.items():
                        if collection == artist:
                            # Get list of nodes for this shape
                            shape_nodes = node_groups[shape]
                            if idx < len(shape_nodes):
                                hovered_node_name = shape_nodes[idx]

                                # Create the annotation text with full representation name
                                annotation = f"Representation: {hovered_node_name}\n\n"

                                # add whether the node needs to be recomputed
                                if (
                                    hovered_node_name != OutputRepresentationName
                                    and hovered_node_name in self._data
                                ):
                                    data = self._data[hovered_node_name]
                                    if isinstance(data, DeletedRepresentation):
                                        annotation += "needs to be\nrecomputed\n\n"

                                # add info whether the node is chunked or not
                                annotation += "chunked: "
                                if (
                                    hovered_node_name != OutputRepresentationName
                                    and hovered_node_name in self.is_chunked
                                ):
                                    annotation += str(
                                        self.is_chunked[hovered_node_name]
                                    )
                                else:
                                    annotation += "(see previous node(s))"

                                # add shape information to the annotation
                                annotation += "\n" + "shape: "
                                if (
                                    hovered_node_name != OutputRepresentationName
                                    and hovered_node_name in self._data
                                ):
                                    data = self._data[hovered_node_name]
                                    if isinstance(data, np.ndarray):
                                        annotation += str(data.shape)
                                    elif isinstance(data, DeletedRepresentation):
                                        annotation += str(data.shape)
                                else:
                                    annotation += "(see previous node(s))"

                                sel.annotation.set_text(annotation)
                                sel.annotation.get_bbox_patch().set(
                                    fc="white", alpha=0.9
                                )  # Background color
                                sel.annotation.set_fontsize(12)  # Font size
                                sel.annotation.set_fontstyle("italic")
                                break
                except Exception as e:
                    # If any error occurs, show a simplified annotation with detailed error info
                    error_info = f"Error in hover: {str(e)}\n"
                    if hasattr(sel, "target"):
                        error_info += f"Sel has target: {True}\n"
                        if hasattr(sel.target, "index"):
                            error_info += f"Target has index: {True}\n"
                    error_info += f"Available attributes: {dir(sel)}"
                    sel.annotation.set_text(error_info)

            cursor.connect("add", on_hover)

        # Improve visual appearance
        plt.grid(False)
        plt.axis("off")
        plt.margins(0.2)  # Increased from 0.15 to give more space around nodes
        plt.tight_layout(pad=2.0)  # Increased padding
        plt.show()

    def apply_filter(
        self,
        filter: FilterBaseClass,
        representations_to_filter: list[str] | None = None,
        keep_representation_to_filter: bool = True,
    ) -> str:
        """Applies a filter to the data.

        Parameters
        ----------
        filter : callable
            The filter to apply.
        representations_to_filter : list[str], optional
            A list of representations to filter. The filter is responsible for handling
            the appropriate number of inputs or raising an error if incompatible.
            If None, creates an empty list.
        keep_representation_to_filter : bool
            Whether to keep the representation(s) to filter or not.
            If the representation to filter is "Input", this parameter is ignored.

        Returns
        -------
        str
            The name of the representation after applying the filter.

        Raises
        ------
        ValueError
            If representations_to_filter is a string instead of a list
        TypeError
            If a filter returns a dictionary (no longer supported)
        """
        representation_name = filter.name

        # Ensure representations_to_filter is a list, not a string
        if isinstance(representations_to_filter, str):
            raise ValueError(
                f"representations_to_filter must be a list, not a string. "
                f"Use ['{representations_to_filter}'] instead of '{representations_to_filter}'."
            )

        # If representations_to_filter is None, create an empty list
        if representations_to_filter is None:
            representations_to_filter = []

        # Check if the list is empty
        if len(representations_to_filter) == 0:
            # For all filters, check if the list is empty
            raise ValueError(
                f"The filter {filter.name} requires an input representation. "
                f"Please provide at least one representation to filter."
            )

        # Replace LastRepresentationName with the actual last processing step
        representations_to_filter = [
            self._last_processing_step if rep == LastRepresentationName else rep
            for rep in representations_to_filter
        ]

        # Add edges to the graph for all input representations
        for rep in representations_to_filter:
            if rep not in self._processed_representations:
                self._processed_representations.add_node(rep)

            # Add filter node and create edges from inputs to filter
            if representation_name not in self._processed_representations:
                self._processed_representations.add_node(representation_name)
            # Add edge from the representation to filter to the new representation if it doesn't exist yet
            if not self._processed_representations.has_edge(rep, representation_name):
                self._processed_representations.add_edge(rep, representation_name)

        # Get the data for each representation
        input_arrays = [self[rep] for rep in representations_to_filter]

        # Automatically extract all data object parameters to pass to the filter
        data_params = {}
        # Use inspect to get all instance attributes
        for attr_name, attr_value in inspect.getmembers(self):
            # Skip private attributes, methods, and callables
            if (
                not attr_name.startswith("_")
                and not callable(attr_value)
                and not isinstance(attr_value, property)
            ):
                data_params[attr_name] = attr_value

        # Check if a standard filter is receiving multiple inputs inappropriately
        if len(input_arrays) > 1:
            raise ValueError(
                f"You're trying to pass multiple representations ({', '.join(representations_to_filter)}) to a "
                f"standard filter that only accepts a single input."
            )

        # If there's only one input, pass it directly; otherwise pass the list
        # This maintains backward compatibility with existing filters
        if len(input_arrays) == 1:
            filtered_data = filter(input_arrays[0], **data_params)
        else:
            filtered_data = filter(input_arrays, **data_params)

        # Store the filtered data
        self._data[representation_name] = filtered_data

        # Check if the filter is going to be an output
        # If so, add an edge from the representation to add to the output node
        if filter.is_output:
            self._processed_representations.add_edge(
                representation_name, OutputRepresentationName
            )

        # Save the used filter
        self._filters_used[representation_name] = filter

        # Set the last processing step
        self._last_processing_step = representation_name

        # Remove the representations to filter if needed
        if keep_representation_to_filter is False:
            for rep in representations_to_filter:
                if (
                    rep != InputRepresentationName
                ):  # Never delete the raw representation
                    self.delete_data(rep)

        return representation_name

    def apply_filter_sequence(
        self,
        filter_sequence: List[FilterBaseClass],
        representations_to_filter: List[str] | None = None,
        keep_individual_filter_steps: bool = True,
        keep_representation_to_filter: bool = True,
    ) -> str:
        """Applies a sequence of filters to the data sequentially.

        Parameters
        ----------
        filter_sequence : list[FilterBaseClass]
            The sequence of filters to apply.
        representations_to_filter : List[str], optional
            A list of representations to filter for the first filter in the sequence.
            Each filter is responsible for validating and handling its inputs appropriately.
            For subsequent filters in the sequence, the output of the previous filter is used.
        keep_individual_filter_steps : bool
            Whether to keep the results of each filter or not.
        keep_representation_to_filter : bool
            Whether to keep the representation(s) to filter or not.
            If the representation to filter is "Input", this parameter is ignored.

        Returns
        -------
        str
            The name of the last representation after applying all filters.

        Raises
        ------
        ValueError
            If filter_sequence is empty.
            If representations_to_filter is empty.
            If representations_to_filter is a string instead of a list.
        """
        if len(filter_sequence) == 0:
            raise ValueError("filter_sequence cannot be empty.")

        # Ensure representations_to_filter is a list, not a string
        if isinstance(representations_to_filter, str):
            raise ValueError(
                f"representations_to_filter must be a list, not a string. "
                f"Use ['{representations_to_filter}'] instead of '{representations_to_filter}'."
            )

        # If representations_to_filter is None, create an empty list
        if representations_to_filter is None:
            representations_to_filter = []

        # Replace LastRepresentationName with the actual last processing step
        representations_to_filter = [
            self._last_processing_step if rep == LastRepresentationName else rep
            for rep in representations_to_filter
        ]

        # Apply the first filter with the provided representations
        result = self.apply_filter(
            filter=filter_sequence[0],
            representations_to_filter=representations_to_filter,
            keep_representation_to_filter=True,  # We'll handle this at the end
        )

        # Collect intermediate results for potential cleanup later
        intermediate_results = [result]
        what_to_filter = [result]

        # Apply subsequent filters in sequence
        for i, f in enumerate(filter_sequence[1:], 1):
            # Apply the next filter using the previous result
            result = self.apply_filter(
                filter=f,
                representations_to_filter=what_to_filter,
                keep_representation_to_filter=True,  # Always keep intermediate results until the end
            )

            # Update what to filter for the next iteration
            intermediate_results.append(result)
            what_to_filter = [result]

        # Remove intermediate filter steps if needed, keeping the final result
        if not keep_individual_filter_steps:
            # Delete all intermediates except the final result
            for rep in intermediate_results[:-1]:  # Skip the last result
                self.delete_data(rep)

        # Remove the representation to filter if needed
        if not keep_representation_to_filter:
            for rep in representations_to_filter:
                if (
                    rep != InputRepresentationName
                ):  # Never delete the input representation
                    self.delete_data(rep)

        return result

    def apply_filter_pipeline(
        self,
        filter_pipeline: List[List[FilterBaseClass]],
        representations_to_filter: List[List[str]],
        keep_individual_filter_steps: bool = True,
        keep_representation_to_filter: bool = True,
    ) -> List[str]:
        """Applies a pipeline of filters to the data.

        Parameters
        ----------
        filter_pipeline : list[list[FilterBaseClass]]
            The pipeline of filters to apply. Each inner list represents a branch of filters.
        representations_to_filter : list[list[str]]
            A list of input representations for each branch. Each element corresponds to
            a branch in the filter_pipeline and must be:
            - A list with a single string for standard branches that take one input
            - A list with multiple strings for branches starting with a multi-input filter
            - An empty list is not allowed unless the filter explicitly accepts no input

            .. note :: The length of the representations_to_filter should be the same as
                      the length of the amount of branches in the filter_pipeline.
        keep_individual_filter_steps : bool
            Whether to keep the results of each filter or not.
        keep_representation_to_filter : bool
            Whether to keep the representation(s) to filter or not.
            If the representation to filter is "Input", this parameter is ignored.

        Returns
        -------
        List[str]
            A list containing the names of the final representations from all branches.

        Raises
        ------
        ValueError
            If the number of filter branches and representations to filter is different.
            If a standard filter is provided with multiple representations.
            If no representations are provided for a filter that requires input.
            If any representations_to_filter element is a string instead of a list.

        Notes
        -----
        Each branch in the pipeline is processed sequentially using apply_filter_sequence.

        Examples
        --------
        >>> # Example of a pipeline with multiple processing branches
        >>> from myoverse.datatypes import EMGData
        >>> from myoverse.datasets.filters.generic import ApplyFunctionFilter
        >>> import numpy as np
        >>>
        >>> # Create sample data
        >>> data = EMGData(np.random.rand(10, 8), sampling_frequency=1000)
        >>>
        >>> # Define filter branches that perform different operations on the same input
        >>> branch1 = [ApplyFunctionFilter(function=np.abs, name="absolute_values")]
        >>> branch2 = [ApplyFunctionFilter(function=lambda x: x**2, name="squared_values")]
        >>>
        >>> # Apply pipeline with two branches
        >>> data.apply_filter_pipeline(
        >>>     filter_pipeline=[branch1, branch2],
        >>>     representations_to_filter=[
        >>>         ["input_data"],  # Process branch1 on input_data
        >>>         ["input_data"],  # Process branch2 on input_data
        >>>     ],
        >>> )
        >>>
        >>> # The results are now available as separate representations
        >>> abs_values = data["absolute_values"]
        >>> squared_values = data["squared_values"]
        """
        if len(filter_pipeline) == 0:
            return []

        if len(filter_pipeline) != len(representations_to_filter):
            raise ValueError(
                f"The number of filter branches ({len(filter_pipeline)}) and "
                f"representations to filter ({len(representations_to_filter)}) must be the same."
            )

        # Ensure all elements in representations_to_filter are lists, not strings
        for branch_idx, branch_inputs in enumerate(representations_to_filter):
            if isinstance(branch_inputs, str):
                raise ValueError(
                    f"Element {branch_idx} of representations_to_filter is a string ('{branch_inputs}'), "
                    f"but must be a list. Use ['{branch_inputs}'] instead."
                )
            if branch_inputs is None:
                raise ValueError(
                    f"Element {branch_idx} of representations_to_filter is None, "
                    f"but must be a list. Use an empty list [] for filters that do not require input."
                )

            # Replace LastRepresentationName with the actual last processing step in each branch input
            representations_to_filter[branch_idx] = [
                self._last_processing_step if rep == LastRepresentationName else rep
                for rep in branch_inputs
            ]

        # Collect intermediates to delete after all branches are processed
        intermediates_to_delete = []
        all_results = []

        # Process each branch without deleting intermediates
        for branch_idx, (filter_sequence, branch_inputs) in enumerate(
            zip(filter_pipeline, representations_to_filter)
        ):
            try:
                # Apply filter sequence and get results
                branch_result = self.apply_filter_sequence(
                    filter_sequence=filter_sequence,
                    representations_to_filter=branch_inputs,
                    keep_individual_filter_steps=True,  # Always keep during processing
                    keep_representation_to_filter=keep_representation_to_filter,
                )

                # Track the branch result
                all_results.append(branch_result)

                # Track intermediates that might need to be deleted
                if not keep_individual_filter_steps:
                    # For each filter in the sequence (except the last),
                    # add its name to intermediates to delete
                    for f in filter_sequence[:-1]:
                        if hasattr(f, "name") and f.name:
                            intermediates_to_delete.append(f.name)

            except ValueError as e:
                # Enhance error message with branch information
                raise ValueError(
                    f"Error in branch {branch_idx + 1}/{len(filter_pipeline)}: {str(e)}"
                ) from e

        # After all branches are processed, delete collected intermediates if needed
        if not keep_individual_filter_steps:
            # First, identify all final outputs from the pipeline
            final_outputs = set(all_results)

            # For each representation in the data
            for rep_name in list(self._data.keys()):
                # Skip input and final outputs
                if rep_name == InputRepresentationName or rep_name in final_outputs:
                    continue

                # Check if this is an intermediate from any branch
                is_intermediate = False
                for base_name in intermediates_to_delete:
                    # Either exact match or prefix match for multi-output filters
                    if rep_name == base_name or rep_name.startswith(f"{base_name}_"):
                        is_intermediate = True
                        break

                if is_intermediate:
                    try:
                        self.delete_data(rep_name)
                    except KeyError:
                        # If already deleted or doesn't exist, just continue
                        pass

        return all_results

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
                InputRepresentationName,
                representation,
            )
        )

    def __repr__(self) -> str:
        # Get input data shape directly from _data dictionary to avoid copying
        input_shape = self._data[InputRepresentationName].shape

        # Build a structured string representation
        lines = []
        lines.append(f"{self.__class__.__name__}")
        lines.append(f"Sampling frequency: {self.sampling_frequency} Hz")
        lines.append(f"(0) Input {input_shape}")

        if len(self._processed_representations.nodes) >= 3:
            # Add an empty line for spacing between input and filters
            lines.append("")
            lines.append("Filter(s):")

            # Create mapping of representation to index only if needed
            if self._filters_used:
                representation_indices = {
                    key: index for index, key in enumerate(self._filters_used.keys())
                }

                # Precompute output predecessors for faster lookup
                output_predecessors = set(
                    self._processed_representations.predecessors(
                        OutputRepresentationName
                    )
                )

                for filter_index, (filter_name, filter_representation) in enumerate(
                    self._data.items()
                ):
                    if filter_name == InputRepresentationName:
                        continue

                    # Get history and format it more efficiently
                    history = self.get_representation_history(filter_name)
                    history_str = " -> ".join(
                        str(representation_indices[rep] + 1) for rep in history[1:]
                    )

                    # Build filter representation string
                    is_output = filter_name in output_predecessors
                    shape_str = (
                        filter_representation.shape
                        if not isinstance(filter_representation, str)
                        else filter_representation
                    )

                    filter_str = f"({filter_index} | {history_str}) "
                    if is_output:
                        filter_str += "(Output) "
                    filter_str += f"{filter_name} {shape_str}"

                    lines.append(filter_str)

        # Join all parts with newlines
        return "\n".join(lines)

    def __str__(self) -> str:
        return (
            "--\n"
            + self.__repr__()
            .replace("; ", "\n")
            .replace("Filter(s): ", "\nFilter(s):\n")
            + "\n--"
        )

    def __getitem__(self, key: str) -> np.ndarray:
        if key == InputRepresentationName:
            # Use array.view() for more efficient copying when possible
            data = self.input_data
            return data.view() if data.flags.writeable else data.copy()

        if key == LastRepresentationName:
            return self[self._last_processing_step]

        if key not in self._processed_representations:
            raise KeyError(f'The representation "{key}" does not exist.')

        data_to_return = self._data[key]

        if isinstance(data_to_return, DeletedRepresentation):
            print(f'Recomputing representation "{key}"')

            history = self.get_representation_history(key)
            self.apply_filter_sequence(
                filter_sequence=[
                    self._filters_used[filter_name] for filter_name in history[1:]
                ],
                representations_to_filter=[history[0]],
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

        This replaces the actual numpy array with a DeletedRepresentation object
        that contains metadata about the array, saving memory while allowing
        regeneration when needed.

        Parameters
        ----------
        representation_to_delete : str
            The representation to delete the data from.
        """
        if representation_to_delete == InputRepresentationName:
            return
        if representation_to_delete == LastRepresentationName:
            self.delete_data(self._last_processing_step)
            return

        if representation_to_delete not in self._data:
            raise KeyError(
                f'The representation "{representation_to_delete}" does not exist.'
            )

        data = self._data[representation_to_delete]
        if isinstance(data, np.ndarray):
            self._data[representation_to_delete] = DeletedRepresentation(
                shape=data.shape, dtype=data.dtype
            )

    def delete_history(self, representation_to_delete: str):
        """Delete the processing history for a representation.

        Parameters
        ----------
        representation_to_delete : str
            The representation to delete the history for.
        """
        if representation_to_delete == InputRepresentationName:
            return
        if representation_to_delete == LastRepresentationName:
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
        # Create a new instance with the basic initialization
        new_instance = self.__class__(
            self._data[InputRepresentationName].copy(), self.sampling_frequency
        )

        # Get all attributes of the current instance
        for name, value in inspect.getmembers(self):
            # Skip special methods, methods, and the already initialized attributes
            if (
                (
                    not name.startswith("_")
                    or name
                    in [
                        "_data",
                        "_processed_representations",
                        "_last_processing_step",
                        "_filters_used",
                    ]
                )
                and not inspect.ismethod(value)
                and not name == "sampling_frequency"
            ):
                # Handle different attribute types appropriately
                if name == "_data":
                    # Deep copy the data dictionary
                    setattr(new_instance, name, copy.deepcopy(value))
                elif name == "_processed_representations":
                    # Use the graph's copy method
                    setattr(new_instance, name, value.copy())
                elif name == "_filters_used":
                    # Deep copy the filters used
                    setattr(new_instance, name, copy.deepcopy(value))
                else:
                    # Shallow copy for other attributes
                    setattr(new_instance, name, copy.copy(value))

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
            elif isinstance(value, DeletedRepresentation):
                memory_usage[key] = (
                    str(value.shape),
                    0,  # DeletedRepresentation objects use negligible memory
                )

        return memory_usage


class EMGData(_Data):
    """Class for storing EMG data.

    Parameters
    ----------
    input_data : np.ndarray
        The raw EMG data. The shape of the array should be (n_channels, n_samples) or (n_chunks, n_channels, n_samples).#

        .. important:: The class will only accept 2D or 3D arrays.
        There is no way to check if you actually have it in (n_chunks, n_samples) or (n_chunks, n_channels, n_samples) format.
        Please make sure to provide the correct shape of the data.

    sampling_frequency : float
        The sampling frequency of the EMG data.
    grid_layouts : Optional[List[np.ndarray]], optional
        List of 2D arrays specifying the exact electrode arrangement for each grid.
        Each array element contains the electrode index (0-based).

        .. note:: All electrodes numbers must be unique and non-negative. The numbers must be contiguous (0 to n) spread over however many grids.

        Default is None.

    Attributes
    ----------
    input_data : np.ndarray
        The raw EMG data. The shape of the array should be (n_channels, n_samples) or (n_chunks, n_channels, n_samples).
    sampling_frequency : float
        The sampling frequency of the EMG data.
    grid_layouts : Optional[List[np.ndarray]]
        List of 2D arrays specifying the exact electrode arrangement for each grid.
        Each array element contains the electrode index (0-based).
    processed_data : Dict[str, np.ndarray]
        A dictionary where the keys are the names of filters applied
        to the EMG data and the values are the processed EMG data.

    Raises
    ------
    ValueError
        If the shape of the raw EMG data is not (n_channels, n_samples) or (n_chunks, n_channels, n_samples).
        If the grid layouts are not provided or are not valid.

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

    Working with Multiple Grid Layouts
    ---------------------------------

    Grid layouts enable precise specification of how electrodes are arranged physically.
    This is especially useful for visualizing and analyzing high-density EMG recordings
    with multiple electrode grids:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from myoverse.datatypes import EMGData, create_grid_layout
    >>>
    >>> # Create sample EMG data for 61 electrodes with 1000 samples each
    >>> emg_data = np.random.randn(61, 1000)
    >>> sampling_freq = 2048  # Hz
    >>>
    >>> # Create layouts for three different electrode grids
    >>> # First grid: 5×5 array with sequential numbering (0-24)
    >>> grid1 = create_grid_layout(5, 5, fill_pattern='row')
    >>>
    >>> # Second grid: 6×6 array with column-wise numbering
    >>> grid2 = create_grid_layout(6, 6, fill_pattern='column')
    >>> # Shift indices to start after the first grid (add 25)
    >>> grid2[grid2 >= 0] += 25
    >>>
    >>> # Third grid: Irregular 3×4 array
    >>> grid3 = create_grid_layout(3, 4, fill_pattern='row')
    >>> grid3[grid3 >= 0] += 50
    >>>
    >>> # Create EMGData with all three grids
    >>> emg = EMGData(emg_data, sampling_freq, grid_layouts=[grid1, grid2, grid3])
    >>>
    >>> # Visualize the three grid layouts
    >>> for i in range(3):
    ...     emg.plot_grid_layout(i)
    >>>
    >>> # Plot the raw EMG data using the grid arrangements
    >>> emg.plot('Input', scaling_factor=[15.0, 12.0, 20.0])
    >>>
    >>> # Access individual grid dimensions
    >>> grid_dimensions = emg._get_grid_dimensions()
    >>> for i, (rows, cols, electrodes) in enumerate(grid_dimensions):
    ...     print(f"Grid {i+1}: {rows}×{cols} with {electrodes} electrodes")
    """

    def __init__(
        self,
        input_data: np.ndarray,
        sampling_frequency: float,
        grid_layouts: Optional[List[np.ndarray]] = None,
    ):
        if input_data.ndim != 2 and input_data.ndim != 3:
            raise ValueError(
                "The shape of the raw EMG data should be (n_channels, n_samples) or (n_chunks, n_channels, n_samples)."
            )
        super().__init__(
            input_data, sampling_frequency, nr_of_dimensions_when_unchunked=3
        )

        self.grid_layouts = None  # Initialize to None first

        # Process and validate grid layouts if provided
        if grid_layouts is not None:
            # Transform to list if it is a numpy array
            if isinstance(grid_layouts, np.ndarray):
                grid_layouts = list(grid_layouts)

            for i, layout in enumerate(grid_layouts):
                if not isinstance(layout, np.ndarray) or layout.ndim != 2:
                    raise ValueError(f"Grid layout {i + 1} must be a 2D numpy array")

                # Check that not all elements are -1
                if np.all(layout == -1):
                    raise ValueError(
                        f"Grid layout {i + 1} contains all -1 values, indicating no electrodes!"
                    )

                # Check for duplicate electrode indices
                valid_indices = layout[layout >= 0]
                if len(np.unique(valid_indices)) != len(valid_indices):
                    raise ValueError(
                        f"Grid layout {i + 1} contains duplicate electrode indices"
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
                print(
                    f"Warning: nr_of_grids ({nr_of_grids}) does not match grid_layouts length "
                    f"({len(self.grid_layouts)}). Using grid_layouts."
                )

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

        fig = plt.figure(figsize=(5 * nr_of_grids, 6))

        # Calculate electrode index offset for each grid
        electrode_offsets = [0]
        for i in range(len(electrodes_per_grid) - 1):
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
                        data[0, data_idx]
                        + electrode_idx * data[0].mean() * scaling_factor[grid_idx]
                    )
                else:
                    ax.plot(
                        data[data_idx]
                        + electrode_idx * data.mean() * scaling_factor[grid_idx]
                    )

            ax.set_xlabel("Time (samples)")
            ax.set_ylabel("Electrode #")

            # Set the y-axis ticks to the electrode numbers beginning from 1
            mean_val = (
                data[0].mean() if self.is_chunked[representation] else data.mean()
            )
            ax.set_yticks(
                np.arange(0, n_electrodes) * mean_val * scaling_factor[grid_idx],
                np.arange(1, n_electrodes + 1),
            )

            # Only for grid 1 keep the y-axis label
            if grid_idx != 0:
                ax.set_ylabel("")

        plt.tight_layout()
        plt.show()

    def plot_grid_layout(
        self,
        grid_idx: int = 0,
        show_indices: bool = True,
        cmap: Optional[plt.cm.ScalarMappable] = None,
        figsize: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
        colorbar: bool = True,
        grid_color: str = "black",
        grid_alpha: float = 0.7,
        text_color: str = "white",
        text_fontsize: int = 10,
        text_fontweight: str = "bold",
        highlight_electrodes: Optional[List[int]] = None,
        highlight_color: str = "red",
        save_path: Optional[str] = None,
        dpi: int = 150,
        return_fig: bool = False,
        ax: Optional[plt.Axes] = None,
        autoshow: bool = True,
    ):
        """Plots the 2D layout of a specific electrode grid with enhanced visualization.

        Parameters
        ----------
        grid_idx : int, optional
            The index of the grid to plot. Default is 0.
        show_indices : bool, optional
            Whether to show the electrode indices in the plot. Default is True.
        cmap : Optional[plt.cm.ScalarMappable], optional
            Custom colormap to use for visualization. If None, a default viridis colormap is used.
        figsize : Optional[Tuple[float, float]], optional
            Custom figure size as (width, height) in inches. If None, size is calculated based on grid dimensions.
            Ignored if an existing axes object is provided.
        title : Optional[str], optional
            Custom title for the plot. If None, a default title showing grid dimensions is used.
        colorbar : bool, optional
            Whether to show a colorbar. Default is True.
        grid_color : str, optional
            Color of the grid lines. Default is "black".
        grid_alpha : float, optional
            Transparency of grid lines (0-1). Default is 0.7.
        text_color : str, optional
            Color of the electrode indices text. Default is "white".
        text_fontsize : int, optional
            Font size for electrode indices. Default is 10.
        text_fontweight : str, optional
            Font weight for electrode indices. Default is "bold".
        highlight_electrodes : Optional[List[int]], optional
            List of electrode indices to highlight. Default is None.
        highlight_color : str, optional
            Color to use for highlighting electrodes. Default is "red".
        save_path : Optional[str], optional
            Path to save the figure. If None, figure is not saved. Default is None.
        dpi : int, optional
            DPI for saved figure. Default is 150.
        return_fig : bool, optional
            Whether to return the figure and axes. Default is False.
        ax : Optional[plt.Axes], optional
            Existing axes object to plot on. If None, a new figure and axes will be created.
        autoshow : bool, optional
            Whether to automatically show the figure. Default is True.
            Set to False when plotting multiple grids on the same figure.

        Returns
        -------
        Optional[Tuple[plt.Figure, plt.Axes]]
            Figure and axes objects if return_fig is True.

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
        >>> # Basic visualization
        >>> emg.plot_grid_layout(0)
        >>>
        >>> # Advanced visualization
        >>> emg.plot_grid_layout(
        ...     0,
        ...     figsize=(10, 10),
        ...     colorbar=True,
        ...     highlight_electrodes=[10, 20, 30],
        ...     grid_alpha=0.5
        ... )
        >>>
        >>> # Multiple grids in one figure
        >>> import matplotlib.pyplot as plt
        >>> fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        >>> emg.plot_grid_layout(0, title="Grid 1", ax=ax1, autoshow=False)
        >>> emg.plot_grid_layout(1, title="Grid 2", ax=ax2, autoshow=False)
        >>> plt.tight_layout()
        >>> plt.show()
        """
        if self.grid_layouts is None:
            raise ValueError("Cannot plot grid layout: grid_layouts not provided.")

        if grid_idx < 0 or grid_idx >= len(self.grid_layouts):
            raise ValueError(
                f"Grid index {grid_idx} out of range (0 to {len(self.grid_layouts) - 1})."
            )

        # Get the grid layout
        grid = self.grid_layouts[grid_idx]
        rows, cols = grid.shape

        # Get number of electrodes
        n_electrodes = np.sum(grid >= 0)

        # Set default title if not provided
        if title is None:
            title = f"Grid {grid_idx + 1} layout ({rows}×{cols}) with {n_electrodes} electrodes"

        # Create a masked array for plotting
        masked_grid = np.ma.masked_less(grid, 0)

        # Create figure and axes if not provided
        if ax is None:
            # Calculate optimal figure size if not provided
            if figsize is None:
                # Scale based on grid dimensions with minimum size
                width = max(6, cols * 0.75 + 2)
                height = max(5, rows * 0.75 + 1)
                if colorbar:
                    width += 1  # Add space for colorbar
                figsize = (width, height)

            fig, ax = plt.subplots(figsize=figsize)
        else:
            # Get the figure object from the provided axes
            fig = ax.figure

        # Setup colormap
        if cmap is None:
            cmap = plt.cm.viridis
            cmap.set_bad("white", 1.0)

        # Create custom norm to ensure integer values are centered in color bands
        norm = plt.Normalize(vmin=-0.5, vmax=np.max(grid) + 0.5)

        # Plot the grid with improved visuals
        im = ax.imshow(masked_grid, cmap=cmap, norm=norm, interpolation="nearest")

        # Add colorbar if requested
        if colorbar:
            cbar = plt.colorbar(im, ax=ax, pad=0.01)
            cbar.set_label("Electrode Index")
            # Add tick labels only at integer positions
            cbar.set_ticks(np.arange(0, np.max(grid) + 1))

        # Improve grid lines
        # Major ticks at electrode centers
        ax.set_xticks(np.arange(0, cols, 1))
        ax.set_yticks(np.arange(0, rows, 1))
        # Minor ticks at grid boundaries
        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)

        # Hide major tick labels for cleaner look
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Apply grid styling
        ax.grid(
            which="minor",
            color=grid_color,
            linestyle="-",
            linewidth=1,
            alpha=grid_alpha,
        )
        ax.tick_params(which="minor", bottom=False, left=False)

        # Add axis labels
        ax.set_xlabel("Columns", fontsize=text_fontsize + 1)
        ax.set_ylabel("Rows", fontsize=text_fontsize + 1)

        # Add electrode numbers with improved styling
        if show_indices:
            for i in range(rows):
                for j in range(cols):
                    if grid[i, j] >= 0:
                        # Create a dictionary for text properties
                        text_props = {
                            "ha": "center",
                            "va": "center",
                            "color": text_color,
                            "fontsize": text_fontsize,
                            "fontweight": text_fontweight,
                        }

                        # Add highlight if this electrode is in highlight list
                        if highlight_electrodes and grid[i, j] in highlight_electrodes:
                            # Draw a circle around highlighted electrodes
                            circle = plt.Circle(
                                (j, i),
                                0.4,
                                fill=False,
                                edgecolor=highlight_color,
                                linewidth=2,
                                alpha=0.8,
                            )
                            ax.add_patch(circle)
                            # Change text properties for highlighted electrodes
                            text_props["fontweight"] = "extra bold"

                        # Add the electrode index text
                        ax.text(j, i, str(grid[i, j]), **text_props)

        # Add a title with improved styling
        ax.set_title(title, fontsize=text_fontsize + 4, pad=10)

        # Set aspect ratio to be equal
        ax.set_aspect("equal")

        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

        # Show the figure if autoshow is True
        if autoshow:
            plt.tight_layout()
            plt.show()

        # Return figure and axes if requested
        if return_fig:
            return fig, ax
        return None


class KinematicsData(_Data):
    """Class for storing kinematics data.

    Parameters
    ----------
    input_data : np.ndarray
        The raw kinematics data. The shape of the array should be (n_joints, 3, n_samples)
        or (n_chunks, n_joints, 3, n_samples).

        .. important:: The class will only accept 3D or 4D arrays.
        There is no way to check if you actually have it in (n_chunks, n_joints, 3, n_samples) format.
        Please make sure to provide the correct shape of the data.

    sampling_frequency : float
        The sampling frequency of the kinematics data.

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
        if input_data.ndim != 3 and input_data.ndim != 4:
            raise ValueError(
                "The shape of the raw kinematics data should be (n_joints, 3, n_samples) "
                "or (n_chunks, n_joints, 3, n_samples)."
            )
        super().__init__(
            input_data, sampling_frequency, nr_of_dimensions_when_unchunked=4
        )

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
    """Class for storing virtual hand kinematics data from MyoGestic [1]_.

    Parameters
    ----------
    input_data : np.ndarray
        The raw kinematics data for a virtual hand. The shape of the array should be (9, n_samples)
        or (n_chunks, 9, n_samples).

        .. important:: The class will only accept 2D or 3D arrays.
        There is no way to check if you actually have it in (n_chunks, n_samples) or (n_chunks, 9, n_samples) format.
        Please make sure to provide the correct shape of the data.

    sampling_frequency : float
        The sampling frequency of the kinematics data.

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

    Examples
    --------
    >>> import numpy as np
    >>> from myoverse.datatypes import VirtualHandKinematics
    >>>
    >>> # Create sample virtual hand kinematics data (9 DOFs, 1000 samples)
    >>> joint_data = np.random.randn(9, 1000)
    >>>
    >>> # Create a VirtualHandKinematics object with 100 Hz sampling rate
    >>> kinematics = VirtualHandKinematics(joint_data, 100)
    >>>
    >>> # Access the raw data
    >>> raw_data = kinematics.input_data
    >>> print(f"Data shape: {raw_data.shape}")

    References
    ----------
    .. [1] MyoGestic: https://github.com/NsquaredLab/MyoGestic
    """

    def __init__(self, input_data: np.ndarray, sampling_frequency: float):
        if input_data.ndim != 2 and input_data.ndim != 3:
            raise ValueError(
                "The shape of the raw kinematics data should be (9, n_samples) "
                "or (n_chunks, 9, n_samples)."
            )
        super().__init__(
            input_data, sampling_frequency, nr_of_dimensions_when_unchunked=3
        )

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
