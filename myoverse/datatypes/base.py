"""Base class for all data types."""

from __future__ import annotations

import copy
import inspect
import os
import pickle
from abc import abstractmethod
from typing import Any

import mplcursors
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from myoverse.datatypes.types import (
    DeletedRepresentation,
    InputRepresentationName,
    LastRepresentationName,
    OutputRepresentationName,
    Representation,
)


class _Data:
    """Base class for all data types.

    This class provides common functionality for handling different types of data,
    including maintaining original and processed representations.

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
    _processed_representations : nx.DiGraph
        The graph of the processed representations.
    _data : dict[str, np.ndarray | DeletedRepresentation]
        Dictionary of all data. The keys are the names of the representations and the values are
        either numpy arrays or DeletedRepresentation objects (for representations that have been
        deleted to save memory).

    Raises
    ------
    ValueError
        If the sampling frequency is less than or equal to 0.

    Notes
    -----
    Memory Management:
        When representations are deleted with delete_data(), they are replaced with
        DeletedRepresentation objects that store essential metadata (shape, dtype)
        but don't consume memory for the actual data. The chunking status is determined from
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

        self._data: dict[str, np.ndarray | DeletedRepresentation] = {
            InputRepresentationName: raw_data,
        }

        self._processed_representations: nx.DiGraph = nx.DiGraph()
        self._processed_representations.add_node(InputRepresentationName)
        self._processed_representations.add_node(OutputRepresentationName)

        self.__last_processing_step: str = InputRepresentationName

    @property
    def is_chunked(self) -> dict[str, bool]:
        """Returns whether the data is chunked or not.

        Returns
        -------
        dict[str, bool]
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

    def _check_if_chunked(self, data: np.ndarray | DeletedRepresentation) -> bool:
        """Checks if the data is chunked or not.

        Parameters
        ----------
        data : np.ndarray | DeletedRepresentation
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
    def processed_representations(self) -> dict[str, np.ndarray]:
        """Returns the processed representations of the data."""
        return self._data

    @processed_representations.setter
    def processed_representations(self, value: dict[str, Representation]):
        raise RuntimeError("This property is read-only.")

    @property
    def output_representations(self) -> dict[str, np.ndarray]:
        """Returns the output representations of the data."""
        # Convert to set for faster lookups
        output_nodes = set(
            self._processed_representations.predecessors(OutputRepresentationName)
        )
        return {key: value for key, value in self._data.items() if key in output_nodes}

    @output_representations.setter
    def output_representations(self, value: dict[str, Representation]):
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

    def plot_graph(self, title: str | None = None):
        """Draws the graph of the processed representations.

        Parameters
        ----------
        title : str | None, default=None
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
        except Exception as e:
            # Catch other exceptions
            print(f"Warning: Error in layout algorithm: {e}")

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
        edge_colors = []
        edge_widths = []

        for u, v in G.edges():
            # Define edge properties based on connection type
            if u == InputRepresentationName:
                edge_colors.append("crimson")  # Input connections
                edge_widths.append(2.0)
            elif v == OutputRepresentationName:
                edge_colors.append("forestgreen")  # Output connections
                edge_widths.append(2.0)
            else:
                edge_colors.append("dimgray")  # Intermediate connections
                edge_widths.append(1.5)

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

    def get_representation_history(self, representation: str) -> list[str]:
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

        # Show other representations if they exist
        other_reps = [k for k in self._data.keys() if k != InputRepresentationName]
        if other_reps:
            lines.append("")
            lines.append("Representations:")

            # Precompute output predecessors for faster lookup
            output_predecessors = set(
                self._processed_representations.predecessors(OutputRepresentationName)
            )

            for idx, rep_name in enumerate(other_reps, 1):
                rep_data = self._data[rep_name]
                is_output = rep_name in output_predecessors
                shape_str = (
                    rep_data.shape
                    if not isinstance(rep_data, str)
                    else rep_data
                )

                rep_str = f"({idx}) "
                if is_output:
                    rep_str += "(Output) "
                rep_str += f"{rep_name} {shape_str}"
                lines.append(rep_str)

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

        if key not in self._data:
            raise KeyError(f'The representation "{key}" does not exist.')

        data_to_return = self._data[key]

        if isinstance(data_to_return, DeletedRepresentation):
            raise RuntimeError(
                f'The representation "{key}" was deleted and cannot be automatically '
                f'recomputed. Use the new Transform API for preprocessing.'
            )

        # Use view when possible for more efficient memory usage
        return data_to_return.view() if data_to_return.flags.writeable else data_to_return.copy()

    def __setitem__(self, key: str, value: np.ndarray) -> None:
        raise RuntimeError(
            "Direct assignment is not supported. Use the Transform API for preprocessing."
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

    def memory_usage(self) -> dict[str, tuple[str, int]]:
        """Calculate memory usage of each representation.

        Returns
        -------
        dict[str, tuple[str, int]]
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
