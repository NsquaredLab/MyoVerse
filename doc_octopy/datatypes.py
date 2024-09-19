import copy
from abc import abstractmethod
from typing import Dict, Optional, Sequence, TypedDict, Any

import mplcursors
import networkx
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from numpy import ndarray

from doc_octopy.datasets.filters._template import FilterBaseClass

Representation = TypedDict(
    "Representation",
    {"data": np.ndarray, "filter_sequence": list[FilterBaseClass]},
)


class _Data:
    def __init__(self, raw_data: np.ndarray, sampling_frequency: float):
        self.sampling_frequency = sampling_frequency

        if self.sampling_frequency <= 0:
            raise ValueError("The sampling frequency should be greater than 0.")

        self.__input_representation_name = "Input"
        self.__output_representation_name = "Output"
        self.__last_representation_name = "Last"

        self._data: dict[str, np.ndarray | str] = {
            self.__input_representation_name: raw_data,
        }
        self._filters_used: dict[str, FilterBaseClass] = {}

        self._processed_representations = networkx.DiGraph()
        self._processed_representations.add_node(self.__input_representation_name)
        self._processed_representations.add_node(self.__output_representation_name)

        self.__last_processing_step = self.__input_representation_name

    @property
    def is_chunked(self) -> dict[str, bool]:
        """Returns whether the data is chunked or not.

        Returns
        -------
        dict[str, bool]
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
    def _check_if_chunked(self, data: np.ndarray) -> bool:
        """Checks if the data is chunked or not.

        Parameters
        ----------
        data : np.ndarray
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
    def processed_representations(self) -> dict[str, ndarray]:
        """Returns the processed representations of the data."""
        return self._data

    @processed_representations.setter
    def processed_representations(self, value: dict[str, Representation]):
        raise RuntimeError("This property is read-only.")

    @property
    def output_representations(self) -> dict[str, ndarray]:
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
        filter_sequence: list[FilterBaseClass],
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
        filter_pipeline: list[list[FilterBaseClass]],
        representations_to_filter: list[str],
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
            return copy.copy(self.input_data)
        if key == self.__last_representation_name:
            return copy.copy(self[self._last_processing_step])

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

        return copy.copy(self._data[key])

    def __setitem__(self, key: str, value: np.ndarray) -> None:
        raise RuntimeError(
            "This method is not supported. Run apply_filter or apply_filters instead."
        )

    def delete_data(self, representation_to_delete: str):
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
        self.delete_data(representation_to_delete)
        self.delete_history(representation_to_delete)

    def __copy__(self) -> "_Data":
        new_instance = self.__class__(
            self._data[self.__input_representation_name].copy(), self.sampling_frequency
        )
        new_instance._processed_representations = copy.deepcopy(
            self._processed_representations
        )
        new_instance._data = copy.deepcopy(self._data)
        new_instance._last_processing_step = self._last_processing_step

        return new_instance


class EMGData(_Data):
    """Class for storing EMG data.

    Attributes
    ----------
    input_data : np.ndarray
        The raw EMG data. The shape of the array should be (n_channels, n_samples) or (n_chunks, n_channels, n_samples).
    sampling_frequency : float
        The sampling frequency of the EMG data.
    electrode_config : Optional[Sequence[str]]
        The configuration of the electrodes.
    processed_data : Dict[str, np.ndarray]
        A dictionary where the keys are the names of filters applied
        to the EMG data and the values are the processed EMG data.

    Parameters
    ----------
    is_chunked : bool
        Whether the EMG data is chunked or not.
        If True, the shape of the raw EMG data should be (n_chunks, n_channels, n_samples).

    """

    def __init__(
        self,
        input_data: np.ndarray,
        sampling_frequency: float,
        electrode_config: Optional[Sequence[str]] = None,
    ):
        """Initializes the EMGData object.

        Parameters
        ----------
        input_data : np.ndarray
            The raw EMG data. The shape of the array should be (n_channels, n_samples)
             or (n_chunks, n_channels, n_samples).
        sampling_frequency : float
            The sampling frequency of the EMG data.
        """
        if input_data.ndim != 2 and input_data.ndim != 3:
            raise ValueError(
                "The shape of the raw EMG data should be (n_channels, n_samples) or (n_chunks, n_channels, n_samples)."
            )
        super().__init__(input_data, sampling_frequency)

        self.electrode_config = electrode_config

    def _check_if_chunked(self, data: np.ndarray | str) -> bool:
        """Checks if the data is chunked or not.

        Parameters
        ----------
        data : np.ndarray | str
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
        nr_of_grids: int,
        nr_of_electrodes_per_grid: int,
        scaling_factor: float | list[float] = 20.0,
    ):
        """Plots the data for a specific representation.

        Parameters
        ----------
        representation : str
            The representation to plot.
        nr_of_grids : int
            The number of electrode grids to plot.
        nr_of_electrodes_per_grid : int
            The number of electrodes per grid to plot.
        scaling_factor : float | list[float], optional
            The scaling factor for the data. The default is 20.0.
            If a list is provided, the scaling factor for each grid is used.
        """
        data = self[representation]

        if isinstance(scaling_factor, float):
            scaling_factor = [scaling_factor] * nr_of_grids

        assert (
            len(scaling_factor) == nr_of_grids
        ), "The number of scaling factors should be equal to the number of grids."

        fig = plt.figure()
        # make for each grid a subplot
        for grid in range(nr_of_grids):
            ax = fig.add_subplot(1, nr_of_grids, grid + 1)
            ax.set_title(f"Grid {grid + 1}")

            for electrode in range(nr_of_electrodes_per_grid):
                ax.plot(
                    data[grid * nr_of_electrodes_per_grid + electrode]
                    + electrode * data.mean() * scaling_factor[grid]
                )

            ax.set_xlabel("Time (samples)")
            ax.set_ylabel("Electrode #")

            # set the y-axis ticks to the electrode numbers begginning from 1
            ax.set_yticks(
                np.arange(0, nr_of_electrodes_per_grid)
                * data.mean()
                * scaling_factor[grid],
                np.arange(1, nr_of_electrodes_per_grid + 1),
            )

            # only for grid 1 keep the y-axis label
            if grid != 0:
                ax.set_ylabel("")

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
        """
        if input_data.ndim != 3 and input_data.ndim != 4:
            raise ValueError(
                "The shape of the raw kinematics data should be (n_joints, 3, n_samples) "
                "or (n_chunks, n_joints, 3, n_samples)."
            )
        super().__init__(input_data, sampling_frequency)

    def _check_if_chunked(self, data: np.ndarray | str) -> bool:
        """Checks if the data is chunked or not.

        Parameters
        ----------
        data : np.ndarray | str
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

        sample_slider = Slider(
            ax=fig.add_axes([0.25, 0.1, 0.65, 0.03]),
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
                    kinematics[
                        [0] + list(reversed(range(1 + finger * 4, 5 + finger * 4))),
                        :,
                        int(val),
                    ].T
                )

            fig.canvas.draw_idle()

        sample_slider.on_changed(update)

        plt.show()


class VirtualHandKinematics(_Data):
    def __init__(self, input_data: np.ndarray, sampling_frequency: float):
        """Initializes the VirtualHandKinematics object.

        Parameters
        ----------
        input_data : np.ndarray
            The raw kinematics data. The shape of the array should be (n_joints, 3, n_samples)
            or (n_chunks, n_joints, 3, n_samples).
            The 3 represents the x, y, and z coordinates of the joints.
        sampling_frequency : float
            The sampling frequency of the kinematics data.
        """
        if input_data.ndim != 2 and input_data.ndim != 3:
            raise ValueError(
                "The shape of the raw kinematics data should be (9, n_samples) "
                "or (n_chunks, 9, n_samples)."
            )
        super().__init__(input_data, sampling_frequency)

    def _check_if_chunked(self, data: np.ndarray | str) -> bool:
        """Checks if the data is chunked or not.

        Parameters
        ----------
        data : np.ndarray | str
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
        self, representation: str, nr_of_fingers: int, wrist_included: bool = True
    ):
        """Plots the data.

        raise NotImplementedError("This method is not implemented yet.")
        """
        # TODO: Implement this method
        raise NotImplementedError("This method is not implemented yet.")



DATA_TYPES_MAP = {"emg": EMGData, "kinematics": KinematicsData, "virtual_hand": VirtualHandKinematics}
