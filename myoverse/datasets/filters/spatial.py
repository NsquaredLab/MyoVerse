from typing import Any, Dict, List, Optional, Tuple, Literal

import numpy as np
from scipy.signal import convolve
import toml

from myoverse.datasets.filters._template import FilterBaseClass

# Dictionary below is used to define differential filters that can be applied across the monopolar electrode grids
_DIFFERENTIAL_FILTERS = {
    "identity": np.array([[1]]),  # identity case when no filtering is applied
    "LSD": np.array([[-1], [1]]),  # longitudinal single differential
    "LDD": np.array([[1], [-2], [1]]),  # longitudinal double differential
    "TSD": np.array([[-1, 1]]),  # transverse single differential
    "TDD": np.array([[1, -2, 1]]),  # transverse double differential
    "NDD": np.array(
        [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
    ),  # normal double differential or Laplacian filter
    "IB2": np.array(
        [[-1, -2, -1], [-2, 12, -2], [-1, -2, -1]]
    ),  # inverse binomial filter of order 2
    "IR": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),  # inverse rectangle
}


class ElectrodeSelector(FilterBaseClass):
    def __init__(
        self,
        electrodes_to_select: Optional[List[int]] = None,
        electrode_setup: Optional[Dict[str, Any]] = None,
        input_is_chunked: bool = None,
        is_output: bool = False,
        name: str = None,
    ):
        """Initialize the class.

        Parameters
        ----------
        electrodes_to_select : Optional[List[int]]
            List of electrodes to select from the original electrode setup. If None, then the electrode setup must be
            provided as a dictionary in electrode_setup.
        electrode_setup : Optional[Dict[str, Any]]
            Dictionary containing the electrode setup to be used. If None, then the electrode setup must be
            provided as a list of integers in electrodes_to_select.
        input_is_chunked : bool
            Whether the input is chunked or not.
        is_output : bool
            Whether the filter is an output filter.
        name : str
            Name of the filter.
        """
        super().__init__(
            input_is_chunked=input_is_chunked,
            allowed_input_type="both",
            is_output=is_output,
            name=name,
        )

        if electrode_setup is None:
            # check if electrodes_to_select is a list of integers
            if not all(isinstance(x, int) for x in electrodes_to_select):
                raise ValueError(
                    "The provided electrode setup is None and the electrodes_to_select is not a list of integers."
                )
            self.electrodes_to_select = electrodes_to_select
        else:
            self.electrode_setup = electrode_setup
            try:
                self.electrodes_to_select = np.concatenate(
                    [
                        np.arange(x[0], x[1])
                        for x in electrode_setup["grid"]["electrodes_to_select"]
                    ]
                )
            except KeyError:
                raise KeyError(
                    "The provided electrode setup dictionary does not contain the information about"
                    " the electrodes to select under the key 'electrodes_to_select' in the 'grid' key."
                )

    def _filter(self, input_array: np.ndarray) -> np.ndarray:
        """Select the electrodes from the input array."""
        return (
            input_array[:, :, self.electrodes_to_select]
            if self.input_is_chunked
            else input_array[:, self.electrodes_to_select]
        )


class GridReshaper(FilterBaseClass):
    def __init__(
        self,
        operation: Literal["c2g", "g2c", "concat"],
        shape: Optional[Tuple[int, int, int]] = None,
        grid_type: Optional[str] = None,
        electrode_setup: Optional[Dict[str, Any]] = None,
        input_is_chunked: bool = None,
        is_output: bool = False,
        name: str = None,
        **kwargs,
    ):
        """Initialize the class.

        Parameters
        ----------
        operation : Literal["c2g", "g2c", "concat"]
            Operation to be performed. Can be either "c2g" for channels to grid, "g2c" for grid to channels or
            "concat" for concatenating all grids together. If concatenation is performed, then the concatenated axis can
            be specified with the keyword argument "axis".
        shape : Optional[Tuple[int, int, int]]
            Shape of the grid to be reshaped. Tuple of (nr_grids, nr_rows, nr_col). If None, then the grid shape must
            be provided as a dictionary in electrode_setup.
        grid_type : Optional[str]
            Type of grid to be reshaped. Either 8x8 grid of 10mm IED or a 13x5 grid of 8mm IED.
            Can be either "GR10MM0808" or "GR08MM1305". If None, then the grid shape must be provided as a dictionary
            in electrode_setup.
        electrode_setup : Optional[Dict[str, Any]]
            Dictionary containing the electrode setup to be used. If None, then the grid shape must be provided as a tuple
            in shape and the grid type must be provided as a string in grid_type.
        input_is_chunked : bool
            Whether the input is chunked or not.
        is_output : bool
            Whether the filter is an output filter.
        name : str
            Name of the filter.
        kwargs : Any
            Additional keyword arguments to be passed to the specific operation.
        """
        super().__init__(
            input_is_chunked=input_is_chunked,
            allowed_input_type="both",
            is_output=is_output,
            name=name,
        )

        if operation not in ["c2g", "g2c", "concat"]:
            raise ValueError(
                "The provided operation is not supported. Please choose between 'c2g', 'g2c' or 'concat'."
            )
        self.operation = operation

        self.kwargs = kwargs

        if electrode_setup is None:
            if shape is None or grid_type is None:
                raise ValueError(
                    "The provided electrode setup is None and either the shape or the grid_type is None."
                )

            # check if shape is a tuple of integers
            if not all(isinstance(x, int) for x in shape):
                raise ValueError(
                    "The provided electrode setup is None and the shape is not a tuple of integers."
                )

            # check if grid_type is a string
            if not isinstance(grid_type, str):
                raise ValueError(
                    "The provided electrode setup is None and the grid_type is not a string."
                )

            self.nr_grids, self.nr_rows, self.nr_col = shape
            self.grid_type = grid_type
        else:
            self.electrode_setup = electrode_setup
            try:
                self.nr_grids, self.nr_rows, self.nr_col = electrode_setup["grid"][
                    "shape"
                ]
            except KeyError:
                raise KeyError(
                    "The provided electrode setup dictionary does not contain the information about"
                    " the grid shape under the key 'shape' in the 'grid' key."
                )
            try:
                self.grid_type = electrode_setup["grid"]["grid_type"]
            except KeyError:
                raise KeyError(
                    "The provided electrode setup dictionary does not contain the information about"
                    " the grid type under the key 'grid_type' in the 'grid' key."
                )

    def _filter(self, chunk: np.ndarray) -> np.ndarray:
        """Reshape the chunk based on the operation."""
        if self.operation == "c2g":
            return self._channels_to_grid(chunk)
        elif self.operation == "g2c":
            return self._grid_to_channels(chunk)
        elif self.operation == "concat":
            return self._grid_concatenation(chunk)

    def _channels_to_grid(self, chunk: np.ndarray) -> np.ndarray:
        """Reshape input chunk to grid shape.
        Use this function before any spatial filtering to avoid reshaping errors."""
        nr_filter_representations = chunk.shape[0]
        if self.grid_type == "GR10MM0808":
            if self.input_is_chunked:
                return np.stack(
                    [
                        chunk[:, chunk_idx]
                        .reshape((nr_filter_representations, self.nr_grids, 64, -1))
                        .reshape(
                            (nr_filter_representations, self.nr_grids, 8, 8, -1),
                            order="F",
                        )[:, :, ::-1]
                        for chunk_idx in range(chunk.shape[1])
                    ],
                    axis=1,
                )

            return chunk.reshape(
                (nr_filter_representations, self.nr_grids, 64, -1)
            ).reshape((nr_filter_representations, self.nr_grids, 8, 8, -1), order="F")[
                :, :, ::-1
            ]

        elif self.grid_type == "GR08MM1305":
            chunk = np.pad(
                chunk.reshape((nr_filter_representations, self.nr_grids, 64, -1)),
                ((0, 0), (0, 0), (1, 0), (0, 0)),
                "constant",
            ).reshape((nr_filter_representations, self.nr_grids, 13, 5, -1), order="F")
            chunk[:, :, :, [1, 3]] = np.flip(chunk[:, :, :, [1, 3]], axis=2)

            return chunk

        else:
            raise ValueError("This electrode grid is not defined.")

    def _grid_to_channels(self, chunk: np.ndarray) -> np.ndarray:
        """Reshape input chunk with the electrode grid shape back to channel x samples format. Use this function after
        applying spatial filters.
        """
        nr_filter_representations = chunk.shape[0]

        try:
            concatenated = self.kwargs["concatenate"]
        except KeyError:
            pass
        try:
            concatenated = self.electrode_setup["concatenate"]
        except KeyError:
            raise KeyError(
                "The provided electrode setup dictionary does not contain the information about"
                " the concatenation of the grids under the key 'concatenate'. You can also provide the"
                " information as a keyword argument."
            )

        total_nr_of_electrodes = chunk.shape[-3] * chunk.shape[-2]
        if concatenated:
            total_nr_of_electrodes *= self.nr_grids

        if self.grid_type == "GR10MM0808":
            chunk = chunk[:, :, :, ::-1] if self.input_is_chunked else chunk[:, :, ::-1]

            if self.input_is_chunked:
                return np.stack(
                    [
                        np.concatenate(
                            [
                                chunk[:, chunk_index, grid_index].reshape(
                                    (
                                        nr_filter_representations,
                                        total_nr_of_electrodes,
                                        -1,
                                    ),
                                    order="F",
                                )
                                for grid_index in range(chunk.shape[2])
                            ],
                            axis=1,
                        )
                        for chunk_index in range(chunk.shape[1])
                    ],
                    axis=1,
                )

            return np.concatenate(
                [
                    chunk[:, grid_index].reshape(
                        (nr_filter_representations, total_nr_of_electrodes, -1),
                        order="F",
                    )
                    for grid_index in range(chunk.shape[1])
                ],
                axis=1,
            )

        elif self.grid_type == "GR08MM1305":
            chunk[:, :, :, 1] = np.flip(chunk[:, :, :, 1], axis=2)

            if 5 - self.nr_col < 2:
                chunk[:, :, :, 3] = np.flip(chunk[:, :, :, 3], axis=2)

            orig_chunk = chunk[:, 0].reshape(
                (nr_filter_representations, total_nr_of_electrodes, -1), order="F"
            )

            for i in range(1, chunk.shape[1]):
                orig_chunk = np.concatenate(
                    (
                        orig_chunk,
                        chunk[:, i].reshape(
                            (nr_filter_representations, total_nr_of_electrodes, -1),
                            order="F",
                        ),
                    ),
                    axis=1,
                )

            return orig_chunk

        else:
            raise ValueError("This electrode grid is not defined.")

    def _grid_concatenation(
        self, chunk: np.ndarray, axis: Literal["row", "col"] = "col"
    ) -> np.ndarray:
        """Concatenate all electrode grids along specified axis. Function can be used to apply a spatial filter
        across multiple arrays, for e.g., in the direction of the arm circumference.
        """
        try:
            axis = self.kwargs["axis"]
        except KeyError:
            pass

        return np.concatenate(
            [chunk[:, i] for i in range(chunk.shape[1])],
            axis=-3 if axis == "row" else -2,
        )[:, None]


class DifferentialSpatialFilter(FilterBaseClass):
    def __init__(
        self, 
        filter_name: str, 
        input_is_chunked: bool = None,
        is_output: bool = False,
        name: str = None,
    ):
        """Initialize the class.

        Parameters
        ----------
        filter_name : str
            Name of the filter to be applied: "LSD", "TSD", "LDD", "TDD", "NDD", "IB2" or "IR". Filters are defined
            according to https://doi.org/10.1109/TBME.2003.808830. In case no filter is applied, use "identity".
        input_is_chunked : bool
            Whether the input is chunked or not.
        is_output : bool
            Whether the filter is an output filter.
        name : str
            Name of the filter.
        """
        super().__init__(
            input_is_chunked=input_is_chunked,
            allowed_input_type="both",
            is_output=is_output,
            name=name,
        )
        self.filter_name = filter_name

    def _filter(self, chunk: np.ndarray) -> np.ndarray:
        """This function applies the filters to the chunk."""
        return convolve(
            chunk,
            np.expand_dims(  # Extend filter dimensions prior to performing a convolution
                _DIFFERENTIAL_FILTERS[self.filter_name],
                axis=(0, 1, 2, -1) if self.input_is_chunked else (0, 1, -1),
            ),
            mode="valid",
        ).astype(np.float32)


class AveragingSpatialFilter(FilterBaseClass):
    def __init__(
        self, 
        order: int, 
        filter_direction: str,
        input_is_chunked: bool = None,
        is_output: bool = False,
        name: str = None,
    ):
        """Initialize the class.

        Parameters
        ----------
        order : int
            Order of the moving average filter.
        filter_direction : str
            Grid direction over which the filter is applied. Can be either "longitudinal" or "transverse".
        input_is_chunked : bool
            Whether the input is chunked or not.
        is_output : bool
            Whether the filter is an output filter.
        name : str
            Name of the filter.
        """
        super().__init__(
            input_is_chunked=input_is_chunked,
            allowed_input_type="both",
            is_output=is_output,
            name=name,
        )
        self.order = order
        self.filter_direction = filter_direction

    def _filter(self, chunk: np.ndarray) -> np.ndarray:
        """This function applies the moving average filter across the chunk."""
        if self.filter_direction == "longitudinal":
            flt_coeff = np.expand_dims(
                1
                / self.order
                * np.ones(self.order, dtype=int).reshape((self.order, -1)),
                axis=(0, 1, -1),
            )
        elif self.filter_direction == "transversal":
            flt_coeff = np.expand_dims(
                1
                / self.order
                * np.ones(self.order, dtype=int).reshape((-1, self.order)),
                axis=(0, 1, -1),
            )
        else:
            raise ValueError("Averaging direction name not correct.")

        # Extend filter dimensions prior to performing a convolution
        filtered_chunk = convolve(chunk, flt_coeff, mode="valid").astype(
            np.float32
        )

        return filtered_chunk


class ChannelSelector(FilterBaseClass):
    def __init__(
        self,
        grid_position: Optional[List[Tuple[int, int]]] = None,
        electrode_setup: Optional[Dict[str, Any]] = None,
        input_is_chunked: bool = None,
        is_output: bool = False,
        name: str = None,
    ):
        """Initialize the class.

        Parameters
        ----------
        grid_position : List[Tuple[int, int]]
            List of all grid electrode indexes based on row-column combination. If no channel selection is performed,
            set to None. If None, then the electrode setup must be provided as a dictionary in electrode_setup.
        electrode_setup : Optional[Dict[str, Any]]
            Dictionary containing the electrode setup to be used. If None, then the grid shape must be provided as a tuple
            in shape and the grid type must be provided as a string in grid_type.
        input_is_chunked : bool
            Whether the input is chunked or not.
        is_output : bool
            Whether the filter is an output filter.
        name : str
            Name of the filter.
        """
        super().__init__(
            input_is_chunked=input_is_chunked,
            allowed_input_type="both",
            is_output=is_output,
            name=name,
        )
        
        if electrode_setup is None:
            if grid_position is None:
                raise ValueError(
                    "The provided electrode setup is None and the grid_position is None."
                )

            # check if grid_position is a list of tuples
            if not all(isinstance(x, tuple) for x in grid_position):
                raise ValueError(
                    "The provided electrode setup is None and the grid_position is not a list of tuples."
                )

            self.grid_position = grid_position
        else:
            try:
                self.grid_position = electrode_setup["channel_selection"]
            except KeyError:
                raise KeyError(
                    "The provided electrode setup dictionary does not contain the information about"
                    " the channel selection under the key 'channel_selection'."
                )

    def _filter(self, chunk: np.ndarray) -> np.ndarray:
        """Select the channels from the input array."""
        if self.grid_position is None or self.grid_position == "all":
            return chunk
        else:
            selected_channel = []

            for index in self.grid_position:
                selected_channel.append(chunk[:, :, index[0], index[1]])

            chunk = np.array(selected_channel)
            chunk = chunk.reshape((1, -1, chunk.shape[-1]), order="F")

            return chunk


class BraceletDifferential(FilterBaseClass):
    def __init__(
        self, 
        input_is_chunked: bool = None, 
        is_output: bool = False,
        name: str = None,
    ):
        """Initialize the class.

        Parameters
        ----------
        input_is_chunked : bool
            Whether the input is chunked or not.
        is_output : bool
            Whether the filter is an output filter.
        name : str
            Name of the filter.
        """
        super().__init__(
            input_is_chunked=input_is_chunked,
            allowed_input_type="both", 
            is_output=is_output,
            name=name,
        )

    def _filter(self, chunk: np.ndarray) -> np.ndarray:
        """This function applies the filters to the chunk."""
        output = []
        if self.input_is_chunked:
            for representation in range(chunk.shape[0]):
                temp = []
                for chunk_index in range(chunk.shape[1]):
                    chunk_representation = chunk[representation, chunk_index].reshape(
                        2, 16, -1
                    )
                    # add circular padding to the chunk
                    chunk_representation = np.pad(
                        chunk_representation, ((0, 0), (1, 1), (0, 0)), "wrap"
                    )
                    # add zero padding to the chunk
                    chunk_representation = np.pad(
                        chunk_representation, ((1, 1), (0, 0), (0, 0)), "constant"
                    )

                    # Longitudinal differential
                    chunk_representation = convolve(
                        chunk_representation,
                        (np.array([[0, 1, 0], [1, 0.5, 1], [0, 1, 0]]) / 4)[..., None],
                        mode="valid",
                    )

                    temp.append(chunk_representation.reshape(32, -1))
                output.append(np.array(temp))
        else:
            for representation in range(chunk.shape[0]):
                chunk_representation = chunk[representation].reshape(2, 16, -1)
                # add circular padding to the chunk
                chunk_representation = np.pad(
                    chunk_representation, ((0, 0), (1, 1), (0, 0)), "wrap"
                )
                # add zero padding to the chunk
                chunk_representation = np.pad(
                    chunk_representation, ((1, 1), (0, 0), (0, 0)), "constant"
                )

                # Longitudinal differential
                chunk_representation = convolve(
                    chunk_representation,
                    (np.array([[0, 1, 0], [1, 0.5, 1], [0, 1, 0]]) / 4)[..., None],
                    mode="valid",
                )
                output.append(chunk_representation.reshape(32, -1))

        return np.array(output)


if __name__ == "__main__":
    # Test the classes out
    # All the non-grid reshaping functions below should be applied to chunks of shape grid x row x col x samples and
    # not to chunks of shape channels x samples
    ELECTRODE_SETUP = toml.load("electrode_setup.toml")

    emg_data = np.random.rand(1, 320, 192)
    test_emg = emg_data
    # emg_setup = "Thalmic-Myoarmband"
    emg_setup = "Quattrocento_forearm"

    print(
        "EMG dataset shape prior to any spatial filtering and reshaping:",
        emg_data.shape,
    )

    # Update the test code to use the new API
    electrode_selector = ElectrodeSelector(electrode_setup=ELECTRODE_SETUP[emg_setup])
    emg_data = electrode_selector(emg_data)
    
    grid_reshaper = GridReshaper(
        electrode_setup=ELECTRODE_SETUP[emg_setup], operation="c2g"
    )
    emg_data = grid_reshaper(emg_data)

    print(
        "EMG dataset shape after reshaping from 1D channels array to grid shape",
        emg_data.shape,
    )

    if ELECTRODE_SETUP[emg_setup]["concatenate"]:
        # Create a new grid reshaper for concatenation
        grid_concat_reshaper = GridReshaper(
            electrode_setup=ELECTRODE_SETUP[emg_setup], operation="concat"
        )
        emg_data = grid_concat_reshaper(emg_data)

    print("EMG dataset shape after concatenating all grids together", emg_data.shape)

    # Create a proper filter instance
    averaging_filter = AveragingSpatialFilter(
        **ELECTRODE_SETUP[emg_setup]["average"]
    )
    emg_data = averaging_filter(emg_data)

    print("EMG dataset shape after applying the averaging filter", emg_data.shape)

    differential_filter = DifferentialSpatialFilter(
        filter_name=ELECTRODE_SETUP[emg_setup]["differential"]
    )
    emg_data = differential_filter(emg_data)

    print("EMG dataset shape after applying the differential filter", emg_data.shape)

    if ELECTRODE_SETUP[emg_setup]["channel_selection"] != "all":
        channel_selector = ChannelSelector(electrode_setup=ELECTRODE_SETUP[emg_setup])
        emg_data = channel_selector(emg_data)
    else:
        # Create a new grid reshaper for g2c operation
        g2c_reshaper = GridReshaper(
            electrode_setup=ELECTRODE_SETUP[emg_setup], 
            operation="g2c",
            concatenate=ELECTRODE_SETUP[emg_setup]["concatenate"]
        )
        emg_data = g2c_reshaper(emg_data)

    print(
        "EMG dataset shape after selecting the needed channels and returning to 1D array shape",
        emg_data.shape,
    )
