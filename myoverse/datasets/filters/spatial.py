from abc import ABC, abstractmethod
import math
from typing import Any, Dict, List, Optional, Tuple, Literal, Union

import numpy as np
from scipy.signal import convolve

from myoverse.datasets.filters._template import FilterBaseClass


class SpatialFilterGridAware(FilterBaseClass, ABC):
    """Base class for spatial filters that need to be grid-aware.

    This class provides methods for handling multiple electrode grids in spatial filters.
    It allows applying filters to specific grids and optionally preserving unprocessed grids.

    Parameters
    ----------
    input_is_chunked : bool
        Whether the input is chunked or not.
    allowed_input_type : Literal["chunked", "non_chunked", "both"]
        Type of input this filter accepts.
    grids_to_process : Union[Literal["all"], List[int]]
        Specifies which grids to apply the filter to:

        - "all": Process all grids (default)
        - List[int]: Process only the grids with these indices
    is_output : bool, optional
        Whether the filter is an output filter.
    name : str, optional
        Name of the filter.
    run_checks : bool, optional
        Whether to run validation checks when filtering.
    """

    def __init__(
        self,
        input_is_chunked: bool = None,
        allowed_input_type: Literal["both", "chunked", "not chunked"] = "both",
        grids_to_process: Union[Literal["all"], List[int]] = "all",
        is_output: bool = False,
        name: str = None,
        run_checks: bool = True,
    ):
        super().__init__(
            input_is_chunked=input_is_chunked,
            allowed_input_type=allowed_input_type,
            is_output=is_output,
            name=name,
            run_checks=run_checks,
        )
        self.grids_to_process = grids_to_process

    def _get_grids_to_filter(self, grid_layouts) -> list[int]:
        # Determine which grids to process
        if self.grids_to_process == "all":
            output = list(range(len(grid_layouts)))
        elif isinstance(self.grids_to_process, int):
            output = [self.grids_to_process]
        elif isinstance(self.grids_to_process, list):
            output = self.grids_to_process
        else:
            raise ValueError(
                'grids_to_process should be either Literal["all"], int, or list[int]'
            )

        return output

    @abstractmethod
    def _filter(self, input_array: np.ndarray, **kwargs) -> np.ndarray:
        """Apply the filter to the input array.

        Parameters
        ----------
        input_array : np.ndarray
            The input array to filter.
        **kwargs
            Additional keyword arguments from the Data object.

            .. important:: "grid_layouts" must be passed in kwargs for this filter to work.

        Returns
        -------
        np.ndarray
            The filtered array.

        Raises
        ------
        AttributeError
            If the grid_layouts are not provided in kwargs. This filter only operates in grid-aware mode.

        Notes
        -----
        .. important:: Insure that the grid_layouts are updated using the new grid_layout. Use pass by reference stuff.

        .. code-block:: python
            :linenos:

            for i, new_grid_layout in enumerate(new_grid_layouts):
                grid_layouts[i] = new_grid_layout
        """
        raise NotImplementedError


class DifferentialSpatialFilter(SpatialFilterGridAware):
    """Differential spatial filter for EMG data.

    This filter applies various differential spatial filters to EMG data,
    which help improve signal quality by enhancing differences between adjacent electrodes.
    The filters are defined according to https://doi.org/10.1088/1741-2552/ad3498.

    Parameters
    ----------
    filter_name : Literal["LSD", "TSD", "LDD", "TDD", "NDD", "IB2", "IR", "identity"]
        Name of the filter to be applied. Options include:

        - "LSD": Longitudinal Single Differential - computes difference between adjacent electrodes along columns
        - "TSD": Transverse Single Differential - computes difference between adjacent electrodes along rows
        - "LDD": Longitudinal Double Differential - computes double difference along columns
        - "TDD": Transverse Double Differential - computes double difference along rows
        - "NDD": Normal Double Differential - combines information from electrodes in cross pattern
        - "IB2": Inverse Binomial filter of the 2nd order
        - "IR": Inverse Rectangle filter
        - "identity": No filtering, returns the original signal
    input_is_chunked : bool
        Whether the input data is organized in chunks (3D array) or not (2D array).
    grids_to_process : Union[Literal["all"], List[int]]
        Specifies which grids to apply the filter to:

        - "all": Process all grids (default)
        - List[int]: Process only the grids with these indices
    is_output : bool, default=False
        Whether the filter is an output filter.
    name : str, optional
        Custom name for the filter. If None, the class name will be used.
    run_checks : bool, default=True
        Whether to run validation checks when filtering.

    Notes
    -----
    This filter can work with both chunked and non-chunked EMG data, and can selectively
    process specific grids when multiple grids are present in the data.

    The convolution operation reduces the spatial dimensions based on the filter size,
    which means the output will have fewer electrodes than the input.

    Examples
    --------
    >>> import numpy as np
    >>> from myoverse.datatypes import EMGData
    >>> from myoverse.datasets.filters.spatial import DifferentialSpatialFilter
    >>>
    >>> # Create sample EMG data (64 channels, 1000 samples)
    >>> emg_data = np.random.randn(64, 1000)
    >>> emg = EMGData(emg_data, 2000)
    >>>
    >>> # Apply Laplacian filter to all grids
    >>> ndd_filter = DifferentialSpatialFilter(
    >>>     filter_name="NDD",
    >>>     input_is_chunked=False
    >>> )
    >>> filtered_data = emg.apply_filter(ndd_filter)
    >>>
    >>> # Apply Laplacian filter to only the first grid
    >>> ndd_first_grid = DifferentialSpatialFilter(
    >>>     filter_name="NDD",
    >>>     input_is_chunked=False,
    >>>     grids_to_process=0
    >>> )
    >>> filtered_first = emg.apply_filter(ndd_first_grid)
    """

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

    def __init__(
        self,
        input_is_chunked: bool,
        is_output: bool = False,
        name: str | None = None,
        run_checks: bool = True,
        *,
        filter_name: Literal[
            "LSD", "TSD", "LDD", "TDD", "NDD", "IB2", "IR", "identity"
        ],
        grids_to_process: Union[Literal["all"], int, List[int]] = "all",
    ):
        super().__init__(
            input_is_chunked=input_is_chunked,
            allowed_input_type="both",
            grids_to_process=grids_to_process,
            is_output=is_output,
            name=name,
            run_checks=run_checks,
        )
        self.filter_name = filter_name

        # Validate filter name
        if self.run_checks and filter_name not in (
            valid_filters := list(self._DIFFERENTIAL_FILTERS.keys())
        ):
            raise ValueError(
                f"Invalid filter_name: '{filter_name}'. Must be one of: {', '.join(valid_filters)}"
            )

    def _run_filter_checks(self, input_array: np.ndarray):
        """Additional validation for input data.

        Parameters
        ----------
        input_array : np.ndarray
            The input array to validate.
        """
        super()._run_filter_checks(input_array)

    def _filter(self, input_array: np.ndarray, **kwargs) -> np.ndarray:
        """Apply the selected differential spatial filter to the input array.

        Parameters
        ----------
        input_array : np.ndarray
            The input EMG data to filter.
        **kwargs
            Additional keyword arguments from the Data object, including:
            - grid_layouts: List of 2D arrays specifying electrode arrangements
            - sampling_frequency: The sampling frequency of the EMG data

        Returns
        -------
        np.ndarray
            The filtered EMG data, with dimensions depending on the filter size and
            convolution mode. The number of electrodes will typically be reduced.

        Raises
        ------
        AttributeError
            If the grid_layouts are not provided in kwargs.
        """
        # Get grid_layouts from kwargs
        try:
            grid_layouts = kwargs["grid_layouts"]
        except KeyError:
            raise AttributeError(
                "grid_layouts not found in kwargs. This filter only operates in grid-aware mode."
            )

        grids_to_filter = self._get_grids_to_filter(grid_layouts)

        outputs, new_grid_layouts = [], []
        for i, grid_layout in enumerate(grid_layouts):
            output = input_array[
                ...,
                grid_layout.shape[0] * grid_layout.shape[1] * i : grid_layout.shape[0]
                * grid_layout.shape[1]
                * (i + 1),
                :,
            ]

            new_grid_layout = grid_layout - np.min(grid_layout)

            if i in grids_to_filter:
                output, new_grid_layout = self._apply_differential_filter(
                    output,
                    new_grid_layout,
                )

            outputs.append(output)
            new_grid_layouts.append(
                new_grid_layout + (0 if i == 0 else np.max(new_grid_layouts[-1]) + 1)
            )

        # Have to do this so that values are updated in the original grid_layouts. Passing by reference stuff
        for i, new_grid_layout in enumerate(new_grid_layouts):
            grid_layouts[i] = new_grid_layout

        return np.concatenate(outputs, axis=-2)

    def _apply_differential_filter(self, grid_data, grid_layout):
        """Apply differential filter to a single grid's data.

        Parameters
        ----------
        grid_data : np.ndarray
            Data for a single grid to filter
        grid_layout : np.ndarray
            The grid layout. Shape is (n_rows, n_cols).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Filtered grid data and new grid layout
        """

        # Special case for identity filter
        if self.filter_name == "identity":
            return grid_data, grid_layout

        reshaped_grid_data = np.zeros(
            ((grid_data.shape[0],) if self.input_is_chunked else ())
            + (*grid_layout.shape, grid_data.shape[-1])
        )
        for i, j in np.ndindex(grid_layout.shape):
            index_to_select = grid_layout[i, j]
            reshaped_grid_data[..., i, j, :] = grid_data[..., index_to_select, :]

        differential_filter = self._DIFFERENTIAL_FILTERS[self.filter_name]

        new_grid_layout = np.lib.stride_tricks.sliding_window_view(
            grid_layout.copy(), differential_filter.shape
        ).min((-2, -1))
        mapping = {val: idx for idx, val in enumerate(np.unique(new_grid_layout))}
        new_grid_layout = np.vectorize(lambda x: mapping[x])(new_grid_layout)

        convolution_result = convolve(
            reshaped_grid_data,
            np.expand_dims(
                differential_filter,
                axis=(0, -1) if self.input_is_chunked else (-1,),
            ),
            mode="valid",
        ).astype(np.float32)

        output = np.zeros(
            ((convolution_result.shape[0],) if self.input_is_chunked else ())
            + (np.max(new_grid_layout) + 1, convolution_result.shape[-1]),
            dtype=convolution_result.dtype,
        )
        for i, j in np.ndindex(new_grid_layout.shape):
            output[..., new_grid_layout[i, j], :] = convolution_result[..., i, j, :]

        return output, new_grid_layout


class ApplyFunctionSpatialFilter(SpatialFilterGridAware):
    """Apply a function over the EMG grids using a user defined kernel.

    This filter applies a function over a user defined grid. The user can define the
    kernel and the function will be applied over the grid using the kernel.

    Parameters
    ----------
    kernel_size : tuple[int, int]
        The kernel size to use for the convolution. Must be a tuple of two integers.
    strides : tuple[int, int]
        The strides to use for the convolution. Must be a tuple of two integers.
    padding : str
        The padding to use for the convolution. Must be one of "same" or "valid".
    function : callable
        The function to apply over the grid. If input_is_chunked is True, the function must take and return a 4D array otherwise it must take and return a 3D array.

        .. note:: The input shape will be (chunks, time, y, x) if input_is_chunked is True and (time, y, x) if input_is_chunked is False.

        .. warning:: The function should only modify the y and x dimensions of the input array.

    input_is_chunked : bool
        Whether the input data is organized in chunks (3D array) or not (2D array).
    grids_to_process : Union[Literal["all"], List[int]]
        Specifies which grids to apply the filter to:

        - "all": Process all grids (default)
        - List[int]: Process only the grids with these indices

    is_output : bool, default=False
        Whether the filter is an output filter.
    name : str, optional
        Custom name for the filter. If None, the class name will be used.
    run_checks : bool, default=True
        Whether to run validation checks when filtering.

    Notes
    -----
    This filter can work with both chunked and non-chunked EMG data, and can selectively
    process specific grids when multiple grids are present in the data.

    .. important:: Because the filter can have strides the output will always be row-major. This also reflects in the new grid layout.
    """

    def __init__(
        self,
        input_is_chunked: bool,
        is_output: bool = False,
        name: str | None = None,
        run_checks: bool = True,
        *,
        kernel_size: tuple[int, int],
        function: callable,
        strides: tuple[int, int] = (1, 1),
        padding: str = "same",
        grids_to_process: Union[Literal["all"], List[int]] = "all",
    ):
        super().__init__(
            input_is_chunked=input_is_chunked,
            allowed_input_type="both",
            grids_to_process=grids_to_process,
            is_output=is_output,
            name=name,
            run_checks=run_checks,
        )

        if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
            raise ValueError("Kernel must be a tuple of two integers.")

        self.kernel_size = kernel_size

        if not callable(function):
            raise ValueError("Function must be a callable.")

        self.function = function

        if not isinstance(strides, tuple) or len(strides) != 2:
            raise ValueError("Strides must be a tuple of two integers.")

        self.strides = strides

        if padding not in ["same", "valid"]:
            raise ValueError("Padding must be 'same' or 'valid'.")

        self.padding = padding

    def _run_filter_checks(self, input_array: np.ndarray):
        """Additional validation for input data.

        Parameters
        ----------
        input_array : np.ndarray
            The input array to validate.
        """
        super()._run_filter_checks(input_array)

    def _filter(self, input_array: np.ndarray, **kwargs) -> np.ndarray:
        """Apply the selected differential spatial filter to the input array.

        Parameters
        ----------
        input_array : np.ndarray
            The input EMG data to filter.
        **kwargs
            Additional keyword arguments from the Data object, including:
            - grid_layouts: List of 2D arrays specifying electrode arrangements
            - sampling_frequency: The sampling frequency of the EMG data

        Returns
        -------
        np.ndarray
            The filtered EMG data, with dimensions depending on the filter size and
            convolution mode. The number of electrodes will typically be reduced.

        Raises
        ------
        AttributeError
            If the grid_layouts are not provided in kwargs.
        """
        # Get grid_layouts from kwargs
        try:
            grid_layouts = kwargs["grid_layouts"]
        except KeyError:
            raise AttributeError(
                "grid_layouts not found in kwargs. This filter only operates in grid-aware mode."
            )

        grids_to_filter = self._get_grids_to_filter(grid_layouts)

        outputs, new_grid_layouts = [], []
        for i, grid_layout in enumerate(grid_layouts):
            output = input_array[
                ...,
                grid_layout.shape[0] * grid_layout.shape[1] * i : grid_layout.shape[0]
                * grid_layout.shape[1]
                * (i + 1),
                :,
            ]

            new_grid_layout = grid_layout - np.min(grid_layout)

            if i in grids_to_filter:
                output, new_grid_layout = self._apply_custom_filter(
                    output,
                    new_grid_layout,
                )

            outputs.append(output)
            new_grid_layouts.append(
                new_grid_layout + (0 if i == 0 else np.max(new_grid_layouts[-1]) + 1)
            )

        # Have to do this so that values are updated in the original grid_layouts. Passing by reference stuff
        for i, new_grid_layout in enumerate(new_grid_layouts):
            grid_layouts[i] = new_grid_layout

        return np.concatenate(outputs, axis=-2)

    def _apply_custom_filter(self, grid_data, grid_layout):
        """Apply custom filter to a single grid's data.

        Parameters
        ----------
        grid_data : np.ndarray
            Data for a single grid to filter
        grid_layout : np.ndarray
            The grid layout. Shape is (n_rows, n_cols).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Filtered grid data and new grid layout
        """
        ky, kx = self.kernel_size
        sy, sx = self.strides

        reshaped_grid_data = np.zeros(
            ((grid_data.shape[0],) if self.input_is_chunked else ())
            + (*grid_layout.shape, grid_data.shape[-1])
        )
        for i, j in np.ndindex(grid_layout.shape):
            index_to_select = grid_layout[i, j]
            reshaped_grid_data[..., i, j, :] = grid_data[..., index_to_select, :]

        # Calculate padding for 'same' mode
        if self.padding == "same":
            # For y-axis
            input_y = reshaped_grid_data.shape[-3]
            output_y = math.ceil(input_y / sy)
            pad_total_y = max((output_y - 1) * sy + ky - input_y, 0)
            pad_y_before = pad_total_y // 2
            pad_y_after = pad_total_y - pad_y_before

            # For x-axis
            input_x = reshaped_grid_data.shape[-2]
            output_x = math.ceil(input_x / sx)
            pad_total_x = max((output_x - 1) * sx + kx - input_x, 0)
            pad_x_before = pad_total_x // 2
            pad_x_after = pad_total_x - pad_x_before

            grid_data_padded = np.pad(  # noqa
                reshaped_grid_data,
                (((0, 0),) if self.input_is_chunked else ())
                + ((pad_y_before, pad_y_after), (pad_x_before, pad_x_after), (0, 0)),
                mode="constant",
            )
        elif self.padding == "valid":
            grid_data_padded = grid_data
        else:
            raise ValueError("Padding must be 'same' or 'valid'")

        # Generate sliding windows and apply strides
        windows = np.lib.stride_tricks.sliding_window_view(  # noqa
            grid_data_padded, (ky, kx), axis=(-3, -2)
        )[..., ::sy, ::sx, :, :, :]

        # Apply the function to each window
        new_y, new_x = None, None
        new_windows = None

        for y, x in np.ndindex(windows.shape[-5:-3]):
            function_output = self.function(windows[..., y, x, :, :, :])
            if y == x == 0:
                new_y, new_x = function_output.shape[-2], function_output.shape[-1]
                new_windows = np.zeros(
                    ((windows.shape[0],) if self.input_is_chunked else ())
                    + (
                        new_y * windows.shape[-5],
                        new_x * windows.shape[-4],
                        windows.shape[-3],
                    ),
                    dtype=windows.dtype,
                )

            new_windows[
                ..., y * new_y : (y + 1) * new_y, x * new_y : (x + 1) * new_x, :
            ] = np.transpose(
                self.function(windows[..., y, x, :, :, :]),
                (2, 3, 0, 1) if self.input_is_chunked else (1, 2, 0),
            )

        y, x = new_windows.shape[-3:-1]
        new_grid_layout = np.arange(0, y * x).reshape((y, x), order="F")
        new_windows = np.reshape(
            new_windows,
            ((new_windows.shape[0],) if self.input_is_chunked else ())
            + (-1, new_windows.shape[-1]),
            order="F",
        )

        return new_windows, new_grid_layout
