from typing import Literal
from abc import abstractmethod
import inspect

import numpy as np


class FilterBaseClass:
    """Base class for filters.

    Parameters
    ----------
    input_is_chunked : bool
        Whether the input is chunked or not.
    allowed_input_type : Literal["both", "chunked", "not chunked"]
        Whether the filter accepts chunked input, not chunked input or both.
    is_output : bool
        Whether the filter is an output filter. If True, the resulting signal will be outputted by and dataset pipeline.
    name : str, optional
        The name of the filter. This is used to identify the filter in the dataset. If not provided, the name of the filter class will be used.
    run_checks : bool
        Whether to run the checks when filtering. By default, True. If False can potentially speed up performance.

        .. warning:: If False, the user is responsible for ensuring that the input array is valid.

    Methods
    -------
    __call__(input_array: np.ndarray | list[np.ndarray]) -> np.ndarray
        Filters the input array(s).

        .. note:: A filter can accept a single numpy array OR a list of numpy arrays. Not both.

        Input shape is determined by whether the allowed_input_type is "both", "chunked" or "not chunked".

    Raises
    ------
    ValueError
        If input_is_chunked is not explicitly set.
        If allowed_input_type is not valid.
    """

    def __init__(
        self,
        input_is_chunked: bool,
        allowed_input_type: Literal["both", "chunked", "not chunked"],
        is_output: bool = False,
        name: str | None = None,
        run_checks: bool = True,
    ):
        self.input_is_chunked = input_is_chunked
        self._allowed_input_type = allowed_input_type

        self.is_output = is_output
        self._name = name

        self.run_checks = run_checks

        if self.run_checks:
            self._filter_function_to_run = self._filter_with_checks
            self._run_init_checks()
        else:
            self._filter_function_to_run = self._filter

    @property
    def name(self):
        """Return the filter name, using the class name if no custom name was provided."""
        if self._name is None:
            self._name = self.__class__.__name__
        return self._name

    def _run_filter_checks(self, input_array: np.ndarray | list[np.ndarray]):
        """Run checks on the input array.

        This method is called by the __call__ method before the filter is applied.
        It can be overridden by subclasses to add additional checks.
        """
        pass

    def _run_init_checks(self):
        """Run checks on the filter initialization parameters."""
        # Input type validation
        # Validate allowed_input_type
        valid_types = ["both", "chunked", "not chunked"]

        if self._allowed_input_type is None:
            raise ValueError(
                f"allowed_input_type must be specified. Valid values are: {', '.join(valid_types)}"
            )

        if self._allowed_input_type not in valid_types:
            raise ValueError(
                f"allowed_input_type must be one of: {', '.join(valid_types)}. Got '{self._allowed_input_type}' instead."
            )

        # Require input_is_chunked to be explicitly set
        if self.input_is_chunked is None:
            raise ValueError(
                f"input_is_chunked must be explicitly set for {self.__class__.__name__}. "
                f"The user should specify whether their data is chunked or not."
            )

        # Chunked vs non-chunked validation
        if self._allowed_input_type == "both":
            # Filter accepts both chunked and non-chunked input
            return
        elif self._allowed_input_type == "chunked":
            if not self.input_is_chunked:
                raise ValueError(
                    f"This filter ({self.name}) only accepts chunked input, but input_is_chunked=False was provided."
                )
        elif self._allowed_input_type == "not chunked":
            if self.input_is_chunked:
                raise ValueError(
                    f"This filter ({self.name}) only accepts non-chunked input, but input_is_chunked=True was provided."
                )

    def _filter_with_checks(
        self, input_array: np.ndarray | list[np.ndarray]
    ) -> np.ndarray:
        self._run_filter_checks(input_array)
        return self._filter(input_array)

    def __call__(self, input_array: np.ndarray | list[np.ndarray]) -> np.ndarray:
        return self._filter_function_to_run(input_array)

    @abstractmethod
    def _filter(self, input_array: np.ndarray | list[np.ndarray]) -> np.ndarray:
        raise NotImplementedError("This method must be implemented in the subclass.")

    def __repr__(self):
        cls_name = self.__class__.__name__

        # Use name if provided, otherwise use class name
        display_name = self._name if self._name is not None else cls_name

        # Use inspection to get parameters
        params = []

        # Get the initialization signature
        init_signature = inspect.signature(self.__class__.__init__)

        # Skip 'self' parameter
        for param_name in list(init_signature.parameters.keys())[1:]:
            # Skip 'kwargs' if present
            if param_name in ("args", "kwargs"):
                continue

            # Check if the parameter exists in self
            if hasattr(self, param_name) or hasattr(self, f"_{param_name}"):
                attr_name = (
                    param_name if hasattr(self, param_name) else f"_{param_name}"
                )
                value = getattr(self, attr_name)

                # Format the value
                if isinstance(value, str):
                    formatted_value = f"'{value}'"
                else:
                    formatted_value = repr(value)

                params.append(f"{param_name}={formatted_value}")

        # Join all parameters
        params_str = ", ".join(params)

        # Always use the same format for consistency
        return f"{display_name} ({cls_name}): {params_str}"

    def __str__(self):
        return self.__repr__()


class EMGAugmentation(FilterBaseClass):
    """Base class for EMG augmentation_pipelines."""

    def __init__(self, input_is_chunked: bool = None, is_output: bool = False):
        super().__init__(
            input_is_chunked=input_is_chunked,
            allowed_input_type="not chunked",
            is_output=is_output,
        )
