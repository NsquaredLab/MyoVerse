from typing import Literal

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

    Methods
    -------
    __call__(input_array: np.ndarray) -> np.ndarray
        Filters the input array.
        Input shape is determined by whether the allowed_input_type is "both", "chunked" or "not chunked".
    """

    def __init__(
        self,
        input_is_chunked: bool = None,
        allowed_input_type: Literal["both", "chunked", "not chunked"] = None,
        is_output: bool = False,
        name: str = None,
    ):
        self.input_is_chunked = input_is_chunked
        self._allowed_input_type = allowed_input_type

        self.is_output = is_output
        self._name = name
        
    @property
    def name(self):
        if self._name is None:
            return self.__class__.__name__
        return self._name

    def __run_checks(self):
        if self._allowed_input_type is None:
            raise ValueError("allowed_input_type must be specified.")
        if self._allowed_input_type not in ["both", "chunked", "not chunked"]:
            raise ValueError(
                "allowed_input_type must be either 'both', 'chunked' or 'not chunked'."
            )

        if self._allowed_input_type == "both":
            return
        elif self._allowed_input_type == "chunked":
            if not self.input_is_chunked:
                raise ValueError(
                    f"This filter ({self.__class__.__name__}) only accepts chunked input."
                )
        elif self._allowed_input_type == "not chunked":
            if self.input_is_chunked:
                raise ValueError(
                    f"This filter ({self.__class__.__name__}) only accepts **un** chunked input."
                )

    def __call__(self, input_array: np.ndarray) -> np.ndarray:
        self.__run_checks()

        return self._filter(input_array)

    def _filter(self, input_array: np.ndarray) -> np.ndarray:
        raise NotImplementedError("This method must be implemented in the subclass.")

    def __repr__(self):
        # return (
        #     f"{self.__class__.__name__}"
        #     f'({", ".join([f"{k}={v}" for k, v in self.__dict__.items() if not k.startswith("_")])})'
        # )
        
        if self.name:
            return f"{self.name} ({self.__class__.__name__})"
        return f"{self.__class__.__name__}"

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
