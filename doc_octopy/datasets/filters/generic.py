import warnings
from functools import partial
from typing import Sequence, Union

import numpy as np

from doc_octopy.datasets.filters._template import FilterBaseClass


class ApplyFunctionFilter(FilterBaseClass):
    """Filter that applies a function to the input array.

    Parameters
    ----------
    input_is_chunked : bool
        Whether the input is chunked or not.
    function : callable
        The function to apply. This can be any function that accepts a numpy array as input
        and returns a numpy array as output. Example: `np.mean` or lambda x: x + 1.
    is_output : bool
        Whether the filter is an output filter. If True, the resulting signal will be outputted by and dataset pipeline.

    Methods
    -------
    __call__(input_array: np.ndarray) -> np.ndarray
        Apply the function to the input array.
    """

    def __init__(
        self,
        input_is_chunked: bool = None,
        is_output: bool = False,
        name: str = None,
        function: callable = None,
        **function_kwargs,
    ):
        super().__init__(
            input_is_chunked=input_is_chunked,
            allowed_input_type="both",
            is_output=is_output,
            name=name,
        )

        self.function = partial(function, **function_kwargs)

        if not callable(self.function):
            raise ValueError("function must be a callable.")

    def _filter(self, input_array: np.ndarray) -> np.ndarray:
        return self.function(input_array)


class IndexDataFilter(FilterBaseClass):
    """Filter that indexes the input array.

    Parameters
    ----------
    indices : Sequence[Union[int, slice]]
        The indices to use for indexing the input array. Example: [0, 1, slice(2, 4)] will select the
        first two elements of the first dimension and the third and fourth elements of the second dimension.
    input_is_chunked : bool
        Whether the input is chunked or not.
    is_output : bool
        Whether the filter is an output filter. If True, the resulting signal will be outputted by and dataset pipeline.

    Methods
    -------
    __call__(input_array: np.ndarray) -> np.ndarray
        Filters the input array. Input shape is determined by whether the allowed_input_type
        is "both", "chunked" or "not chunked".
    """

    def __init__(
        self,
        input_is_chunked: bool = None,
        indices: Sequence[Union[int, slice]] = None,
        is_output: bool = False,
    ):
        super().__init__(
            input_is_chunked=input_is_chunked,
            allowed_input_type="both",
            is_output=is_output,
        )

        self.indices = indices

        if not isinstance(self.indices, (list, tuple)):
            raise ValueError("indices must be a list or tuple.")
        for i in self.indices:
            if not isinstance(i, (int, slice, tuple)):
                raise ValueError("Each element in indices must be an int or a slice.")

    def _filter(self, input_array: np.ndarray) -> np.ndarray:
        return input_array[self.indices]


class ChunkizeDataFilter(FilterBaseClass):
    """Filter that chunks the input array.

    Parameters
    ----------
    chunk_size : int
        The size of each chunk.
    chunk_shift : int
        The shift between each chunk. If provided, the chunk_overlap parameter is ignored.
    chunk_overlap : int
        The overlap between each chunk. If provided, the chunk_shift parameter is ignored.
    input_is_chunked : bool
        Whether the input is chunked or not.
    is_output : bool
        Whether the filter is an output filter. If True, the resulting signal will be outputted by and dataset pipeline.

    Methods
    -------
    __call__(input_array: np.ndarray) -> np.ndarray
        Filters the input array. Input shape is determined by whether the allowed_input_type
        is "both", "chunked" or "not chunked.
    """

    def __init__(
        self,
        input_is_chunked: bool = False,
        chunk_size: int = None,
        chunk_shift: int = None,
        chunk_overlap: int = None,
        is_output: bool = False,
        name: str = None,
    ):
        super().__init__(
            input_is_chunked=input_is_chunked,
            allowed_input_type="not chunked",
            is_output=is_output,
            name=name
        )

        if input_is_chunked == True:
            raise ValueError("This filter only accepts unchunked input.")

        self.chunk_size = chunk_size
        self.chunk_shift = chunk_shift
        self.chunk_overlap = chunk_overlap

        if self.chunk_size is None:
            raise ValueError("chunk_size must be specified.")
        if self.chunk_shift is None and self.chunk_overlap is None:
            raise ValueError("Either chunk_shift or chunk_overlap must be specified.")
        if self.chunk_shift is not None:
            if self.chunk_shift < 1:
                raise ValueError("chunk_shift must be greater than 0.")
            if self.chunk_shift >= self.chunk_size:
                warnings.warn(
                    "chunk_shift is greater than or equal to chunk_size. "
                    "Some parts of the data will be skipped. Be sure this is intended."
                )
        if self.chunk_overlap is not None:
            if self.chunk_overlap < 0:
                raise ValueError("chunk_overlap must be greater than or equal to 0.")
            if self.chunk_overlap > self.chunk_size:
                raise ValueError(
                    "chunk_overlap must be less than or equal to chunk_size."
                )

    def _filter(self, input_array: np.ndarray) -> np.ndarray:
        if self.chunk_shift is not None:
            return np.array(
                [
                    input_array[..., i : i + self.chunk_size]
                    for i in range(0, input_array.shape[-1], self.chunk_shift)
                    if i + self.chunk_size <= input_array.shape[-1]
                ]
            )

        return np.array(
            [
                input_array[..., i : i + self.chunk_size]
                for i in range(
                    0, input_array.shape[-1], self.chunk_size - self.chunk_overlap
                )
                if i + self.chunk_size <= input_array.shape[-1]
            ]
        )


class IdentityFilter(FilterBaseClass):
    """Filter that returns the input array unchanged.

    This filter is useful for debugging and testing purposes.

    Parameters
    ----------
    input_is_chunked : bool
        Whether the input is chunked or not.
    is_output : bool
        Whether the filter is an output filter. If True, the resulting signal will be outputted by and dataset pipeline.

    Methods
    -------
    __call__(input_array: np.ndarray) -> np.ndarray
        Returns the input array unchanged. If the input_array attribute is not None, this array will be returned.
    """

    def __init__(
        self,
        input_is_chunked: bool = None,
        is_output: bool = False,
        name: str = None,
    ):
        super().__init__(
            input_is_chunked=input_is_chunked,
            allowed_input_type="both",
            is_output=is_output,
            name=name,
        )

    def _filter(self, input_array: np.ndarray) -> np.ndarray:
        return input_array
