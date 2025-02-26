import warnings
from functools import partial
from typing import Sequence, Union

import numpy as np

from myoverse.datasets.filters._template import FilterBaseClass


class ApplyFunctionFilter(FilterBaseClass):
    """Filter that applies a function to the input array.

    This filter provides a flexible way to apply any function to the input data array.
    The function can be a simple lambda, a NumPy function, or any custom function that
    operates on numpy arrays.

    Parameters
    ----------
    input_is_chunked : bool, optional
        Whether the input is chunked or not. If None, the filter will infer this from the
        input shape during the first call.
    function : callable
        The function to apply. This can be any function that accepts a numpy array as input
        and returns a numpy array as output. Examples include `np.mean`, `np.abs`, or 
        lambda x: x + 1.
    is_output : bool, optional
        Whether the filter is an output filter. If True, the resulting signal will be 
        outputted by any dataset pipeline, by default False.
    name : str, optional
        The name of the filter, by default None.
    **function_kwargs
        Additional keyword arguments to pass to the function when it is called.

    Raises
    ------
    ValueError
        If the provided function is not callable.

    Examples
    --------
    >>> import numpy as np
    >>> from myoverse.datasets.filters.generic import ApplyFunctionFilter
    >>> # Create data
    >>> data = np.random.rand(10, 500)
    >>> # Apply absolute value function
    >>> abs_filter = ApplyFunctionFilter(function=np.abs)
    >>> abs_data = abs_filter(data)
    >>> # Apply mean function with axis parameter
    >>> mean_filter = ApplyFunctionFilter(function=np.mean, axis=-1)
    >>> mean_data = mean_filter(data)
    >>> # Apply custom function
    >>> custom_filter = ApplyFunctionFilter(function=lambda x: x**2 - x)
    >>> custom_data = custom_filter(data)

    Notes
    -----
    The function is applied directly to the input array without any pre-processing.
    The output shape depends on the function being applied. For example, np.mean with
    axis=-1 will reduce the last dimension, changing the output shape.
    
    See Also
    --------
    IdentityFilter : A filter that returns the input unchanged
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
    """Filter that indexes the input array using NumPy-style indexing.

    This filter provides a flexible way to select specific elements or slices from the input array
    using NumPy's powerful indexing syntax. It supports basic slicing, integer array indexing,
    boolean masks, ellipsis, and advanced indexing.

    Parameters
    ----------
    input_is_chunked : bool, optional
        Whether the input is chunked or not. If None, the filter will infer this from
        the input shape during the first call.
    indices : any valid NumPy index
        The indices to use for indexing the input array. Can be:
        - Single index: 0, -1
        - Slice: slice(0, 10), slice(None, None, 2)
        - Tuple of indices for multiple dimensions: (0, slice(None), [1, 2, 3])
        - Ellipsis: ... or Ellipsis
        - Boolean mask: array([True, False, True])
        - Integer arrays for fancy indexing: np.array([0, 2, 4])
        - Any combination of the above
    is_output : bool, optional
        Whether the filter is an output filter. If True, the resulting signal will be 
        outputted by any dataset pipeline, by default False.
    name : str, optional
        The name of the filter, by default None.

    Examples
    --------
    >>> import numpy as np
    >>> from myoverse.datasets.filters.generic import IndexDataFilter
    >>> # Create data
    >>> data = np.random.rand(5, 10, 100)
    >>> 
    >>> # Select first element of first dimension
    >>> filter_first = IndexDataFilter(indices=0)
    >>> output = filter_first(data)  # shape: (10, 100)
    >>> 
    >>> # Select first three elements of last dimension
    >>> filter_slice = IndexDataFilter(indices=(slice(None), slice(None), slice(0, 3)))
    >>> output = filter_slice(data)  # shape: (5, 10, 3)
    >>> 
    >>> # Select specific elements with fancy indexing
    >>> filter_fancy = IndexDataFilter(indices=([0, 2], slice(None), [0, 50, 99]))
    >>> output = filter_fancy(data)  # shape: (2, 10, 3)
    >>> 
    >>> # Use ellipsis to simplify indexing (equivalent to the above)
    >>> filter_ellipsis = IndexDataFilter(indices=([0, 2], ..., [0, 50, 99]))
    >>> output = filter_ellipsis(data)  # shape: (2, 10, 3)
    >>> 
    >>> # Select specific elements from the last dimension
    >>> filter_lastdim = IndexDataFilter(indices=(slice(None), slice(None), [0, 1, 2]))
    >>> output = filter_lastdim(data)  # shape: (5, 10, 3)

    Notes
    -----
    This filter directly passes the provided indices to NumPy's indexing system.
    The behavior will match exactly what you would expect from numpy.ndarray indexing.
    """

    def __init__(
        self,
        input_is_chunked: bool = None,
        indices=None,
        is_output: bool = False,
        name: str = None,
    ):
        super().__init__(
            input_is_chunked=input_is_chunked,
            allowed_input_type="both",
            is_output=is_output,
            name=name,
        )

        self.indices = indices

    def _filter(self, input_array: np.ndarray) -> np.ndarray:
        """Apply the indices to the input array.
        
        This method directly applies the indices to the input array using NumPy's 
        indexing system, which supports basic slicing, integer array indexing, 
        boolean masks, ellipsis, and advanced indexing.
        
        Parameters
        ----------
        input_array : np.ndarray
            The input array to index
            
        Returns
        -------
        np.ndarray
            The indexed array
        """
        # Use direct NumPy indexing
        return input_array[self.indices]


class ChunkizeDataFilter(FilterBaseClass):
    """Filter that chunks the input array into overlapping or non-overlapping segments.

    This filter divides a continuous signal into chunks along the last dimension.
    It's useful for preparing data for window-based analysis or for applying 
    sliding window techniques.

    Parameters
    ----------
    input_is_chunked : bool, optional
        Whether the input is chunked or not. This filter only accepts unchunked input
        (input_is_chunked=False), by default False.
    chunk_size : int
        The size of each chunk along the last dimension.
    chunk_shift : int, optional
        The shift between consecutive chunks. If provided, chunk_overlap is ignored.
        A small shift creates more overlapping chunks.
    chunk_overlap : int, optional
        The overlap between consecutive chunks. If provided, chunk_shift is ignored.
        Overlap = chunk_size - chunk_shift.
    is_output : bool, optional
        Whether the filter is an output filter. If True, the resulting signal will be 
        outputted by any dataset pipeline, by default False.
    name : str, optional
        The name of the filter, by default None.

    Raises
    ------
    ValueError
        If input_is_chunked is True (this filter only accepts unchunked input).
    ValueError
        If chunk_size is not specified.
    ValueError
        If neither chunk_shift nor chunk_overlap is specified.
    ValueError
        If chunk_shift is less than 1.
    ValueError
        If chunk_overlap is less than 0 or greater than chunk_size.

    Examples
    --------
    >>> import numpy as np
    >>> from myoverse.datasets.filters.generic import ChunkizeDataFilter
    >>> # Create data
    >>> data = np.random.rand(10, 1000)
    >>> # Create non-overlapping chunks
    >>> no_overlap = ChunkizeDataFilter(
    ...     chunk_size=100,
    ...     chunk_shift=100,
    ...     input_is_chunked=False
    ... )
    >>> chunked_data = no_overlap(data)  # shape: (10, 10, 100)
    >>> # Create overlapping chunks
    >>> with_overlap = ChunkizeDataFilter(
    ...     chunk_size=100,
    ...     chunk_overlap=50,
    ...     input_is_chunked=False
    ... )
    >>> overlapped_data = with_overlap(data)  # shape: (19, 10, 100)

    Notes
    -----
    The output shape will be (n_chunks, *original_dims, chunk_size), where
    n_chunks = (input_length - chunk_size) // chunk_shift + 1 or
    n_chunks = (input_length - chunk_size) // (chunk_size - chunk_overlap) + 1

    When both chunk_shift and chunk_overlap are provided, chunk_shift takes precedence.

    See Also
    --------
    _get_windows_with_shift : Efficient windowing function used in temporal filters
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
            name=name,
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

    This filter passes the input data through without modification. It's useful for 
    debugging, testing, or as a placeholder in a pipeline when no transformation is needed.

    Parameters
    ----------
    input_is_chunked : bool, optional
        Whether the input is chunked or not. If None, the filter will infer this from
        the input shape during the first call.
    is_output : bool, optional
        Whether the filter is an output filter. If True, the resulting signal will be 
        outputted by any dataset pipeline, by default False.
    name : str, optional
        The name of the filter, by default None.

    Examples
    --------
    >>> import numpy as np
    >>> from myoverse.datasets.filters.generic import IdentityFilter
    >>> # Create data
    >>> data = np.random.rand(10, 500)
    >>> # Apply identity filter
    >>> identity = IdentityFilter(is_output=True)
    >>> output_data = identity(data)
    >>> # Verify data is unchanged
    >>> np.array_equal(data, output_data)
    True

    Notes
    -----
    While this filter might seem trivial, it can be useful in several scenarios:
    - As a placeholder in a configurable pipeline
    - For debugging or logging purposes when integrated with logging
    - To mark a data stream as an output within a pipeline without modifying it
    - As a base for creating more complex filters that conditionally apply transformations

    See Also
    --------
    ApplyFunctionFilter : A more general filter for applying arbitrary functions
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
