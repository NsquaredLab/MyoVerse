.. _filters:

Filters
=======

Filters are used to preprocess the data before it is used by the model.
Some filters can only work on chunked data, while others can work on both chunked and unchunked data.

Generic Filters
---------------
Generic filters e.g. allow the use of custom methods to preprocess the data.

.. currentmodule:: myoverse.datasets.filters.generic
.. autosummary::
    :toctree: generated/filters
    :template: class.rst

    ApplyFunctionFilter
    IndexDataFilter
    ChunkizeDataFilter
    IdentityFilter

Temporal Filters
----------------
Temporal filters can be used to compute most EMG features such as the root mean square or the mean absolute value.

.. currentmodule:: myoverse.datasets.filters.temporal
.. autosummary::
    :toctree: generated/filters
    :template: class.rst

    SOSFrequencyFilter
    RectifyFilter
    WindowedFunctionFilter
    RMSFilter
    VARFilter
    MAVFilter
    IAVFilter
    WFLFilter
    ZCFilter
    SSCFilter
    SpectralInterpolationFilter


Spatial Filters
---------------
Spatial filters can be used to compute spatial features such as the Laplacian.

.. currentmodule:: myoverse.datasets.filters.spatial
.. autosummary::
    :toctree: generated/filters
    :template: class.rst

    DifferentialSpatialFilter
    ApplyFunctionSpatialFilter


Base Filter Classes
-------------------
.. important:: If you wish to add a new filter make sure they inherit from the following base classes.

.. currentmodule:: myoverse.datasets.filters.generic
.. autosummary::
    :toctree: generated/filters
    :template: class.rst

    FilterBaseClass

.. currentmodule:: myoverse.datasets.filters.spatial
.. autosummary::
    :toctree: generated/filters
    :template: class.rst

    SpatialFilterGridAware

