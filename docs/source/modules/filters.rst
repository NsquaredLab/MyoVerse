.. _filters:

Filters
=======

Filters are used to preprocess the data before it is used by the model.
Some filters can only work on chunked data, while others can work on both chunked and unchunked data.

DocOctoPy provides following types of filters:

    Generic filters e.g. allow the use of custom methods to preprocess the data.

    .. toctree::
        :maxdepth: 1

        filters/generic

    Temporal filters can be used to compute most EMG features such as RMS.

    .. toctree::
        :maxdepth: 1

        filters/temporal

    Augmentations can be used to generate new data points by applying transformations to the existing data.

    .. toctree::
        :maxdepth: 1

        filters/augmentation

    The base filter template is provided in the following section. The template can be used to create custom filters.

    .. toctree::
        :maxdepth: 1

        filters/template