
Data Types
================

Data types are used to store and process data in MyoVerse. They are used to store data in a structured way and to apply filters to the data.

.. currentmodule:: myoverse.datatypes
.. autosummary::
    :toctree: generated/datatypes
    :template: class.rst

    EMGData
    KinematicsData
    VirtualHandKinematics

If you wish to apply :ref:`filters` to the data, you can use the following functions:

.. currentmodule:: myoverse.datatypes
.. autosummary::
    :toctree: generated/datatypes
    :template: function.rst

    _Data.apply_filter
    _Data.apply_filter_sequence
    _Data.apply_filter_pipeline

Base Data Class
-----------------
.. important:: If you wish to add a new data type make sure they inherit from the following base class.

.. currentmodule:: myoverse.datatypes
.. autosummary::
    :toctree: generated/datatypes
    :template: class.rst

    _Data


