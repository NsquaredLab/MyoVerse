"""
Package basics
==============

This example is a brief introduction to the basic functionalities of the package.
"""


# %%
# Loading data
# ------------
# First we load the EMG example data and convert it to a DocOctoPy Data object.
# The Data object is the main object in the package and is used to store the data and apply filters to it.
import pickle as pkl
from doc_octopy.datatypes import EMGData

emg_data = {}
with open("data/emg.pkl", "rb") as f:
    for k, v in pkl.load(f).items():
        emg_data[k] = EMGData(v, sampling_frequency=2044)

print(emg_data)

# %%
# Looking at one specific task for simplicity
# -------------------------------------------
# The example data contains EMG from two different tasks labeled as "1" and "2".
# In the following we will only look at task one to explain the filtering functionalities.
task_one_data = emg_data["1"]

print(task_one_data)

# %%
# Understanding the saving format
# -------------------------------
# The EMGData object has a input_data attribute that stores the raw data.
#
# .. note:: The raw data is stored as a dictionary where the keys are "data" and "filter_sequence". The "data" key stores the raw data and the "filter_sequence" key stores the filter sequence applied to the data. For the raw data the filter sequence is always "Raw". However, when filters are applied the filter sequence is updated.
print(task_one_data.input_data)

# %%
# Plotting the raw data
# ---------------------
# We can plot the raw data using matplotlib.
import matplotlib.pyplot as plt
import numpy as np

raw_emg = task_one_data.input_data

# set plt font size
plt.rcParams.update({"font.size": 14})

for channel in range(raw_emg.shape[0]):
    plt.plot(raw_emg[channel], color="black", alpha=0.1)

plt.title("Raw EMG data")
plt.ylabel("Amplitude (a. u.)")
plt.xticks(
    np.arange(0, raw_emg.shape[-1] + 1, 2044).astype(int),
    np.arange(0, raw_emg.shape[-1] / 2044 + 1, 1).astype(int),
)
plt.xlabel("Time (s)")

plt.tight_layout()
plt.show()

# %%
# Attributes of the EMGData object
# --------------------------------
# Any Data object, of which EMGData is inheriting from, posses a processed_representations attribute where filtered data  will be stored.
#
# .. note :: We refer to a filtered data as a representation.
#
# At the beginning this attribute is empty.
print(task_one_data.processed_representations)

# %%
# Applying a filter
# -----------------
# The EMGData object has a method called apply_filter that applies a filter to the data.
# For example, we can apply a 4th order 20 HZ lowpass filter to the data.
from scipy.signal import butter
from doc_octopy.datasets.filters.temporal import SOSFrequencyFilter

sos_filter_coefficients = butter(4, 20, "lowpass", output="sos", fs=2044)

# %%
# Creating the filter
# -------------------
# Each filter has a parameter input_is_chunked that specifies if the input data is chunked or not.
# This must be set explicitly as some filters can only be used on either chunked or non-chunked data.
# Further having the user specify this parameter forces them to think about the data they are working with.
lowpass_filter = SOSFrequencyFilter(
    sos_filter_coefficients, is_output=True, name="Lowpass"
)
print(lowpass_filter)

# %%
# Applying the filter
# -------------------
# To apply the filter we call the apply_filter method on the EMGData object.
task_one_data.apply_filter(
    lowpass_filter, representation_to_filter="Last"
)

print(task_one_data.processed_representations)

# %%
# Accessing the filtered data
# ---------------------------
# The filtered data is saved in the processed_data attribute of the EMGData object.
# Processed_data is a dictionary where the keys are the names of the filters sequence applied to the data.
#
# The key of the last filter sequence always is marked with "(Output)". This is the data that will be outputted by the dataset pipeline.
#
# In our example the key is (Output) Raw->SOSFrequencyFilter.
#
# The data can be accessed by indexing the processed_data attribute or by indexing the EMGData directly.
# In case you do not want to index using the filter sequence name, you can retrieve the last processed data by indexing with "Last".

print(task_one_data)

print(
    np.allclose(
        task_one_data.processed_representations["Lowpass"],
        task_one_data["Last"],
    )
)
