"""
Package basics
==============

This example is a brief introduction to the basic functionalities of the package.
"""


# %%
# Loading data
# ------------
# First we load the EMG example data and convert it to a MyoVerse Data object.
# The Data object is the primary component of the package, designed to store data and apply filters.
# The only required parameter is the sampling frequency of the data and the data itself.
import pickle as pkl
from myo_verse.datatypes import EMGData

with open("data/emg.pkl", "rb") as f:
    emg_data =  {k: EMGData(v, sampling_frequency=2044) for k, v in pkl.load(f).items()}

print(emg_data)

# %%
# Looking at one specific task for simplicity
# -------------------------------------------
# The example data contains EMG from two different tasks labeled as "1" and "2".
# In the following we will only look at task 1 to explain the filtering functionalities.
task_one_data = emg_data["1"]

print(task_one_data)

# %%
# Understanding the saving format
# -------------------------------
# The EMGData object has an **input_data** attribute that stores the raw data.
#
# The raw data is also added to the **processed_representations** attribute with the key "Input".
# The processed_representations attribute is a dictionary where all filtering sequences are stored.
# At the beginning this dictionary contains only the raw data.
#
# The attribute **is_chunked** is a dictionary that stores if the data of a particular representation is chunked or not.
#
# The attribute **output_representations** is a dictionary that stores the representation that will be outputted by the dataset pipeline.
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
# At the beginning this attribute only contains the raw data with the key "Input".
print(task_one_data.processed_representations)

# %%
# Applying a filter
# -----------------
# The EMGData object has a method called **apply_filter** that applies a filter to the data.
# For example, we can apply a 4th order 20 HZ lowpass filter to the data.
from scipy.signal import butter
from myo_verse.datasets.filters.temporal import SOSFrequencyFilter

sos_filter_coefficients = butter(4, 20, "lowpass", output="sos", fs=2044)

# %%
# Creating the filter
# -------------------
# Each filter has a parameter **input_is_chunked** that specifies if the input data is chunked or not.
# This must be set explicitly as some filters can only be used on either chunked or non-chunked data.
# Since we want to have the result of the filter as an output representation, we set the parameter **is_output** to True.
# Further having the user specify this parameter forces them to think about the data they are working with.
lowpass_filter = SOSFrequencyFilter(
    sos_filter_coefficients, is_output=True, name="Lowpass", input_is_chunked=False
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
# The filtered data is saved in the **output_representations** and the **processed_representations** attributes of the EMGData object.
# In our example the key is "Lowpass".
#
# In case you do not want to index using the filter sequence name, you can retrieve the last processed data by indexing with "Last".

print(task_one_data)

print(
    np.allclose(
        task_one_data.output_representations["Lowpass"],
        task_one_data["Last"],
    )
)
