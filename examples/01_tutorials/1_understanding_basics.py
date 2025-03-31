"""
Package basics
==============

This example is a brief introduction to the basic functionalities of the package.
"""

import os
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from myoverse.datatypes import EMGData
from scipy.signal import butter
from myoverse.datasets.filters.temporal import SOSFrequencyFilter

# %%
# Loading data
# ------------
# First we load the EMG example data and convert it to a MyoVerse Data object.
# The Data object is the primary component of the package, designed to store data and apply filters.
# The only required parameter is the sampling frequency of the data and the data itself.

# Get the path to the data file
data_path = os.path.join("..", "data", "emg.pkl")

with open(data_path, "rb") as f:
    emg_data = {k: EMGData(v, sampling_frequency=2044) for k, v in pkl.load(f).items()}

print("EMG data loaded successfully:")
print(emg_data)

# %%
# Looking at one specific task for simplicity
# -------------------------------------------
# The example data contains EMG from two different tasks labeled as "1" and "2".
# In the following we will only look at task 1 to explain the filtering functionalities.
task_one_data = emg_data["1"]

print("\nTask 1 data details:")
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
print("\nRaw input data shape:")
print(task_one_data.input_data.shape)

# %%
# Plotting the raw data
# ---------------------
# We can plot the raw data using matplotlib.

# Create a figure with better dimensions for visualization
plt.figure(figsize=(12, 6))

# Get the raw EMG data
raw_emg = task_one_data.input_data

# Set plt font size for better readability
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

# Add a grid for better readability
plt.grid(True, linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()

# %%
# Attributes of the EMGData object
# --------------------------------
# Any Data object, of which EMGData is inheriting from, possesses a processed_representations attribute where filtered data will be stored.
#
# .. note :: We refer to a filtered data as a representation.
#
# At the beginning this attribute only contains the raw data with the key "Input".
print("\nInitial processed representations:")
print(task_one_data.processed_representations.keys())

# %%
# Applying a filter
# -----------------
# The EMGData object has a method called **apply_filter** that applies a filter to the data.
# For example, we can apply a 4th order 20 Hz lowpass filter to the data.

# Define the filter parameters
FILTER_ORDER = 4
CUTOFF_FREQ = 20  # Hz
SAMPLING_FREQ = 2044  # Hz

# Create the filter coefficients using a Butterworth filter design
sos_filter_coefficients = butter(
    FILTER_ORDER, CUTOFF_FREQ, "lowpass", output="sos", fs=SAMPLING_FREQ
)

# %%
# Creating the filter
# -------------------
# Each filter has a parameter **input_is_chunked** that specifies if the input data is chunked or not.
# This must be set explicitly as some filters can only be used on either chunked or non-chunked data.
# Since we want to have the result of the filter as an output representation, we set the parameter **is_output** to True.
# Further having the user specify this parameter forces them to think about the data they are working with.
lowpass_filter = SOSFrequencyFilter(
    sos_filter_coefficients=sos_filter_coefficients,
    is_output=True,
    name="Lowpass",
    input_is_chunked=False,
)
print("\nFilter configuration:")
print(lowpass_filter)

# %%
# Applying the filter
# -------------------
# To apply the filter we call the apply_filter method on the EMGData object.
task_one_data.apply_filter(lowpass_filter, representations_to_filter=["Input"])
print("\nFilter applied successfully!")
print("Available processed representations:")
print(task_one_data.processed_representations.keys())

# %%
# Accessing the filtered data
# ---------------------------
# The filtered data is saved in the **output_representations** and the **processed_representations** attributes of the EMGData object.
# In our example the key is "Lowpass".
#
# In case you do not want to index using the filter sequence name, you can retrieve the last processed data by indexing with "Last".

print("\nFiltered data details:")
print(task_one_data)

# Verify that both methods of accessing the filtered data return the same result
identical = np.allclose(
    task_one_data.output_representations["Lowpass"],
    task_one_data["Last"],
)
print(f"\nLowpass and Last point to the same data: {identical}")

# %%
# Visualizing the difference between raw and filtered data
# -------------------------------------------------------
# Let's plot both the raw and filtered data to see the effect of the lowpass filter

plt.figure(figsize=(12, 8))

# For visualization clarity, let's plot only one channel
channel_to_plot = 0
filtered_emg = task_one_data["Lowpass"]

# Plot raw EMG for the selected channel
plt.subplot(2, 1, 1)
plt.plot(raw_emg[channel_to_plot], color="blue", label="Raw EMG")
plt.title(f"Raw EMG - Channel {channel_to_plot + 1}")
plt.ylabel("Amplitude (a.u.)")
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()

# Plot filtered EMG for the selected channel
plt.subplot(2, 1, 2)
plt.plot(filtered_emg[channel_to_plot], color="red", label="Lowpass Filtered EMG")
plt.title(
    f"Lowpass Filtered EMG (Cutoff: {CUTOFF_FREQ} Hz) - Channel {channel_to_plot + 1}"
)
plt.ylabel("Amplitude (a.u.)")
plt.xlabel("Time (s)")
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()

plt.tight_layout()
plt.show()
