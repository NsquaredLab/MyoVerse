"""
Complex filtering
=================

This example shows how to apply a complex filter sequence to the data.
"""

# %%
# Loading data and selection one task for simplicity
# --------------------------------------------------
# Just as in the previous example we load the EMG example data and convert it to a MyoVerse Data object.
# Afterward, we select one task to work with.
import pickle as pkl

import numpy as np

from myoverse.datatypes import EMGData

emg_data = {}
with open("data/emg.pkl", "rb") as f:
    for k, v in pkl.load(f).items():
        emg_data[k] = EMGData(v, sampling_frequency=2048)

print(emg_data)

task_one_data = emg_data.copy()["1"]

print(task_one_data)

# %%
# Applying a basic filter sequence
# --------------------------------
# A common filter sequence used by us in our deep learning papers is to first apply a bandstop between 47.5 and 52.5 Hz
# to remove the powerline noise.
#
# Then we copied this filtered data and applied a lowpass filter at 20 Hz to remove high-frequency noise.
# The deep learning models were thus trained with 2 representations of the data, one with the powerline noise removed and one with the high-frequency and powerline noise removed.
#
# We can achieve this by applying two filters to the data using the apply_filters method.
#
# .. note:: Please run this code on your local machine as the plots are interactive and information can be seen by hovering over the nodes.
from scipy.signal import butter

from myoverse.datasets.filters.temporal import SOSFrequencyFilter, RMSFilter

# Define the filters
bandstop_filter = SOSFrequencyFilter(
    sos_filter_coefficients=butter(4, [47.5, 52.5], "bandstop", output="sos", fs=2048),
    is_output=True,
    name="Bandstop 50",
    input_is_chunked=False,
)
lowpass_filter = SOSFrequencyFilter(
    sos_filter_coefficients=butter(4, 20, "lowpass", output="sos", fs=2048),
    is_output=True,
    name="Lowpass 20",
    input_is_chunked=False,
)

# Apply the filters
task_one_data.apply_filter_sequence(
    filter_sequence=[bandstop_filter, lowpass_filter],
    representations_to_filter=["Input"],
)

print()
print(task_one_data)

task_one_data.plot_graph()


# %%
# Applying a complex filter sequence
# ----------------------------------
# In this example we will apply a more complex filter sequence to the data.
#
# The filter shall apply the following steps:
#
# 1. Chunk the data into 100 ms windows.
#
# 2. Apply a bandstop filter between 47.5 and 52.5 Hz to remove powerline noise.
#
# 3. Copy the filtered data and apply a lowpass filter at 20 Hz to remove high-frequency noise.
# 4. Compute the root mean square of the data from step 3.
#
# 5. The other copy of step 2 should be used to calculate the root mean square directly.
#
# The computation graph for this filter sequence is shown below:
#
# 1 -> 2 -> 3 -> 4
#      L --------> 5
#
# We can achieve this by applying five filters to the data using the **apply_filter_sequence** method and setting the is_output
# flag to True for the filters that should be kept in the dataset object.
from myoverse.datasets.filters.generic import ChunkizeDataFilter
from myoverse.datasets.filters.temporal import SOSFrequencyFilter

# reset the data
task_one_data = EMGData(emg_data["1"].input_data, sampling_frequency=2048)

# Define the filters
bandstop_filter = butter(4, [47.5, 52.5], "bandstop", output="sos", fs=2048)
lowpass_filter = butter(4, 20, "lowpass", output="sos", fs=2048)

# %%
# Apply the filters for steps 1 and 2
# -----------------------------------
CHUNK_SIZE = int(100 / 1000 * 2048)
CHUNK_SHIFT = 64

task_one_data.apply_filter_sequence(
    filter_sequence=[
        ChunkizeDataFilter(
            chunk_size=CHUNK_SIZE,
            chunk_shift=CHUNK_SHIFT,
            name="Windowed",
            input_is_chunked=False,
        ),
        SOSFrequencyFilter(
            sos_filter_coefficients=bandstop_filter,
            name="Bandstop 50",
            input_is_chunked=True,
        ),
    ],
    representations_to_filter=["Input"],
)

print(task_one_data)

task_one_data.plot_graph()

# %%
# Apply the filters for step 3 and 4
# ----------------------------------
task_one_data.apply_filter_sequence(
    filter_sequence=[
        SOSFrequencyFilter(
            sos_filter_coefficients=lowpass_filter,
            name="Lowpass 20",
            input_is_chunked=True,
        ),
        RMSFilter(
            is_output=True,
            name="RMS on Lowpass 20",
            input_is_chunked=True,
            window_size=CHUNK_SIZE,
        ),
    ],
    representations_to_filter=["Bandstop 50"],
)

print(task_one_data)

task_one_data.plot_graph()

# %%
# Apply the filters for step 5
# -----------------------------
task_one_data.apply_filter(
    RMSFilter(
        is_output=True,
        name="RMS on Bandstop 50",
        input_is_chunked=True,
        window_size=CHUNK_SIZE,
    ),
    representations_to_filter=["Bandstop 50"],
)

print(task_one_data)

task_one_data.plot_graph()

# %%
# Displaying the output
# ---------------------

import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 14})

filter_sequences = {0: "1->2->3->4", 1: "1->2->5"}

# make 2 subplots
fig, axs = plt.subplots(2, sharex=True, sharey=True)

for i, (key, value) in enumerate(task_one_data.output_representations.items()):
    for channel in range(value.shape[-2]):
        axs[i].plot(value[:, channel, 0], color="black", alpha=0.01)

    axs[i].set_title(f"Filter sequence: {filter_sequences[i]}")
    axs[i].set_ylabel("Amplitude (a. u.)")

plt.xlabel("Samples (a. u.)")

plt.tight_layout()
plt.show()

# %%
# Easier way of applying the filter pipeline
# ------------------------------------------
# This can be achieved in a more concise way by using the **apply_filter_pipeline** method.
#
# To reduce memory usage, we can set the **keep_individual_filter_steps** flag to False.
# This will remove the intermediate representations (shown in grey) from the dataset object.
#
# .. note:: If new filters rely on the intermediate representations, they will be recalculated which can be computationally expensive.
task_one_data = EMGData(emg_data["1"].input_data, sampling_frequency=2048)

print(task_one_data)

# Apply the filters
task_one_data.apply_filter_pipeline(
    filter_pipeline=[
        [
            ChunkizeDataFilter(
                chunk_size=CHUNK_SIZE,
                chunk_shift=CHUNK_SHIFT,
                name="Windowed",
                input_is_chunked=False,
            ),
            SOSFrequencyFilter(
                sos_filter_coefficients=bandstop_filter,
                name="Bandstop 50",
                input_is_chunked=True,
            ),
            SOSFrequencyFilter(
                sos_filter_coefficients=lowpass_filter,
                name="Lowpass 20",
                input_is_chunked=True,
            ),
            RMSFilter(
                is_output=True,
                name="RMS on Lowpass 20",
                input_is_chunked=True,
                window_size=CHUNK_SIZE,
            ),
        ],
        [
            RMSFilter(
                is_output=True,
                name="RMS on Bandstop 50",
                input_is_chunked=True,
                window_size=CHUNK_SIZE,
            ),
        ],
    ],
    representations_to_filter=[["Input"], ["Bandstop 50"]],
    keep_individual_filter_steps=False,
)

print(task_one_data)

task_one_data.plot_graph()
