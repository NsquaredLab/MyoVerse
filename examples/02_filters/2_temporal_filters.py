"""
============================
Working with Temporal Filters
============================

This example demonstrates how to use the temporal filters available in MyoVerse.
Temporal filters process EMG signals in the time domain, such as frequency filtering,
rectification, and feature extraction like RMS, MAV, etc.
"""

import os
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from myoverse.datatypes import EMGData
from scipy.signal import butter
from myoverse.datasets.filters.temporal import (
    SOSFrequencyFilter,
    RectifyFilter,
    RMSFilter,
    MAVFilter,
    VARFilter,
    IAVFilter,
    WFLFilter,
    ZCFilter,
    SSCFilter,
    SpectralInterpolationFilter,
)
from myoverse.datasets.filters._template import FilterBaseClass
from scipy.signal import medfilt

# %%
# Loading data
# ------------
# First we load example EMG data

# Get the path to the data file
data_path = os.path.join("..", "data", "emg.pkl")

with open(data_path, "rb") as f:
    emg_data = {k: EMGData(v, sampling_frequency=2044) for k, v in pkl.load(f).items()}

# Use task 1 data for our examples
task_one_data = emg_data["1"]
print("EMG data loaded:", task_one_data)

# %%
# Visualizing the raw signal
# -------------------------
plt.figure(figsize=(12, 4))
raw_emg = task_one_data.input_data
channel = 0  # Choose one channel for visualization

# Plot first 5 seconds of data
time_in_sec = 5
samples_to_plot = int(time_in_sec * task_one_data.sampling_frequency)

plt.plot(raw_emg[channel, :samples_to_plot], color="blue")
plt.title("Raw EMG Signal (Channel 0)")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()

# %%
# 1. Basic Frequency Filtering
# ---------------------------
#
# Frequency filtering is one of the most important preprocessing steps for EMG signals.
# Let's apply a bandpass filter to remove noise and extract the useful EMG frequency band.

# Define the filter parameters for a bandpass filter (typical EMG range: 20-450 Hz)
FILTER_ORDER = 4
LOW_CUT = 20  # Hz
HIGH_CUT = 450  # Hz
SAMPLING_FREQ = 2044  # Hz

# Create bandpass filter coefficients
sos_bandpass = butter(
    FILTER_ORDER, [LOW_CUT, HIGH_CUT], btype="bandpass", output="sos", fs=SAMPLING_FREQ
)

# Create the MyoVerse filter
bandpass_filter = SOSFrequencyFilter(
    sos_filter_coefficients=sos_bandpass,
    is_output=True,
    name="Bandpass",
    input_is_chunked=False,
)

# Apply the filter
task_one_data.apply_filter(bandpass_filter, representations_to_filter=["Input"])

# Visualize the raw vs filtered signal
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(raw_emg[channel, :samples_to_plot], color="blue")
plt.title("Raw EMG Signal")
plt.ylabel("Amplitude")
plt.grid(True, linestyle="--", alpha=0.7)

plt.subplot(2, 1, 2)
filtered_emg = task_one_data["Bandpass"]
plt.plot(filtered_emg[channel, :samples_to_plot], color="red")
plt.title(f"Bandpass Filtered EMG ({LOW_CUT}-{HIGH_CUT} Hz)")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.grid(True, linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()

# %%
# 2. Rectification
# --------------
#
# Rectification is a common step in EMG processing that converts negative values to positive,
# making the signal suitable for envelope extraction.

# Create a rectification filter
rectify_filter = RectifyFilter(input_is_chunked=False, is_output=True, name="Rectified")

# Apply the filter to the bandpass filtered data
task_one_data.apply_filter(rectify_filter, representations_to_filter=["Bandpass"])

# Visualize the filtered vs rectified signal
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(filtered_emg[channel, :samples_to_plot], color="red")
plt.title("Bandpass Filtered EMG")
plt.ylabel("Amplitude")
plt.grid(True, linestyle="--", alpha=0.7)

plt.subplot(2, 1, 2)
rectified_emg = task_one_data["Rectified"]
plt.plot(rectified_emg[channel, :samples_to_plot], color="green")
plt.title("Rectified EMG")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.grid(True, linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()

# %%
# 3. RMS Feature Extraction
# -----------------------
#
# Root Mean Square (RMS) is a common feature extracted from EMG that represents
# the power of the signal over time windows.

# Define parameters for RMS calculation
WINDOW_SIZE = 400  # ~200ms at 2044 Hz
SHIFT = 100  # ~50ms step size for overlapping windows

# Create RMS filter
rms_filter = RMSFilter(
    input_is_chunked=False,
    is_output=True,
    name="RMS",
    window_size=WINDOW_SIZE,
    shift=SHIFT,
)

# Apply RMS filter to the rectified data
task_one_data.apply_filter(rms_filter, representations_to_filter=["Rectified"])

# Get the RMS feature
rms_feature = task_one_data["RMS"]

# Visualize the rectified signal and its RMS envelope
plt.figure(figsize=(12, 8))

# Determine how many RMS samples we have for our 5-second window
rms_samples_to_plot = len(rms_feature[channel])

# Calculate timestamps for proper alignment
time_raw = np.arange(samples_to_plot) / SAMPLING_FREQ
time_rms = np.arange(rms_samples_to_plot) * SHIFT / SAMPLING_FREQ

plt.subplot(2, 1, 1)
plt.plot(time_raw, rectified_emg[channel, :samples_to_plot], color="green", alpha=0.7)
plt.title("Rectified EMG")
plt.ylabel("Amplitude")
plt.grid(True, linestyle="--", alpha=0.7)

plt.subplot(2, 1, 2)
plt.plot(
    time_rms, rms_feature[channel, :rms_samples_to_plot], color="purple", linewidth=2
)
plt.title(
    f"RMS Feature (Window: {WINDOW_SIZE / SAMPLING_FREQ:.2f}s, Shift: {SHIFT / SAMPLING_FREQ:.2f}s)"
)
plt.xlabel("Time (seconds)")
plt.ylabel("RMS Amplitude")
plt.grid(True, linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()

# %%
# 4. Comparing Multiple Temporal Features
# -----------------------------------
#
# MyoVerse provides several temporal features for EMG analysis.
# Let's compare some of the most commonly used ones.

# Create a copy of our data for multiple features
feature_data = EMGData(task_one_data.input_data, sampling_frequency=2044)

# Apply the same bandpass and rectification
feature_data.apply_filter(bandpass_filter, representations_to_filter=["Input"])
feature_data.apply_filter(rectify_filter, representations_to_filter=["Bandpass"])

# Create filters for different features with same window parameters
mav_filter = MAVFilter(
    input_is_chunked=False,
    is_output=True,
    name="MAV",
    window_size=WINDOW_SIZE,
    shift=SHIFT,
)

var_filter = VARFilter(
    input_is_chunked=False,
    is_output=True,
    name="VAR",
    window_size=WINDOW_SIZE,
    shift=SHIFT,
)

iav_filter = IAVFilter(
    input_is_chunked=False,
    is_output=True,
    name="IAV",
    window_size=WINDOW_SIZE,
    shift=SHIFT,
)

# Apply the filters
feature_data.apply_filter(rms_filter, representations_to_filter=["Rectified"])
feature_data.apply_filter(mav_filter, representations_to_filter=["Rectified"])
feature_data.apply_filter(var_filter, representations_to_filter=["Rectified"])
feature_data.apply_filter(iav_filter, representations_to_filter=["Rectified"])

# Extract all features
rms_feature = feature_data["RMS"]
mav_feature = feature_data["MAV"]
var_feature = feature_data["VAR"]
iav_feature = feature_data["IAV"]


# Normalize each feature for better visualization
def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


# Plot all features
plt.figure(figsize=(12, 10))

# Original rectified signal
plt.subplot(5, 1, 1)
plt.plot(
    time_raw,
    feature_data["Rectified"][channel, :samples_to_plot],
    color="green",
    alpha=0.7,
)
plt.title("Rectified EMG")
plt.ylabel("Amplitude")
plt.grid(True, linestyle="--", alpha=0.7)

# RMS
plt.subplot(5, 1, 2)
plt.plot(
    time_rms, normalize(rms_feature[channel, :rms_samples_to_plot]), color="purple"
)
plt.title("RMS Feature (Normalized)")
plt.ylabel("Amplitude")
plt.grid(True, linestyle="--", alpha=0.7)

# MAV
plt.subplot(5, 1, 3)
plt.plot(time_rms, normalize(mav_feature[channel, :rms_samples_to_plot]), color="blue")
plt.title("MAV Feature (Normalized)")
plt.ylabel("Amplitude")
plt.grid(True, linestyle="--", alpha=0.7)

# VAR
plt.subplot(5, 1, 4)
plt.plot(time_rms, normalize(var_feature[channel, :rms_samples_to_plot]), color="red")
plt.title("VAR Feature (Normalized)")
plt.ylabel("Amplitude")
plt.grid(True, linestyle="--", alpha=0.7)

# IAV
plt.subplot(5, 1, 5)
plt.plot(
    time_rms, normalize(iav_feature[channel, :rms_samples_to_plot]), color="orange"
)
plt.title("IAV Feature (Normalized)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True, linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()

# %%
# 5. Frequency-Domain Features
# -------------------------
#
# MyoVerse also provides tools for handling frequency-related problems.
# Let's demonstrate the use of SpectralInterpolationFilter to remove power line noise.

# Create a new instance with original data
power_line_data = EMGData(task_one_data.input_data, sampling_frequency=2044)

# Create a spectral interpolation filter to remove 50Hz noise
spec_filter = SpectralInterpolationFilter(
    input_is_chunked=False,
    is_output=True,
    name="SpecInterp",
    bandwidth=(48, 52),  # 50Hz +/- 2Hz
    number_of_harmonics=3,  # Remove also 100Hz and 150Hz
    sampling_frequency=2044,
    interpolation_window=15,
    interpolation_poly_order=3,
)

# Apply the filter
power_line_data.apply_filter(spec_filter, representations_to_filter=["Input"])

# Apply bandpass for comparison with other methods
power_line_data.apply_filter(bandpass_filter, representations_to_filter=["Input"])


# Calculate FFT for visualization
def calculate_fft(signal, fs):
    n = len(signal)
    fft_result = np.fft.rfft(signal)
    fft_mag = np.abs(fft_result) / n
    freq = np.fft.rfftfreq(n, 1 / fs)
    return freq, fft_mag


# Get frequency data
freq_raw, fft_raw = calculate_fft(task_one_data.input_data[channel, :4096], 2044)
freq_cleaned, fft_cleaned = calculate_fft(
    power_line_data["SpecInterp"][channel, :4096], 2044
)
freq_bandpass, fft_bandpass = calculate_fft(
    power_line_data["Bandpass"][channel, :4096], 2044
)

# Plot the frequency domain for comparison
plt.figure(figsize=(12, 10))

# Time domain signals
plt.subplot(3, 1, 1)
plt.plot(task_one_data.input_data[channel, :samples_to_plot], color="blue")
plt.title("Raw EMG Signal")
plt.ylabel("Amplitude")
plt.grid(True, linestyle="--", alpha=0.7)

# FFT of the raw signal
plt.subplot(3, 1, 2)
plt.plot(freq_raw, fft_raw, color="blue")
plt.axvline(x=50, color="red", linestyle="--", label="50 Hz (Power line)")
plt.axvline(x=100, color="red", linestyle="--", alpha=0.7)
plt.axvline(x=150, color="red", linestyle="--", alpha=0.7)
plt.title("Frequency Spectrum - Raw Signal")
plt.ylabel("Magnitude")
plt.grid(True, linestyle="--", alpha=0.7)
plt.xlim(0, 500)
plt.legend()

# FFT of the cleaned signal
plt.subplot(3, 1, 3)
plt.plot(freq_cleaned, fft_cleaned, color="green", label="Spectral Interpolation")
plt.plot(
    freq_bandpass, fft_bandpass, color="orange", alpha=0.7, label="Bandpass Filter"
)
plt.axvline(x=50, color="red", linestyle="--", label="50 Hz (Power line)")
plt.axvline(x=100, color="red", linestyle="--", alpha=0.7)
plt.axvline(x=150, color="red", linestyle="--", alpha=0.7)
plt.title("Frequency Spectrum - Filtered Signals")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True, linestyle="--", alpha=0.7)
plt.xlim(0, 500)
plt.legend()

plt.tight_layout()
plt.show()

# %%
# 6. Real-time Processing with SOSFrequencyFilter
# ------------------------------------------
#
# MyoVerse's filters can also be used for real-time processing. Here we'll demonstrate
# how to use the real-time mode of SOSFrequencyFilter for streaming applications.

# Create a filter for real-time processing
realtime_filter = SOSFrequencyFilter(
    sos_filter_coefficients=sos_bandpass,
    is_output=True,
    name="Realtime",
    input_is_chunked=True,  # Now using chunked mode
    forwards_and_backwards=False,  # Must be False for real-time
)

# Chunk the data to simulate streaming
chunk_size = 204  # ~100ms chunks
n_chunks = 10  # Process 10 chunks (1 second)

# Create chunked data
chunked_data = np.zeros((n_chunks, task_one_data.input_data.shape[0], chunk_size))
for i in range(n_chunks):
    chunked_data[i] = task_one_data.input_data[:, i * chunk_size : (i + 1) * chunk_size]

# Create a new EMGData object with chunked input
chunked_emg = EMGData(chunked_data, sampling_frequency=2044)

# Apply the real-time filter
chunked_emg.apply_filter(realtime_filter, representations_to_filter=["Input"])

# Visualize the results
plt.figure(figsize=(12, 8))

# Plot original chunks
plt.subplot(2, 1, 1)
channel_to_plot = 0
for i in range(n_chunks):
    chunk_start = i * chunk_size
    plt.plot(
        np.arange(chunk_start, chunk_start + chunk_size),
        chunked_data[i, channel_to_plot],
        color=f"C{i}",
    )
    plt.axvline(x=chunk_start, color="black", linestyle=":", alpha=0.3)

plt.title("Original Signal (Chunked)")
plt.ylabel("Amplitude")
plt.grid(True, linestyle="--", alpha=0.7)

# Plot filtered chunks
plt.subplot(2, 1, 2)
filtered_chunks = chunked_emg["Realtime"]
for i in range(n_chunks):
    chunk_start = i * chunk_size
    plt.plot(
        np.arange(chunk_start, chunk_start + chunk_size),
        filtered_chunks[i, channel_to_plot],
        color=f"C{i}",
    )
    plt.axvline(x=chunk_start, color="black", linestyle=":", alpha=0.3)

plt.title("Real-time Filtered Signal")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.grid(True, linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()

# %%
# 7. Creating Your Own Custom Temporal Filter
# ----------------------------------------
#
# One of the strengths of MyoVerse is the ability to create custom filters.
# Let's create a custom temporal filter that applies median filtering to the EMG signal.


class MedianFilter(FilterBaseClass):
    """Custom filter that applies median filtering to the EMG signal.

    Parameters
    ----------
    input_is_chunked : bool
        Whether the input is chunked or not.
    is_output : bool
        Whether the filter is an output filter.
    name : str | None
        Name of the filter, by default None.
    run_checks : bool
        Whether to run the checks when filtering.
    kernel_size : int
        Size of the median filter kernel. Must be odd.
    """

    def __init__(
        self,
        input_is_chunked: bool,
        is_output: bool = False,
        name: str | None = None,
        run_checks: bool = True,
        *,
        kernel_size: int = 5,
    ):
        super().__init__(
            input_is_chunked=input_is_chunked,
            allowed_input_type="both",  # Works with both chunked and non-chunked data
            is_output=is_output,
            name=name,
            run_checks=run_checks,
        )

        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd")

        self.kernel_size = kernel_size

    def _filter(self, input_array: np.ndarray, **kwargs) -> np.ndarray:
        """Apply median filtering to the input array.

        Parameters
        ----------
        input_array : numpy.ndarray
            Input array to filter.
        **kwargs
            Additional keyword arguments from the Data object.

        Returns
        -------
        numpy.ndarray
            Filtered array.
        """
        # Get the shape of the input array
        original_shape = input_array.shape

        # Handle different input shapes
        if not self.input_is_chunked:
            # Non-chunked case: apply median filter to each channel
            output_array = np.zeros_like(input_array)
            for channel in range(original_shape[0]):
                output_array[channel] = medfilt(
                    input_array[channel], kernel_size=self.kernel_size
                )

            return output_array
        else:
            # Chunked case: apply median filter to each channel in each chunk
            output_array = np.zeros_like(input_array)

            # Loop through each chunk
            for chunk in range(original_shape[0]):
                # Loop through each channel
                for channel in range(original_shape[1]):
                    output_array[chunk, channel] = medfilt(
                        input_array[chunk, channel], kernel_size=self.kernel_size
                    )

            return output_array


# Create a new instance with original data
custom_filter_data = EMGData(task_one_data.input_data, sampling_frequency=2044)

# Create our custom median filter
median_filter = MedianFilter(
    input_is_chunked=False,
    is_output=True,
    name="MedianFiltered",
    kernel_size=11,  # Use a larger kernel to make the effect more visible
)

# Apply our custom filter
custom_filter_data.apply_filter(median_filter, representations_to_filter=["Input"])

# Visualize the result
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(raw_emg[channel, :samples_to_plot], color="blue")
plt.title("Raw EMG Signal")
plt.ylabel("Amplitude")
plt.grid(True, linestyle="--", alpha=0.7)

plt.subplot(2, 1, 2)
median_filtered = custom_filter_data["MedianFiltered"]
plt.plot(median_filtered[channel, :samples_to_plot], color="purple")
plt.title(f"Custom Median Filtered EMG (Kernel Size: {median_filter.kernel_size})")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.grid(True, linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()

# %%
# This example demonstrates how to create custom filters in MyoVerse.
# You can extend the FilterBaseClass to create any filter you need,
# opening up possibilities for specialized EMG processing techniques.
