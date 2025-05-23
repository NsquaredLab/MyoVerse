"""
====================================
Applying Spatial Filters to EMG Data
====================================

This example demonstrates how to use spatial filters on EMG data using MyoVerse.
We will showcase differential filters that help enhance spatial patterns in EMG signals.

"""

# %% Import necessary libraries
import numpy as np
import os
import pickle as pkl
import matplotlib.pyplot as plt

from myoverse.datatypes import EMGData
from myoverse.datasets.filters.spatial import DifferentialSpatialFilter, ApplyFunctionSpatialFilter


def plot_grids(data, grid_layouts, title, vmin=None, vmax=None):
    """Helper function to plot EMG data on grids."""
    n_grids = len(grid_layouts)
    fig, axes = plt.subplots(1, n_grids, figsize=(5 * n_grids, 5))
    if n_grids == 1:
        axes = [axes]  # Make it iterable if only one grid

    processed_channels = 0
    for i, grid_layout in enumerate(grid_layouts):
        rows, cols = grid_layout.shape
        n_channels_grid = rows * cols
        # Ensure we only try to access existing data
        end_channel = processed_channels + n_channels_grid
        grid_data_flat = data[processed_channels : min(end_channel, len(data))]

        # Create an empty grid filled with NaNs
        plot_grid = np.full(grid_layout.shape, np.nan)

        # Fill the grid with data based on layout indices (relative to grid start)
        min_channel_index = np.min(grid_layout)
        for r in range(rows):
            for c in range(cols):
                channel_index_abs = grid_layout[r, c]
                channel_index_rel = channel_index_abs - min_channel_index
                # Check if the relative index corresponds to a valid index in the flattened grid data
                if 0 <= channel_index_rel < len(grid_data_flat):
                    plot_grid[r, c] = grid_data_flat[channel_index_rel]

        min_val = np.nanmin(plot_grid)
        max_val = np.nanmax(plot_grid)

        im = axes[i].imshow(
            plot_grid,
            cmap="magma",
            vmin=min_val,
            vmax=max_val,
            origin="lower",
            interpolation="nearest",
        )
        axes[i].set_title(f"Grid {i}\nShape: {rows}x{cols}")
        axes[i].set_xticks(np.arange(cols))
        axes[i].set_yticks(np.arange(rows))

        # add colorbar
        cbar = plt.colorbar(im, ax=axes[i])

        # Add channel numbers as text
        for r in range(rows):
            for c in range(cols):
                channel_index_abs = grid_layout[r, c]
                # Determine text color based on background brightness
                if not np.isnan(plot_grid[r, c]):
                    bg_val_norm = (plot_grid[r, c] - min_val) / (
                        max_val - min_val + 1e-6
                    )  # Normalize to 0-1
                    text_color = "white" if bg_val_norm < 0.5 else "black"
                    axes[i].text(
                        c,
                        r,
                        str(channel_index_abs),
                        ha="center",
                        va="center",
                        color=text_color,
                    )

        processed_channels += n_channels_grid

    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent title overlap
    return fig, axes


# %%
# Loading EMG Data
# ---------------
#
# We create data with two grids: an 8x8 grid and a 4x4 grid.

data_path = os.path.join("..", "data", "emg.pkl")

# Load the raw data once
with open(data_path, "rb") as f:
    raw_data = pkl.load(f)["1"][: 8 * 8 + 4 * 4]
    sampling_frequency = 2048
    grid_layouts_orig = [np.arange(64).reshape(8, 8), np.arange(64, 80).reshape(4, 4)]


# %%
# Normal Double Differential (NDD) Filter
# --------------------------------------
#
# The NDD filter (also known as Laplacian filter) computes differences between
# adjacent electrodes in a cross pattern. It enhances local spatial patterns and
# reduces common noise.

# Create a new EMG object for NDD filter
emg_ndd = EMGData(
    raw_data.copy(),
    sampling_frequency=sampling_frequency,
    grid_layouts=[g.copy() for g in grid_layouts_orig],
)

ndd_filter = DifferentialSpatialFilter(
    filter_name="NDD",
    input_is_chunked=False,  # Our data is (channels, time)
    grids_to_process="all",
    name="NDD",
)

_ = emg_ndd.apply_filter(ndd_filter, representations_to_filter=["Input"])

ndd_rms = np.sqrt(np.mean(emg_ndd["NDD"] ** 2, axis=1))
fig_ndd, _ = plot_grids(
    ndd_rms,
    emg_ndd.grid_layouts,
    "NDD Filter (All Grids, RMS)",
    vmin=np.min(ndd_rms),
    vmax=np.max(ndd_rms),
)
plt.show()

print(f"Shape after NDD filter: {emg_ndd['NDD'].shape}")
print(f"Grid layouts after NDD filter: {[g.shape for g in emg_ndd.grid_layouts]}")


# %%
# Longitudinal Single Differential (LSD) Filter
# -------------------------------------------
#
# The LSD filter computes differences between adjacent electrodes along columns.
# Here we apply it only to the first grid (Grid 0).

# Create a new EMG object for LSD filter
emg_lsd = EMGData(
    raw_data.copy(),
    sampling_frequency=sampling_frequency,
    grid_layouts=[g.copy() for g in grid_layouts_orig],
)

lsd_filter = DifferentialSpatialFilter(
    filter_name="LSD",
    input_is_chunked=False,
    name="LSD",
)

_ = emg_lsd.apply_filter(lsd_filter, representations_to_filter=["Input"])

lsd_rms = np.sqrt(np.mean(emg_lsd["LSD"] ** 2, axis=1))
fig_lsd, _ = plot_grids(
    lsd_rms,
    emg_lsd.grid_layouts,
    "LSD Filter (Grid 0 Only, RMS)",
    vmin=np.min(lsd_rms),
    vmax=np.max(lsd_rms),
)
plt.show()

print(f"Shape after LSD filter: {emg_lsd['LSD'].shape}")
print(f"Grid layouts after LSD filter: {[g.shape for g in emg_lsd.grid_layouts]}")


# %%
# Transverse Single Differential (TSD) Filter
# -----------------------------------------
#
# The TSD filter computes differences between adjacent electrodes along rows.
# This filter emphasizes activity changes in the transverse direction.

# Create a new EMG object for TSD filter
emg_tsd = EMGData(
    raw_data.copy(),
    sampling_frequency=sampling_frequency,
    grid_layouts=[g.copy() for g in grid_layouts_orig],
)

tsd_filter = DifferentialSpatialFilter(
    filter_name="TSD", 
    input_is_chunked=False, 
    grids_to_process="all",
    name="TSD",
)

_ = emg_tsd.apply_filter(tsd_filter, representations_to_filter=["Input"])

tsd_rms = np.sqrt(np.mean(emg_tsd["TSD"] ** 2, axis=1))
fig_tsd, _ = plot_grids(
    tsd_rms,
    emg_tsd.grid_layouts,
    "TSD Filter (All Grids, RMS)",
    vmin=np.min(tsd_rms),
    vmax=np.max(tsd_rms),
)
plt.show()

print(f"Shape after TSD filter: {emg_tsd['TSD'].shape}")
print(f"Grid layouts after TSD filter: {[g.shape for g in emg_tsd.grid_layouts]}")


# %%
# Inverse Binomial (IB2) Filter
# ---------------------------
#
# The IB2 filter is an inverse binomial filter of the 2nd order that applies
# a more complex spatial weighting to enhance local activity patterns.

# Create a new EMG object for IB2 filter
emg_ib2 = EMGData(
    raw_data.copy(),
    sampling_frequency=sampling_frequency,
    grid_layouts=[g.copy() for g in grid_layouts_orig],
)

ib2_filter = DifferentialSpatialFilter(
    filter_name="IB2", 
    input_is_chunked=False, 
    grids_to_process="all",
    name="IB2",
)

_ = emg_ib2.apply_filter(ib2_filter, representations_to_filter=["Input"])

ib2_rms = np.sqrt(np.mean(emg_ib2["IB2"] ** 2, axis=1))
fig_ib2, _ = plot_grids(
    ib2_rms,
    emg_ib2.grid_layouts,
    "IB2 Filter (All Grids, RMS)",
    vmin=np.min(ib2_rms),
    vmax=np.max(ib2_rms),
)
plt.show()

print(f"Shape after IB2 filter: {emg_ib2['IB2'].shape}")
print(f"Grid layouts after IB2 filter: {[g.shape for g in emg_ib2.grid_layouts]}")

# %%
# Apply Function Spatial Filter
# ------------------------------
#
# The Apply Function Spatial Filter allows us to apply custom functions to the EMG data.
# Here we use it to compute the mean across a 2x2 kernel with a stride of 2 in both dimensions.
def mean_function(data):
    """Custom function to compute the mean across the last two dimensions.

    data: numpy.ndarray
        Input data to apply the function on. The shape should be (channels, time, height, width) if the input is chunked otherwise (time, height, width).

        .. note:: The function must only modify the last two dimensions of the data and provide the same shape as output.
    Returns:
        numpy.ndarray: Mean of the input data across the last two dimensions.
    """
    return np.mean(data, axis=(-1, -2), keepdims=True)

# Create a new EMG object for Apply Function filter
emg_apply_func = EMGData(
    raw_data.copy(),
    sampling_frequency=sampling_frequency,
    grid_layouts=[g.copy() for g in grid_layouts_orig],
)
apply_func_filter = ApplyFunctionSpatialFilter(
    input_is_chunked=False,
    grids_to_process="all",
    name="Apply Function",
    kernel_size=(2, 2),
    padding="same",
    strides=(2, 2),
    function=mean_function,
)

_ = emg_apply_func.apply_filter(
    apply_func_filter, representations_to_filter=["Input"]
)
apply_func_rms = np.sqrt(np.mean(emg_apply_func["Apply Function"] ** 2, axis=1))

fig_apply_func, _ = plot_grids(
    apply_func_rms,
    emg_apply_func.grid_layouts,
    "Apply Function Filter (All Grids, RMS)",
    vmin=np.min(apply_func_rms),
    vmax=np.max(apply_func_rms),
)
plt.show()

print(f"Shape after Apply Function filter: {emg_apply_func['Apply Function'].shape}")
print(f"Grid layouts after Apply Function filter: {[g.shape for g in emg_apply_func.grid_layouts]}")
