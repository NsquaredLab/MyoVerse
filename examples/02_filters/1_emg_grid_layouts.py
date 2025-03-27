"""
========================================
EMG Grid Layouts: Electrode Arrangements
========================================

This example demonstrates how to work with the ``grid_layouts`` parameter in the ``EMGData`` class.
Grid layouts allow you to specify the exact arrangement of electrodes on physical recording grids,
which is essential for:

- Visualizing high-density EMG recordings in their spatial context
- Working with data from multiple electrode arrays
- Handling non-rectangular or incomplete electrode configurations
- Performing spatial filtering or neighborhood-based operations

The ``grid_layouts`` parameter provides precise control over electrode positioning, numbering,
and visualization, making it valuable for analyzing complex multichannel EMG recordings.
"""

# %%
# First, let's import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from myoverse.datatypes import EMGData, create_grid_layout

# %%
# Creating basic grid layouts
# ==========================
#
# The ``create_grid_layout`` function helps create grid layouts with different patterns and configurations.
# Let's create some examples to understand the options:

# Create a 4×4 grid with row-wise numbering (0-15)
grid_row = create_grid_layout(4, 4, fill_pattern="row")
print("4×4 grid with row-wise numbering:")
print(grid_row)
print()

# Create a 4×4 grid with column-wise numbering (0-15)
grid_col = create_grid_layout(4, 4, fill_pattern="column")
print("4×4 grid with column-wise numbering:")
print(grid_col)
print()

# %%
# Visualizing the grid layouts
# ===========================
#
# Let's create random EMG data and use these grid layouts to visualize it.
# We'll use our enhanced `plot_grid_layout` method to create professional visualizations.

# Create sample EMG data (16 channels, 1000 samples)
emg_data_16ch = np.random.randn(16, 1000)
sampling_freq = 2000  # 2000 Hz

# Create EMG objects with each grid layout
emg_row = EMGData(emg_data_16ch, sampling_freq, grid_layouts=[grid_row])
emg_col = EMGData(emg_data_16ch, sampling_freq, grid_layouts=[grid_col])

# Create a figure with subplots for comparison
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 12))
fig.suptitle("Comparing Row-wise vs\nColumn-wise Electrode Numbering", fontsize=16)

# Use the enhanced plot_grid_layout method with explicit axes and autoshow=False
emg_row.plot_grid_layout(
    0,
    title="Row-wise Numbering",
    colorbar=False,
    grid_alpha=0.7,
    ax=ax1,
    autoshow=False,  # Don't show yet - wait until both grids are plotted
)

# Plot column-wise grid on the second axis
emg_col.plot_grid_layout(
    0,
    title="Column-wise Numbering",
    colorbar=False,
    grid_alpha=0.7,
    ax=ax2,
    autoshow=False,  # Don't show yet
)

# Now show the complete figure with both subplots
plt.tight_layout()
plt.show()

# %%
# Working with missing electrodes
# ==============================
#
# In real-world scenarios, electrode grids may have missing or non-functional electrodes.
# The ``grid_layouts`` parameter allows you to specify these gaps using -1 values.

# Create a 5×5 grid with some missing electrodes
missing_indices = [(0, 0), (2, 2), (4, 4)]  # Positions where electrodes are missing
grid_with_gaps = create_grid_layout(
    5, 5, fill_pattern="row", missing_indices=missing_indices
)

print("5×5 grid with missing electrodes:")
print(grid_with_gaps)
print()

# Create EMG data with 22 channels (25 total positions - 3 missing)
emg_data_22ch = np.random.randn(22, 1000)
emg_with_gaps = EMGData(emg_data_22ch, sampling_freq, grid_layouts=[grid_with_gaps])

# Visualize the grid with gaps using the enhanced method
emg_with_gaps.plot_grid_layout(
    0,
    title="5×5 Grid with Missing Electrodes",
    colorbar=True,
    figsize=(8, 8),
    text_fontsize=12,
)

# %%
# Multiple electrode grids
# =======================
#
# Many EMG experiments use multiple electrode grids simultaneously. The ``grid_layouts``
# parameter accepts a list of grids, allowing you to represent complex multi-array setups.
#
# .. note::
#    In this example, we use consecutive electrode indices for each grid (0-15, then 16-31,
#    then 32-40) without any gaps. This approach offers several practical advantages:
#
#    1. It matches how hardware/acquisition systems typically organize electrode channels,
#       where each grid's electrodes are grouped together in the recording system.
#
#    2. It simplifies data interpretation and visualization, making it easier to identify
#       which channels belong to which physical grid.
#
#    3. It makes it more intuitive to apply grid-specific processing or analysis,
#       as you can easily select channel ranges (e.g., channels 0-15 for grid 1).
#
#    You could use any arbitrary, non-overlapping channel indices for each grid
#    as long as they correctly map to the corresponding channels in your EMG data.

# Create all three grid layouts
grid1 = create_grid_layout(4, 4, fill_pattern="row")  # 16 electrodes
grid2 = create_grid_layout(4, 4, fill_pattern="column")  # 16 electrodes
grid2[grid2 >= 0] += 16  # Shift indices to start after grid1
grid3 = create_grid_layout(3, 3, fill_pattern="row")  # 9 electrodes
grid3[grid3 >= 0] += 32  # Shift indices to start after grid2

# Calculate total number of electrodes
n_electrodes = 16 + 16 + 9  # = 41

# Create EMG data with exactly the right number of channels
emg_data_41ch = np.random.randn(n_electrodes, 1000)

# Create EMGData with the validated class
emg_multi = EMGData(emg_data_41ch, sampling_freq, grid_layouts=[grid1, grid2, grid3])

# Create a single figure with multiple subplots
fig, axes = plt.subplots(3, 1, figsize=(5, 15))
fig.suptitle("Multiple Electrode Grids", fontsize=16)

# We'll use our enhanced plot_grid_layout method with provided axes
# Let's highlight some electrodes in each grid to demonstrate that feature
highlights1 = [5, 10]  # Highlight electrodes 5 and 10 in first grid
highlights2 = [20, 25]  # Highlight electrodes 20 and 25 in second grid
highlights3 = [35]  # Highlight electrode 35 in third grid

# Plot all three grids on the same figure
emg_multi.plot_grid_layout(
    0,
    title="Grid 1: 4×4 Row-wise (0-15)",
    colorbar=False,
    highlight_electrodes=highlights1,
    ax=axes[0],
    autoshow=False,
)

emg_multi.plot_grid_layout(
    1,
    title="Grid 2: 4×4 Column-wise (16-31)",
    colorbar=False,
    highlight_electrodes=highlights2,
    ax=axes[1],
    autoshow=False,
)

emg_multi.plot_grid_layout(
    2,
    title="Grid 3: 3×3 Row-wise (32-40)",
    colorbar=False,
    highlight_electrodes=highlights3,
    ax=axes[2],
    autoshow=False,
)

# Now show the complete figure with all three grids
plt.tight_layout()
plt.show()

# %%
# Let's also demonstrate what happens when validation fails:

print("\nDemonstrating validation with incorrect number of channels:")
try:
    # Create EMG data with the wrong number of channels (40 instead of 41)
    incorrect_emg_data = np.random.randn(40, 1000)

    # This would fail validation if implemented in EMGData as recommended
    EMGData(incorrect_emg_data, sampling_freq, grid_layouts=[grid1, grid2, grid3])
    print("Note: In the current version, this doesn't raise an error yet")
except ValueError as e:
    print(f"Validation error: {e}")

print("\nDemonstrating validation with out-of-bounds electrode indices:")
try:
    # Create a grid with an electrode index that's too high
    invalid_grid = grid3.copy()
    invalid_grid[0, 0] = 50  # This exceeds our 41 channels

    # This would fail validation if implemented in EMGData as recommended
    EMGData(emg_data_41ch, sampling_freq, grid_layouts=[grid1, grid2, invalid_grid])
    print("Note: In the current version, this doesn't raise an error yet")
except ValueError as e:
    print(f"Validation error: {e}")

# %%
# The validation logic described earlier would help ensure electrode indices correctly
# map to EMG channels, preventing hard-to-debug issues when working with grid_layouts.

# %%
# Custom irregular grid shapes
# ==========================
#
# Some electrode arrays have non-rectangular shapes. Let's create a custom
# grid layout for a circular electrode array.

# Create a circular-like pattern (8 electrodes in a ring)
circular_grid = np.full((3, 3), -1)  # Start with all -1 (no electrodes)
circular_grid[0, 1] = 0  # Top
circular_grid[1, 2] = 1  # Right
circular_grid[2, 1] = 2  # Bottom
circular_grid[1, 0] = 3  # Left
circular_grid[0, 0] = 4  # Top-left
circular_grid[0, 2] = 5  # Top-right
circular_grid[2, 2] = 6  # Bottom-right
circular_grid[2, 0] = 7  # Bottom-left

# Create EMG data for 8 channels
emg_data_8ch = np.random.randn(
    8, 1000
)  # Use higher amplitude data for better visibility
emg_circular = EMGData(emg_data_8ch, sampling_freq, grid_layouts=[circular_grid])

# Use the enhanced visualization for the circular grid
emg_circular.plot_grid_layout(
    0,
    title="Circular Electrode Arrangement",
    figsize=(8, 8),
    text_fontsize=12,
    colorbar=True,
    grid_alpha=0.5,
    text_color="yellow",
    highlight_electrodes=[0, 4],  # Highlight a couple of electrodes
    highlight_color="cyan",
)
