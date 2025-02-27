import numpy as np

from myoverse.datatypes import EMGData
from myoverse.datasets.filters.spatial import (
    DifferentialSpatialFilter,
    GridReshaper,
)


def test_emg_data_with_spatial_filters():
    """Simple integration test showing how the updated spatial filters work with EMGData."""
    
    # Create test EMG data
    emg_data = np.random.rand(1, 64, 100)  # 1 representation, 64 channels, 100 samples
    sampling_frequency = 1000  # 1000 Hz
    
    # Create EMGData object
    data = EMGData(emg_data, sampling_frequency)
    
    # Create an example grid layout for visualization (8x8 grid)
    rows, cols = 8, 8
    grid_layout = np.arange(64).reshape(rows, cols)
    data.grid_layouts = [grid_layout]  # Only one grid in this example
    
    # Define proper electrode setup
    electrode_setup = {
        "grid": {
            "shape": (1, 8, 8),  # 1 grid of 8x8
            "grid_type": "GR10MM0808",
            "electrodes_to_select": [(0, 64)],  # Select all 64 electrodes
        },
        "concatenate": False,
        "channel_selection": "all",
    }
    
    # Get the name of the input representation (typically "Input")
    input_representation = list(data.processed_representations)[0]
    print(f"Available representations: {list(data.processed_representations)}")
    print(f"Original EMG data shape: {data[input_representation].shape}")
    
    # First convert channels to grid format
    grid_reshaper = GridReshaper(
        operation="c2g",
        electrode_setup=electrode_setup,
        input_is_chunked=False
    )
    
    # Apply the grid reshaper filter first
    grid_rep = data.apply_filter(
        filter=grid_reshaper,
        representation_to_filter=input_representation,
        keep_representation_to_filter=True
    )
    
    print(f"\nGrid reshaper applied, created representation: {grid_rep}")
    print(f"Grid data shape: {data[grid_rep].shape}")
    
    # Now apply the identity filter to the grid data
    identity_filter = DifferentialSpatialFilter(
        filter_name="identity",
        input_is_chunked=False
    )
    
    # Apply the identity filter
    identity_rep = data.apply_filter(
        filter=identity_filter,
        representation_to_filter=grid_rep,
        keep_representation_to_filter=True
    )
    
    print(f"\nIdentity filter applied, created representation: {identity_rep}")
    print(f"Identity filter result shape: {data[identity_rep].shape}")
    
    # Apply Laplacian filter to demonstrate pipeline capability
    laplacian_filter = DifferentialSpatialFilter(
        filter_name="NDD", 
        input_is_chunked=False,
        is_output=True,
        name="Laplacian"
    )
    
    # Apply the Laplacian filter to the grid data
    laplacian_rep = data.apply_filter(
        filter=laplacian_filter,
        representation_to_filter=grid_rep,
        keep_representation_to_filter=True
    )
    
    print(f"\nLaplacian filter applied, created representation: {laplacian_rep}")
    print(f"Laplacian filter result shape: {data[laplacian_rep].shape}")
    
    # Print all available representations
    print("\nAll available representations:")
    for rep in data.processed_representations:
        print(f"- {rep}: {data[rep].shape}")
    
    # Plot the processing graph to visualize the filter pipeline
    print("\nPlotting the processing graph...")
    data.plot_graph()


if __name__ == "__main__":
    test_emg_data_with_spatial_filters() 