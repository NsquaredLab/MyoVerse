import unittest
import numpy as np

from myoverse.datasets.filters.spatial import (
    ElectrodeSelector,
    GridReshaper,
    DifferentialSpatialFilter,
    AveragingSpatialFilter,
    ChannelSelector,
    BraceletDifferential,
)


class TestSpatialFilters(unittest.TestCase):
    def setUp(self):
        """Setup test data for all test cases."""
        # Create test EMG data - non-chunked data
        self.emg_data = np.random.rand(1, 64, 100)  # 1 representation, 64 channels, 100 samples
        
        # Create chunked test EMG data
        self.emg_data_chunked = np.random.rand(1, 5, 64, 100)  # 1 representation, 5 chunks, 64 channels, 100 samples
        
        # Example electrode setup
        self.electrode_setup = {
            "grid": {
                "shape": (1, 8, 8),  # 1 grid of 8x8
                "grid_type": "GR10MM0808",
                "electrodes_to_select": [(0, 64)],  # Select all 64 electrodes
            },
            "concatenate": False,
            "differential": "LSD",
            "average": {
                "order": 3,
                "filter_direction": "longitudinal"
            },
            "channel_selection": "all",
        }

    def test_electrode_selector(self):
        """Test ElectrodeSelector with the new FilterBaseClass API."""
        # Test with electrodes_to_select parameter
        selector = ElectrodeSelector(electrodes_to_select=[0, 1, 2, 3, 4])
        data = np.random.rand(1, 10, 100)  # 1 representation, 10 channels, 100 samples
        
        # Test non-chunked data
        selector.input_is_chunked = False
        result = selector(data)
        self.assertEqual(result.shape, (1, 5, 100))  # Should select 5 channels
        
        # Test chunked data
        selector.input_is_chunked = True
        data_chunked = np.random.rand(1, 3, 10, 100)  # 3 chunks
        result = selector(data_chunked)
        self.assertEqual(result.shape, (1, 3, 5, 100))  # Should select 5 channels from each chunk
        
        # Test with electrode_setup parameter
        selector = ElectrodeSelector(electrode_setup=self.electrode_setup)
        result = selector(self.emg_data)
        self.assertEqual(result.shape, self.emg_data.shape)  # Should select all channels
        
        # Test auto-detection of chunked input
        selector = ElectrodeSelector(electrode_setup=self.electrode_setup, input_is_chunked=None)
        selector.input_is_chunked = True  # This would be set by the _Data class based on the input
        result = selector(self.emg_data_chunked)
        self.assertEqual(result.shape, self.emg_data_chunked.shape)

    def test_grid_reshaper(self):
        """Test GridReshaper with the new FilterBaseClass API."""
        # Test channels to grid (c2g) operation
        reshaper_c2g = GridReshaper(
            operation="c2g",
            electrode_setup=self.electrode_setup,
            input_is_chunked=False
        )
        result = reshaper_c2g(self.emg_data)
        self.assertEqual(result.shape, (1, 1, 8, 8, 100))  # Should reshape to 1 grid of 8x8
        
        # Test grid to channels (g2c) operation
        # First we need grid-shaped data
        grid_data = reshaper_c2g(self.emg_data)
        
        reshaper_g2c = GridReshaper(
            operation="g2c",
            electrode_setup=self.electrode_setup,
            input_is_chunked=False
        )
        result = reshaper_g2c(grid_data)
        self.assertEqual(result.shape, (1, 64, 100))  # Should reshape back to 64 channels
        
        # Test concatenation operation
        # Create a setup with multiple grids
        multi_grid_setup = self.electrode_setup.copy()
        multi_grid_setup["grid"] = multi_grid_setup["grid"].copy()
        multi_grid_setup["grid"]["shape"] = (2, 8, 8)  # 2 grids of 8x8
        
        # Create test data with 2 grids
        multi_grid_data = np.random.rand(1, 2, 8, 8, 100)
        
        reshaper_concat = GridReshaper(
            operation="concat",
            electrode_setup=multi_grid_setup,
            input_is_chunked=False
        )
        result = reshaper_concat(multi_grid_data)
        self.assertEqual(result.shape, (1, 1, 8, 16, 100))  # Should concatenate to 8x16
        
        # Test with axis parameter
        reshaper_concat_row = GridReshaper(
            operation="concat",
            electrode_setup=multi_grid_setup,
            input_is_chunked=False,
            axis="row"
        )
        result = reshaper_concat_row(multi_grid_data)
        self.assertEqual(result.shape, (1, 1, 16, 8, 100))  # Should concatenate to 16x8

    def test_differential_spatial_filter(self):
        """Test DifferentialSpatialFilter with the new FilterBaseClass API."""
        # Create a grid reshaper to prepare grid data
        reshaper = GridReshaper(
            operation="c2g",
            electrode_setup=self.electrode_setup,
            input_is_chunked=False
        )
        grid_data = reshaper(self.emg_data)
        
        # Test with different filter types
        filter_types = ["identity", "LSD", "LDD", "TSD", "TDD", "NDD", "IB2", "IR"]
        
        for filter_type in filter_types:
            diff_filter = DifferentialSpatialFilter(
                filter_name=filter_type,
                input_is_chunked=False
            )
            result = diff_filter(grid_data)
            
            # Check that output shape is changed appropriately based on filter kernel size
            if filter_type == "identity":
                self.assertEqual(result.shape[:4], (1, 1, 8, 8))  # No change in grid dimensions
            elif filter_type in ["LSD", "LDD"]:
                self.assertLess(result.shape[2], 8)  # Reduced rows
                self.assertEqual(result.shape[3], 8)  # Same columns
            elif filter_type in ["TSD", "TDD"]:
                self.assertEqual(result.shape[2], 8)  # Same rows
                self.assertLess(result.shape[3], 8)  # Reduced columns
            elif filter_type in ["NDD", "IB2", "IR"]:
                self.assertLess(result.shape[2], 8)  # Reduced rows
                self.assertLess(result.shape[3], 8)  # Reduced columns
    
    def test_averaging_spatial_filter(self):
        """Test AveragingSpatialFilter with the new FilterBaseClass API."""
        # Create a grid reshaper to prepare grid data
        reshaper = GridReshaper(
            operation="c2g",
            electrode_setup=self.electrode_setup,
            input_is_chunked=False
        )
        grid_data = reshaper(self.emg_data)
        
        # Test longitudinal averaging
        avg_filter_long = AveragingSpatialFilter(
            order=3,
            filter_direction="longitudinal",
            input_is_chunked=False
        )
        result = avg_filter_long(grid_data)
        self.assertEqual(result.shape, (1, 1, 6, 8, 100))  # Reduced rows by order-1, same columns
        
        # Test transversal averaging
        avg_filter_trans = AveragingSpatialFilter(
            order=3,
            filter_direction="transversal",
            input_is_chunked=False
        )
        result = avg_filter_trans(grid_data)
        self.assertEqual(result.shape, (1, 1, 8, 6, 100))  # Same rows, reduced columns by order-1

    def test_channel_selector(self):
        """Test ChannelSelector with the new FilterBaseClass API."""
        # Create a grid reshaper to prepare grid data
        reshaper = GridReshaper(
            operation="c2g",
            electrode_setup=self.electrode_setup,
            input_is_chunked=False
        )
        grid_data = reshaper(self.emg_data)
        
        # Test with specific grid positions
        grid_positions = [(0, 0), (1, 1), (2, 2), (3, 3)]
        selector = ChannelSelector(
            grid_position=grid_positions,
            input_is_chunked=False
        )
        result = selector(grid_data)
        self.assertEqual(result.shape, (1, 4, 100))  # Selected 4 channels
        
        # Test with "all" selection from electrode_setup
        selector = ChannelSelector(
            electrode_setup=self.electrode_setup,
            input_is_chunked=False
        )
        result = selector(grid_data)
        self.assertEqual(result.shape, grid_data.shape)  # Should remain unchanged

    def test_bracelet_differential(self):
        """Test BraceletDifferential with the new FilterBaseClass API."""
        # Create bracelet EMG data (2 rows, 16 columns)
        bracelet_data = np.random.rand(1, 32, 100)  # 32 channels (2x16 grid)
        
        # Reshape to grid format (2x16)
        bracelet_grid_data = bracelet_data.reshape(1, 1, 2, 16, 100)
        
        # Apply bracelet differential filter
        bracelet_filter = BraceletDifferential(input_is_chunked=False)
        result = bracelet_filter(bracelet_grid_data)
        
        # Output should be 32 channels
        self.assertEqual(result.shape, (1, 32, 100))
        
        # Test with chunked data
        bracelet_data_chunked = np.random.rand(1, 5, 32, 100)  # 5 chunks
        bracelet_grid_data_chunked = bracelet_data_chunked.reshape(1, 5, 2, 16, 100)
        
        bracelet_filter = BraceletDifferential(input_is_chunked=True)
        result = bracelet_filter(bracelet_grid_data_chunked)
        
        # Output should maintain chunk dimension
        self.assertEqual(result.shape[1], 5)
        self.assertEqual(result.shape[2], 32)

    def test_filter_pipeline(self):
        """Test a complete filter pipeline with all spatial filters."""
        # Create a pipeline that applies all filters in sequence
        
        # 1. Select electrodes
        selector = ElectrodeSelector(electrode_setup=self.electrode_setup)
        
        # 2. Reshape to grid
        reshaper_c2g = GridReshaper(operation="c2g", electrode_setup=self.electrode_setup)
        
        # 3. Apply averaging filter
        avg_filter = AveragingSpatialFilter(
            order=3,
            filter_direction="longitudinal"
        )
        
        # 4. Apply differential filter
        diff_filter = DifferentialSpatialFilter(filter_name="LSD")
        
        # 5. Reshape back to channels
        reshaper_g2c = GridReshaper(
            operation="g2c", 
            electrode_setup=self.electrode_setup,
            concatenate=False
        )
        
        # Apply the pipeline
        result = self.emg_data
        result = selector(result)
        result = reshaper_c2g(result)
        result = avg_filter(result)
        result = diff_filter(result)
        result = reshaper_g2c(result)
        
        # Final result should be in channel format (fewer channels due to filtering)
        self.assertEqual(len(result.shape), 3)  # (1, channels, samples)
        self.assertLess(result.shape[1], 64)  # Fewer channels due to filtering
        self.assertEqual(result.shape[2], 100)  # Same number of samples

    def test_filter_auto_detection(self):
        """Test that filters can detect chunked vs non-chunked data automatically."""
        # Create a pipeline with auto-detection
        selector = ElectrodeSelector(electrode_setup=self.electrode_setup, input_is_chunked=None)
        reshaper = GridReshaper(operation="c2g", electrode_setup=self.electrode_setup, input_is_chunked=None)
        diff_filter = DifferentialSpatialFilter(filter_name="LSD", input_is_chunked=None)
        
        # Test with non-chunked data
        selector.input_is_chunked = False  # This would be set by _Data class
        reshaper.input_is_chunked = False
        diff_filter.input_is_chunked = False
        
        result = selector(self.emg_data)
        result = reshaper(result)
        result = diff_filter(result)
        
        # Should process correctly
        self.assertEqual(len(result.shape), 5)  # (1, 1, rows, cols, samples)
        
        # Test with chunked data
        selector.input_is_chunked = True  # This would be set by _Data class
        reshaper.input_is_chunked = True
        diff_filter.input_is_chunked = True
        
        result = selector(self.emg_data_chunked)
        result = reshaper(result)
        result = diff_filter(result)
        
        # Should process correctly with chunks
        self.assertEqual(len(result.shape), 6)  # (1, 5, 1, rows, cols, samples)


if __name__ == "__main__":
    unittest.main() 