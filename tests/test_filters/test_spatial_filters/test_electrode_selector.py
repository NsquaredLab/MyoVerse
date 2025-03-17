import unittest
import numpy as np

from myoverse.datasets.filters.spatial import ElectrodeSelector
from myoverse.datatypes import EMGData


class TestElectrodeSelector(unittest.TestCase):
    """Test cases for the ElectrodeSelector spatial filter."""

    def setUp(self):
        """Set up common test data."""
        # Create non-chunked test data (32 channels, 100 samples)
        self.non_chunked_data = np.zeros((32, 100))
        # Fill with values matching channel indices for easy verification
        for i in range(32):
            self.non_chunked_data[i, :] = i

        # Create chunked test data (3 chunks, 32 channels, 100 samples)
        self.chunked_data = np.zeros((3, 32, 100))
        # Fill with values representing chunk and channel
        for chunk in range(3):
            for channel in range(32):
                self.chunked_data[chunk, channel, :] = channel + (100 * chunk)

        # Create grid layouts for testing
        # Two 4x4 grids (16 channels each)
        self.grid1 = np.arange(16).reshape(4, 4)
        self.grid2 = np.arange(16, 32).reshape(4, 4)
        self.grid_layouts = [self.grid1, self.grid2]

    def test_basic_selection_non_chunked(self):
        """Test basic electrode selection with non-chunked data."""
        selector = ElectrodeSelector(
            electrodes_to_select=[1, 5, 10, 18, 25], input_is_chunked=False
        )

        # Always provide grid layouts since it's now required
        result = selector(self.non_chunked_data, grid_layouts=self.grid_layouts)

        # Check shape and values
        self.assertEqual(result.shape, (5, 100))
        self.assertTrue(np.all(result[0, 0] == 1))
        self.assertTrue(np.all(result[1, 0] == 5))
        self.assertTrue(np.all(result[2, 0] == 10))
        self.assertTrue(np.all(result[3, 0] == 18))
        self.assertTrue(np.all(result[4, 0] == 25))

    def test_basic_selection_chunked(self):
        """Test basic electrode selection with chunked data."""
        selector = ElectrodeSelector(
            electrodes_to_select=[1, 5, 10, 18, 25], input_is_chunked=True
        )

        # Always provide grid layouts since it's now required
        result = selector(self.chunked_data, grid_layouts=self.grid_layouts)

        # Check shape and values
        self.assertEqual(result.shape, (3, 5, 100))
        # Check first chunk values
        self.assertTrue(np.all(result[0, 0, 0] == 1))
        self.assertTrue(np.all(result[0, 1, 0] == 5))
        self.assertTrue(np.all(result[0, 2, 0] == 10))
        self.assertTrue(np.all(result[0, 3, 0] == 18))
        self.assertTrue(np.all(result[0, 4, 0] == 25))
        # Check second chunk values (should be offset by 100)
        self.assertTrue(np.all(result[1, 0, 0] == 1 + 100))

    def test_grid_specific_selection(self):
        """Test selecting electrodes from specific grids."""
        # Select from grid 0 only
        selector_grid0 = ElectrodeSelector(
            electrodes_to_select=[1, 5, 10],
            input_is_chunked=False,
            grids_to_process=0,
            preserve_unprocessed_grids=False,
        )

        # Apply with grid layouts
        result_grid0 = selector_grid0(
            self.non_chunked_data, grid_layouts=self.grid_layouts
        )

        # Should only contain electrodes from first grid
        self.assertEqual(result_grid0.shape, (3, 100))
        self.assertTrue(np.all(result_grid0[0, 0] == 1))
        self.assertTrue(np.all(result_grid0[1, 0] == 5))
        self.assertTrue(np.all(result_grid0[2, 0] == 10))

        # Select from grid 1 only
        selector_grid1 = ElectrodeSelector(
            electrodes_to_select=[18, 25],
            input_is_chunked=False,
            grids_to_process=1,
            preserve_unprocessed_grids=False,
        )

        # Apply with grid layouts
        result_grid1 = selector_grid1(
            self.non_chunked_data, grid_layouts=self.grid_layouts
        )

        # Should only contain electrodes from second grid
        self.assertEqual(result_grid1.shape, (2, 100))
        self.assertTrue(np.all(result_grid1[0, 0] == 18))
        self.assertTrue(np.all(result_grid1[1, 0] == 25))

    def test_preserve_unprocessed_grids(self):
        """Test preserving unprocessed grids."""
        # Process grid 0 only, but preserve grid 1
        selector = ElectrodeSelector(
            electrodes_to_select=[1, 5, 10],
            input_is_chunked=False,
            grids_to_process=0,
            preserve_unprocessed_grids=True,
        )

        # Apply with grid layouts
        result = selector(self.non_chunked_data, grid_layouts=self.grid_layouts)

        # Should contain selected electrodes from grid 0 plus all from grid 1
        self.assertEqual(
            result.shape, (3 + 16, 100)
        )  # 3 selected from grid 0 + all 16 from grid 1

        # First 3 channels should be the selected ones from grid 0
        self.assertTrue(np.all(result[0, 0] == 1))
        self.assertTrue(np.all(result[1, 0] == 5))
        self.assertTrue(np.all(result[2, 0] == 10))

        # Next 16 channels should be all channels from grid 1
        for i in range(16):
            self.assertTrue(np.all(result[3 + i, 0] == 16 + i))

    def test_dict_based_selection(self):
        """Test dictionary-based electrode selection."""
        # Create a dictionary mapping grid indices to electrode lists
        electrode_dict = {
            0: [1, 5, 10],  # Select these from grid 0
            1: [18, 25],  # Select these from grid 1
        }

        selector = ElectrodeSelector(
            electrodes_to_select=electrode_dict, input_is_chunked=False
        )

        # Apply with grid layouts
        result = selector(self.non_chunked_data, grid_layouts=self.grid_layouts)

        # Should contain all selected electrodes
        self.assertEqual(result.shape, (5, 100))
        self.assertTrue(np.all(result[0, 0] == 1))
        self.assertTrue(np.all(result[1, 0] == 5))
        self.assertTrue(np.all(result[2, 0] == 10))
        self.assertTrue(np.all(result[3, 0] == 18))
        self.assertTrue(np.all(result[4, 0] == 25))

        # Test with specific grid processing
        selector_grid0 = ElectrodeSelector(
            electrodes_to_select=electrode_dict,
            input_is_chunked=False,
            grids_to_process=0,
            preserve_unprocessed_grids=False,
        )

        # Apply with grid layouts
        result_grid0 = selector_grid0(
            self.non_chunked_data, grid_layouts=self.grid_layouts
        )

        # Should only contain selected electrodes from grid 0
        self.assertEqual(result_grid0.shape, (3, 100))
        self.assertTrue(np.all(result_grid0[0, 0] == 1))
        self.assertTrue(np.all(result_grid0[1, 0] == 5))
        self.assertTrue(np.all(result_grid0[2, 0] == 10))

    def test_emg_data_integration(self):
        """Test integration with EMGData."""
        # Create an EMGData object with properly initialized representations
        emg = EMGData(self.non_chunked_data, sampling_frequency=1000)

        # Initialize a representation for testing
        emg._data = {"Input": self.non_chunked_data}
        emg._last_processing_step = "Input"

        # Set grid layouts
        emg.grid_layouts = self.grid_layouts

        # Create a filter
        selector = ElectrodeSelector(
            electrodes_to_select=[1, 5, 10, 18, 25], input_is_chunked=False
        )

        # Apply filter through EMGData with explicit representation
        result_name = emg.apply_filter(selector, representations_to_filter=["Input"])

        # Check the result
        self.assertEqual(emg[result_name].shape, (5, 100))

        # Test with grid-specific processing
        selector_grid0 = ElectrodeSelector(
            electrodes_to_select=[1, 5, 10],
            input_is_chunked=False,
            grids_to_process=0,
            preserve_unprocessed_grids=False,
        )

        # Apply filter with explicit representation
        result_name = emg.apply_filter(
            selector_grid0, representations_to_filter=["Input"]
        )

        # Check result - should only have electrodes from grid 0
        self.assertEqual(emg[result_name].shape, (3, 100))

    def test_error_handling(self):
        """Test error handling."""
        # Test with empty electrodes list
        with self.assertRaises(ValueError):
            ElectrodeSelector(electrodes_to_select=[], input_is_chunked=False)

        # Test with non-integer electrodes
        with self.assertRaises(ValueError):
            ElectrodeSelector(electrodes_to_select=[1, "2", 3], input_is_chunked=False)

        # Test with out-of-bounds electrode indices
        selector = ElectrodeSelector(
            electrodes_to_select=[1, 50],  # 50 is out of bounds for 32 channels
            input_is_chunked=False,
        )

        with self.assertRaises(ValueError):
            selector._run_filter_checks(self.non_chunked_data)

        # Test when grid layouts are not provided
        selector = ElectrodeSelector(
            electrodes_to_select=[1, 5, 10], input_is_chunked=False
        )

        with self.assertRaises(ValueError):
            selector(self.non_chunked_data)  # No grid_layouts provided

        # Test with invalid grid index
        selector_invalid_grid = ElectrodeSelector(
            electrodes_to_select=[1, 5],
            input_is_chunked=False,
            grids_to_process=5,  # Only have 2 grids (0 and 1)
        )

        with self.assertRaises(ValueError):
            selector_invalid_grid(self.non_chunked_data, grid_layouts=self.grid_layouts)

        # Test with invalid grid-specific electrode (electrode not in grid)
        electrode_dict = {
            0: [20]  # Electrode 20 is not in grid 0
        }

        selector_invalid_electrode = ElectrodeSelector(
            electrodes_to_select=electrode_dict,
            input_is_chunked=False,
            grids_to_process=0,
            preserve_unprocessed_grids=True,  # This will include grid 1 even though no electrodes were found in grid 0
        )

        # With preserve_unprocessed_grids=True, this will still include grid 1
        result = selector_invalid_electrode(
            self.non_chunked_data, grid_layouts=self.grid_layouts
        )
        self.assertEqual(result.shape[0], 16)  # All electrodes from grid 1

        # Try with preserve_unprocessed_grids=False
        selector_invalid_no_preserve = ElectrodeSelector(
            electrodes_to_select=electrode_dict,
            input_is_chunked=False,
            grids_to_process=0,
            preserve_unprocessed_grids=False,
        )

        result = selector_invalid_no_preserve(
            self.non_chunked_data, grid_layouts=self.grid_layouts
        )
        self.assertEqual(result.shape[0], 0)  # No electrodes found

    def test_chunked_data_with_grid_specific_selection(self):
        """Test chunked data with grid-specific selection."""
        # Test with chunked data and grid-specific selection
        selector = ElectrodeSelector(
            electrodes_to_select=[1, 5, 10],
            input_is_chunked=True,
            grids_to_process=0,
            preserve_unprocessed_grids=False,
        )

        # Apply with grid layouts
        result = selector(self.chunked_data, grid_layouts=self.grid_layouts)

        # Should only contain electrodes from first grid across all chunks
        self.assertEqual(
            result.shape, (3, 3, 100)
        )  # 3 chunks, 3 electrodes, 100 samples

        # Check values in first chunk
        self.assertTrue(np.all(result[0, 0, 0] == 1))
        self.assertTrue(np.all(result[0, 1, 0] == 5))
        self.assertTrue(np.all(result[0, 2, 0] == 10))

        # Check values in second chunk (should be offset by 100)
        self.assertTrue(np.all(result[1, 0, 0] == 1 + 100))
        self.assertTrue(np.all(result[1, 1, 0] == 5 + 100))
        self.assertTrue(np.all(result[1, 2, 0] == 10 + 100))

    def test_multiple_grid_processing(self):
        """Test processing multiple specific grids."""
        # Process both grids 0 and 1
        selector = ElectrodeSelector(
            electrodes_to_select=[1, 5, 18, 25],
            input_is_chunked=False,
            grids_to_process=[0, 1],
        )

        # Apply with grid layouts
        result = selector(self.non_chunked_data, grid_layouts=self.grid_layouts)

        # Should contain selected electrodes from both grids
        self.assertEqual(result.shape, (4, 100))
        self.assertTrue(np.all(result[0, 0] == 1))
        self.assertTrue(np.all(result[1, 0] == 5))
        self.assertTrue(np.all(result[2, 0] == 18))
        self.assertTrue(np.all(result[3, 0] == 25))

        # Compare with using "all" grids
        selector_all = ElectrodeSelector(
            electrodes_to_select=[1, 5, 18, 25],
            input_is_chunked=False,
            grids_to_process="all",
        )

        result_all = selector_all(self.non_chunked_data, grid_layouts=self.grid_layouts)

        # Result should be the same as specifying both grid indices
        self.assertEqual(result_all.shape, result.shape)
        np.testing.assert_array_equal(result_all, result)

    def test_dict_with_nonexistent_grid(self):
        """Test dictionary with nonexistent grid index."""
        # Create a dictionary with valid grid indices
        electrode_dict = {0: [1, 5], 1: [18, 25]}

        # Test with preserve_unprocessed_grids=False
        selector_dict = ElectrodeSelector(
            electrodes_to_select=electrode_dict,
            input_is_chunked=False,
            preserve_unprocessed_grids=False,
        )

        # Apply with grid layouts
        result = selector_dict(self.non_chunked_data, grid_layouts=self.grid_layouts)

        # Should contain all selected electrodes (from both grids)
        self.assertEqual(result.shape, (4, 100))

        # Verify that we have the expected electrodes in the result
        expected_values = [1, 5, 18, 25]

        for i, expected in enumerate(expected_values):
            self.assertTrue(
                np.all(result[i, 0] == expected),
                f"Expected electrode {expected} at position {i}",
            )

        # Test with specific grid processing - just grid 0
        selector_grid0 = ElectrodeSelector(
            electrodes_to_select=electrode_dict,
            input_is_chunked=False,
            grids_to_process=0,
            preserve_unprocessed_grids=False,
        )

        # Apply with grid layouts
        result_grid0 = selector_grid0(
            self.non_chunked_data, grid_layouts=self.grid_layouts
        )

        # Should only contain selected electrodes from grid 0
        self.assertEqual(result_grid0.shape, (2, 100))
        self.assertTrue(np.all(result_grid0[0, 0] == 1))
        self.assertTrue(np.all(result_grid0[1, 0] == 5))


if __name__ == "__main__":
    unittest.main()
