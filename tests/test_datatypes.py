import numpy as np
import pytest
import networkx as nx
from matplotlib import pyplot as plt
from copy import deepcopy

from myoverse.datatypes import EMGData, KinematicsData, VirtualHandKinematics, create_grid_layout
from myoverse.datasets.filters._template import FilterBaseClass


class MockFilter(FilterBaseClass):
    """Mock filter for testing purposes."""
    
    def __init__(self, output=False, name=None, input_is_chunked=None):
        self.is_output = output
        # Initialize with the correct allowed_input_type
        allowed_input_type = "both"  # Allow both chunked and non-chunked input
        super().__init__(
            name=name, 
            input_is_chunked=input_is_chunked,
            allowed_input_type=allowed_input_type,
            is_output=output
        )
        
    def _filter(self, input_array):
        # Just return the input array multiplied by 2
        return input_array * 2


class TestEMGData:
    """Test cases for EMGData class."""
    
    @pytest.fixture
    def emg_data_2d(self):
        """Create a 2D EMG data fixture."""
        # 8 channels, 1000 samples
        return np.random.rand(8, 1000)
    
    @pytest.fixture
    def emg_data_3d(self):
        """Create a 3D (chunked) EMG data fixture."""
        # 10 chunks, 8 channels, 100 samples
        return np.random.rand(10, 8, 100)
    
    @pytest.fixture
    def sampling_frequency(self):
        """Return a sampling frequency for testing."""
        return 2000.0
    
    def test_init_2d(self, emg_data_2d, sampling_frequency):
        """Test initialization with 2D data."""
        emg = EMGData(emg_data_2d, sampling_frequency)
        assert emg.sampling_frequency == sampling_frequency
        assert np.array_equal(emg.input_data, emg_data_2d)
        assert not emg.is_chunked["Input"]
    
    def test_init_3d(self, emg_data_3d, sampling_frequency):
        """Test initialization with 3D data."""
        emg = EMGData(emg_data_3d, sampling_frequency)
        assert emg.sampling_frequency == sampling_frequency
        assert np.array_equal(emg.input_data, emg_data_3d)
        assert emg.is_chunked["Input"]
    
    def test_init_invalid_dim(self, sampling_frequency):
        """Test initialization with invalid dimensionality."""
        # 4D data (invalid)
        invalid_data = np.random.rand(2, 3, 4, 5)
        with pytest.raises(ValueError):
            EMGData(invalid_data, sampling_frequency)
    
    def test_init_invalid_frequency(self, emg_data_2d):
        """Test initialization with invalid sampling frequency."""
        with pytest.raises(ValueError):
            EMGData(emg_data_2d, 0)
        
        with pytest.raises(ValueError):
            EMGData(emg_data_2d, -100)
    
    def test_apply_filter(self, emg_data_2d, sampling_frequency):
        """Test applying a filter to EMG data."""
        emg = EMGData(emg_data_2d, sampling_frequency)
        
        # Create a mock filter
        mock_filter = MockFilter(name="MockFilter1")
        
        # Apply the filter
        result_name = emg.apply_filter(mock_filter, "Input")
        
        # Check results
        assert result_name == "MockFilter1"
        assert np.array_equal(emg["MockFilter1"], emg_data_2d * 2)
        
        # Test with setting as output
        output_filter = MockFilter(output=True, name="OutputFilter")
        emg.apply_filter(output_filter, "MockFilter1")
        
        # Check that the filter is marked as output
        assert "OutputFilter" in emg.output_representations
    
    def test_apply_filter_sequence(self, emg_data_2d, sampling_frequency):
        """Test applying a sequence of filters."""
        emg = EMGData(emg_data_2d, sampling_frequency)
    
        # Create a sequence of filters
        filters = [
            MockFilter(name=f"MockFilter{i}")
            for i in range(1, 4)
        ]
        filters[-1].is_output = True  # Mark the last filter as output
    
        # Apply the filter sequence
        emg.apply_filter_sequence(filters, "Input")
    
        # Check results - each filter doubles the values
        expected_result = emg_data_2d * (2 ** 3)  # 3 filters, each doubling
        assert np.array_equal(emg["MockFilter3"], expected_result)
        assert "MockFilter3" in emg.output_representations
    
        # Now check if we can explicitly delete the intermediate steps
        # Let's get the current keys in the _data dictionary
        for filter_name in ["MockFilter1", "MockFilter2"]:
            if filter_name in emg._data:
                emg.delete_data(filter_name)
        
        # Now check that only Input and MockFilter3 remain as numpy arrays
        assert isinstance(emg._data["Input"], np.ndarray)
        assert isinstance(emg._data["MockFilter3"], np.ndarray)
        
        # The intermediate filters should be strings (shape representations)
        assert isinstance(emg._data["MockFilter1"], str)
        assert isinstance(emg._data["MockFilter2"], str)
    
    def test_delete_data(self, emg_data_2d, sampling_frequency):
        """Test deleting data from a representation."""
        emg = EMGData(emg_data_2d, sampling_frequency)
        
        # Apply a filter
        filter1 = MockFilter(name="Filter1")
        emg.apply_filter(filter1, "Input")
        
        # Check data is present
        assert isinstance(emg._data["Filter1"], np.ndarray)
        
        # Delete the data
        emg.delete_data("Filter1")
        
        # Check data is now a string representation
        assert isinstance(emg._data["Filter1"], str)
        
        # Accessing the representation should regenerate the data
        regenerated = emg["Filter1"]
        assert isinstance(regenerated, np.ndarray)
        assert np.array_equal(regenerated, emg_data_2d * 2)
    
    def test_chunked_property(self, emg_data_2d, emg_data_3d, sampling_frequency):
        """Test is_chunked property."""
        emg_2d = EMGData(emg_data_2d, sampling_frequency)
        emg_3d = EMGData(emg_data_3d, sampling_frequency)
        
        # Check initial state
        assert not emg_2d.is_chunked["Input"]
        assert emg_3d.is_chunked["Input"]
        
        # Apply a filter and check the result maintains chunked state
        filter1 = MockFilter(name="Filter1")
        emg_2d.apply_filter(filter1, "Input")
        emg_3d.apply_filter(filter1, "Input")
        
        assert not emg_2d.is_chunked["Filter1"]
        assert emg_3d.is_chunked["Filter1"]
    
    def test_plot_graph(self, emg_data_2d, sampling_frequency, monkeypatch):
        """Test plot_graph method."""
        # Mock plt.show to avoid displaying the plot during tests
        monkeypatch.setattr(plt, "show", lambda: None)
        
        emg = EMGData(emg_data_2d, sampling_frequency)
        
        # Apply some filters to create a graph
        filter1 = MockFilter(name="Filter1")
        filter2 = MockFilter(name="Filter2", output=True)
        
        emg.apply_filter(filter1, "Input")
        emg.apply_filter(filter2, "Filter1")
        
        # This should not raise any errors
        emg.plot_graph()


class TestKinematicsData:
    """Test cases for KinematicsData class."""
    
    @pytest.fixture
    def kinematics_data_3d(self):
        """Create a 3D kinematics data fixture."""
        # 21 joints, 3 coordinates (x,y,z), 1000 samples
        return np.random.rand(21, 3, 1000)
    
    @pytest.fixture
    def kinematics_data_4d(self):
        """Create a 4D (chunked) kinematics data fixture."""
        # 10 chunks, 21 joints, 3 coordinates, 100 samples
        return np.random.rand(10, 21, 3, 100)
    
    @pytest.fixture
    def sampling_frequency(self):
        """Return a sampling frequency for testing."""
        return 60.0
    
    def test_init_3d(self, kinematics_data_3d, sampling_frequency):
        """Test initialization with 3D data."""
        kin = KinematicsData(kinematics_data_3d, sampling_frequency)
        assert kin.sampling_frequency == sampling_frequency
        assert np.array_equal(kin.input_data, kinematics_data_3d)
        assert not kin.is_chunked["Input"]
    
    def test_init_4d(self, kinematics_data_4d, sampling_frequency):
        """Test initialization with 4D data."""
        kin = KinematicsData(kinematics_data_4d, sampling_frequency)
        assert kin.sampling_frequency == sampling_frequency
        assert np.array_equal(kin.input_data, kinematics_data_4d)
        assert kin.is_chunked["Input"]
    
    def test_init_invalid_dim(self, sampling_frequency):
        """Test initialization with invalid dimensionality."""
        # 2D data (invalid)
        invalid_data = np.random.rand(10, 100)
        with pytest.raises(ValueError):
            KinematicsData(invalid_data, sampling_frequency)
    
    def test_plot(self, kinematics_data_3d, sampling_frequency, monkeypatch):
        """Test plot method."""
        # Mock plt.show to avoid displaying the plot during tests
        monkeypatch.setattr(plt, "show", lambda: None)
        
        kin = KinematicsData(kinematics_data_3d, sampling_frequency)
        
        # This should not raise any errors
        kin.plot("Input", nr_of_fingers=5)
        
        # Test with wrist_included=False
        kin.plot("Input", nr_of_fingers=5, wrist_included=False)


class TestVirtualHandKinematics:
    """Test cases for VirtualHandKinematics class."""
    
    @pytest.fixture
    def sample_vhk_data(self):
        # Create sample data for VirtualHandKinematics
        data = np.random.randn(9, 100)  # 9 DOFs, 100 samples
        return VirtualHandKinematics(data, 100.0)
        
    def test_init_valid(self, sample_vhk_data):
        """Test that initialization works with valid data."""
        assert sample_vhk_data.sampling_frequency == 100.0
        assert sample_vhk_data.input_data.shape == (9, 100)
        
    def test_init_invalid_dims(self):
        """Test that initialization fails with invalid dimensions."""
        # 1D data (invalid)
        with pytest.raises(ValueError):
            VirtualHandKinematics(np.random.randn(100), 100.0)
            
        # 4D data (invalid)
        with pytest.raises(ValueError):
            VirtualHandKinematics(np.random.randn(2, 2, 2, 2), 100.0)
            
        # This one might not raise an error because the class only checks ndim, not specific shape
        # But we can verify it raises an error during plot() when it expects 9 DOFs
        vhk = VirtualHandKinematics(np.random.randn(10, 100), 100.0)
        with pytest.raises(ValueError, match="Expected 9 degrees of freedom"):
            vhk.plot("Input")
        
    def test_init_invalid_frequency(self):
        """Test that initialization fails with invalid frequency."""
        with pytest.raises(ValueError):
            VirtualHandKinematics(np.random.randn(9, 100), -1.0)
        
    def test_plot(self, sample_vhk_data, monkeypatch):
        """Test that plot method works without errors."""
        # Create a more comprehensive mock for matplotlib components
        class MockLine:
            def __init__(self, *args, **kwargs):
                pass
        
        class MockAxes:
            def __init__(self):
                self.lines = []
                self.title = None
                self.xlabel = None
                self.ylabel = None
                
            def set_title(self, title):
                self.title = title
                
            def plot(self, *args, **kwargs):
                line = MockLine()
                self.lines.append(line)
                return [line]
                
            def legend(self):
                pass
                
            def set_xlabel(self, label):
                self.xlabel = label
                
            def set_ylabel(self, label):
                self.ylabel = label
                
            def grid(self, *args, **kwargs):
                # Properly handle grid(True) or grid(visible=True)
                pass
                
        class MockFigure:
            def __init__(self, *args, **kwargs):
                self.axes = []
                
            def add_subplot(self, *args, **kwargs):
                ax = MockAxes()
                self.axes.append(ax)
                return ax
                
        # Create a mock for tight_layout
        def mock_tight_layout():
            pass
            
        # Mock matplotlib components
        monkeypatch.setattr(plt, 'figure', MockFigure)
        monkeypatch.setattr(plt, 'show', lambda: None)
        monkeypatch.setattr(plt, 'tight_layout', mock_tight_layout)

        # Test plotting with different options
        sample_vhk_data.plot("Input", visualize_wrist=True)
        sample_vhk_data.plot("Input", nr_of_fingers=3, visualize_wrist=False)
        
        # Test with chunked data
        chunked_data = np.random.randn(5, 9, 50)  # 5 chunks, 9 DOFs, 50 samples
        chunked_vhk = VirtualHandKinematics(chunked_data, 100.0)
        chunked_vhk.plot("Input")


class TestDataTypesCopy:
    """Test copying behavior of data classes."""
    
    def test_emg_data_copy(self):
        """Test copying EMGData objects."""
        data = np.random.rand(8, 1000)
        emg = EMGData(data, 2000.0)
        
        # Apply a filter
        filter1 = MockFilter(name="Filter1")
        emg.apply_filter(filter1, "Input")
        
        # Copy the object
        emg_copy = deepcopy(emg)
        
        # Check that the copy has the same data
        assert np.array_equal(emg.input_data, emg_copy.input_data)
        assert np.array_equal(emg["Filter1"], emg_copy["Filter1"])
        
        # Modify the original
        filter2 = MockFilter(name="Filter2")
        emg.apply_filter(filter2, "Filter1")
        
        # Check that the copy is unaffected
        assert "Filter2" in emg._data
        assert "Filter2" not in emg_copy._data
    
    def test_kinematics_data_copy(self):
        """Test copying KinematicsData objects."""
        data = np.random.rand(21, 3, 1000)
        kin = KinematicsData(data, 60.0)
        
        # Apply a filter
        filter1 = MockFilter(name="Filter1")
        kin.apply_filter(filter1, "Input")
        
        # Copy the object
        kin_copy = deepcopy(kin)
        
        # Check that the copy has the same data
        assert np.array_equal(kin.input_data, kin_copy.input_data)
        assert np.array_equal(kin["Filter1"], kin_copy["Filter1"])


class TestCreateGridLayout:
    """Test cases for the standalone create_grid_layout function."""
    
    def test_row_pattern(self):
        """Test creating a grid with row-wise pattern."""
        grid = create_grid_layout(4, 4, fill_pattern='row')
        
        # Check dimensions
        assert grid.shape == (4, 4)
        
        # Check row-wise numbering (0-15)
        expected = np.array([
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15]
        ])
        assert np.array_equal(grid, expected)
    
    def test_column_pattern(self):
        """Test creating a grid with column-wise pattern."""
        grid = create_grid_layout(4, 4, fill_pattern='column')
        
        # Check column-wise numbering (0-15)
        expected = np.array([
            [0, 4, 8, 12],
            [1, 5, 9, 13],
            [2, 6, 10, 14],
            [3, 7, 11, 15]
        ])
        assert np.array_equal(grid, expected)
    
    def test_missing_indices(self):
        """Test creating a grid with missing indices."""
        missing = [(0, 0), (1, 1), (2, 2), (3, 3)]
        grid = create_grid_layout(4, 4, fill_pattern='row', missing_indices=missing)
        
        # Create expected grid with -1 at diagonal positions
        expected = np.array([
            [-1, 0, 1, 2],
            [3, -1, 4, 5],
            [6, 7, -1, 8],
            [9, 10, 11, -1]
        ])
        assert np.array_equal(grid, expected)
    
    def test_limited_electrodes(self):
        """Test creating a grid with limited number of electrodes."""
        grid = create_grid_layout(3, 3, n_electrodes=5, fill_pattern='row')
        
        # Only the first 5 positions should be filled (0-4)
        expected = np.array([
            [0, 1, 2],
            [3, 4, -1],
            [-1, -1, -1]
        ])
        assert np.array_equal(grid, expected)
    
    def test_invalid_pattern(self):
        """Test with invalid fill pattern."""
        with pytest.raises(ValueError, match="Invalid fill pattern"):
            create_grid_layout(3, 3, fill_pattern='invalid')
    
    def test_too_many_electrodes(self):
        """Test with too many electrodes for the grid size."""
        with pytest.raises(ValueError, match="Number of electrodes .* exceeds available positions"):
            create_grid_layout(2, 2, n_electrodes=5)


class TestEMGDataWithGridLayouts:
    """Test cases for EMGData class with grid layouts."""
    
    @pytest.fixture
    def emg_data_2d(self):
        """Create a 2D EMG data fixture."""
        # 16 channels, 1000 samples
        return np.random.rand(16, 1000)
    
    @pytest.fixture
    def sampling_frequency(self):
        """Return a sampling frequency for testing."""
        return 2000.0
    
    @pytest.fixture
    def grid_layout_4x4(self):
        """Create a 4x4 grid layout."""
        return create_grid_layout(4, 4, fill_pattern='row')
    
    def test_init_with_grid_layout(self, emg_data_2d, sampling_frequency, grid_layout_4x4):
        """Test initialization with grid layout."""
        emg = EMGData(emg_data_2d, sampling_frequency, grid_layouts=[grid_layout_4x4])
        
        # Check that grid_layouts is properly stored
        assert len(emg.grid_layouts) == 1
        assert np.array_equal(emg.grid_layouts[0], grid_layout_4x4)
    
    def test_init_with_multiple_grid_layouts(self, emg_data_2d, sampling_frequency):
        """Test initialization with multiple grid layouts."""
        # Create two 4x2 grids for a total of 16 electrodes
        grid1 = create_grid_layout(4, 2, 8, fill_pattern='row')
        grid2 = np.copy(grid1)
        # Adjust indices for second grid (8-15)
        grid2[grid2 >= 0] += 8
        
        emg = EMGData(emg_data_2d, sampling_frequency, grid_layouts=[grid1, grid2])
        
        # Check that both grids are stored
        assert len(emg.grid_layouts) == 2
        assert np.array_equal(emg.grid_layouts[0], grid1)
        assert np.array_equal(emg.grid_layouts[1], grid2)
    
    def test_invalid_grid_layout(self, emg_data_2d, sampling_frequency):
        """Test initialization with invalid grid layout."""
        # Create a grid with electrode indices exceeding the data dimensions
        invalid_grid = create_grid_layout(4, 4, fill_pattern='row')
        invalid_grid[0, 0] = 20  # Out of bounds for 16-channel data
        
        with pytest.raises(ValueError, match="Grid layout .* contains electrode indices that exceed"):
            EMGData(emg_data_2d, sampling_frequency, grid_layouts=[invalid_grid])
    
    def test_duplicate_indices_in_grid(self, emg_data_2d, sampling_frequency):
        """Test grid layout with duplicate electrode indices."""
        # Create a grid with duplicate indices
        invalid_grid = create_grid_layout(4, 4, fill_pattern='row')
        invalid_grid[1, 1] = invalid_grid[0, 0]  # Duplicate index
        
        with pytest.raises(ValueError, match="Grid layout .* contains duplicate electrode indices"):
            EMGData(emg_data_2d, sampling_frequency, grid_layouts=[invalid_grid])
    
    def test_plot_with_grid_layouts(self, emg_data_2d, sampling_frequency, grid_layout_4x4, monkeypatch):
        """Test plot method with grid layouts."""
        # Mock plt.show to avoid displaying the plot during tests
        monkeypatch.setattr(plt, "show", lambda: None)
        
        emg = EMGData(emg_data_2d, sampling_frequency, grid_layouts=[grid_layout_4x4])
        
        # This should not raise any errors
        emg.plot("Input", use_grid_layouts=True)
        
        # Test with custom scaling factor
        emg.plot("Input", scaling_factor=30.0)
        
        # Test with multiple scaling factors
        emg.plot("Input", scaling_factor=[20.0])
    
    def test_plot_grid_layout(self, emg_data_2d, sampling_frequency, grid_layout_4x4, monkeypatch):
        """Test plot_grid_layout method."""
        # Mock plt.show to avoid displaying the plot during tests
        monkeypatch.setattr(plt, "show", lambda: None)
        
        emg = EMGData(emg_data_2d, sampling_frequency, grid_layouts=[grid_layout_4x4])
        
        # This should not raise any errors
        emg.plot_grid_layout(0)
        
        # Test with show_indices=False
        emg.plot_grid_layout(0, show_indices=False)
    
    def test_plot_grid_layout_errors(self, emg_data_2d, sampling_frequency):
        """Test error cases for plot_grid_layout method."""
        # EMG without grid layouts
        emg_no_grid = EMGData(emg_data_2d, sampling_frequency)
        
        with pytest.raises(ValueError, match="Cannot plot grid layout"):
            emg_no_grid.plot_grid_layout(0)
        
        # EMG with grid layouts but invalid index
        emg_with_grid = EMGData(emg_data_2d, sampling_frequency, 
                               grid_layouts=[create_grid_layout(4, 4)])
        
        with pytest.raises(ValueError, match="Grid index .* out of range"):
            emg_with_grid.plot_grid_layout(1)  # Only have grid index 0
    
    def test_get_grid_dimensions(self, emg_data_2d, sampling_frequency):
        """Test _get_grid_dimensions method."""
        # Create two grids with different dimensions
        grid1 = create_grid_layout(3, 4, 12, fill_pattern='row')
        grid2 = create_grid_layout(2, 2, 4, fill_pattern='row')
        
        emg = EMGData(emg_data_2d, sampling_frequency, grid_layouts=[grid1, grid2])
        
        dimensions = emg._get_grid_dimensions()
        assert len(dimensions) == 2
        assert dimensions[0] == (3, 4, 12)  # rows, cols, electrodes
        assert dimensions[1] == (2, 2, 4)


if __name__ == "__main__":
    pytest.main() 