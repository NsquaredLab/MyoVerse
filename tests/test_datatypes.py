import numpy as np
import pytest
import networkx as nx
from matplotlib import pyplot as plt
from copy import deepcopy

from myoverse.datatypes import EMGData, KinematicsData, VirtualHandKinematics
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
        np.random.seed(42)
        # 9 DoFs (3 for wrist, 2 for each finger) and 100 timesteps
        data = np.random.rand(9, 100)  
        sampling_frequency = 100  # Hz
        return VirtualHandKinematics(data, sampling_frequency)

    def test_init_valid(self, sample_vhk_data):
        """Test initialization with valid data."""
        assert sample_vhk_data.input_data.shape == (9, 100)
        assert sample_vhk_data.sampling_frequency == 100
        assert sample_vhk_data.is_chunked["Input"] is False

    def test_init_invalid_dims(self):
        """Test initialization with invalid dimensions."""
        with pytest.raises(ValueError):
            # Only 1D, should be 2D
            VirtualHandKinematics(np.random.rand(100), 100)

    def test_init_invalid_frequency(self):
        """Test initialization with invalid sampling frequency."""
        with pytest.raises(ValueError):
            # Negative sampling frequency
            VirtualHandKinematics(np.random.rand(9, 100), -10)

    def test_plot(self, sample_vhk_data, monkeypatch):
        """Test that plot method works without errors."""
        # Mock plt.show to prevent actual plot display during tests
        monkeypatch.setattr(plt, 'show', lambda: None)
        
        # Test plotting with different options
        sample_vhk_data.plot("Input", visualize_wrist=True)
        sample_vhk_data.plot("Input", visualize_wrist=False)
        
        # No assertions needed as we're just checking it doesn't raise exceptions


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


if __name__ == "__main__":
    pytest.main() 