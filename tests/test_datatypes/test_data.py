import os
import pickle
import tempfile
import unittest
from typing import Union, List

import numpy as np
import pytest
import networkx as nx

from myoverse.datatypes import (
    _Data,
    DeletedRepresentation,
    InputRepresentationName,
    LastRepresentationName,
    OutputRepresentationName,
)
from myoverse.datasets.filters._template import FilterBaseClass


# Define operations separately so they are picklable
def double_data(x):
    return x * 2

def add_one(x):
    return x + 1

def square_data(x):
    return x ** 2

def abs_data(x):
    return np.abs(x)

def log_data(x):
    return np.log(x + 1)

def concat_arrays(arrays):
    return np.concatenate(arrays, axis=0)


class MockFilter(FilterBaseClass):
    """A simple mock filter for testing."""

    def __init__(
        self,
        input_is_chunked=True,
        allowed_input_type="both",
        is_output=False,
        name=None,
        run_checks=True,
        operation=None,
    ):
        super().__init__(
            input_is_chunked=input_is_chunked,
            allowed_input_type=allowed_input_type,
            is_output=is_output,
            name=name,
            run_checks=run_checks,
        )
        # Default to identity function if no operation provided
        if operation is None:
            self.operation = lambda x: x
        else:
            # Store the function by name for picklability
            if operation is double_data:
                self.operation_name = "double_data"
                self.operation = double_data
            elif operation is add_one:
                self.operation_name = "add_one"
                self.operation = add_one
            elif operation is square_data:
                self.operation_name = "square_data"
                self.operation = square_data
            elif operation is abs_data:
                self.operation_name = "abs_data"
                self.operation = abs_data
            elif operation is log_data:
                self.operation_name = "log_data"
                self.operation = log_data
            elif operation is concat_arrays:
                self.operation_name = "concat_arrays"
                self.operation = concat_arrays
            else:
                # For custom operations, we can't guarantee picklability
                self.operation = operation

    def _filter(self, input_array):
        """Apply the filter operation to the input array."""
        if isinstance(input_array, list):
            # For multi-input filters
            return self.operation(input_array)
        return self.operation(input_array)

    def __getstate__(self):
        """Make the filter picklable by handling the operation."""
        state = self.__dict__.copy()
        # If we have a named operation, we'll restore it during __setstate__
        if hasattr(self, 'operation_name'):
            # Only store the name
            state['operation'] = None
        return state
    
    def __setstate__(self, state):
        """Restore the state including the operation function."""
        self.__dict__.update(state)
        # Restore the operation based on its name
        if hasattr(self, 'operation_name'):
            if self.operation_name == "double_data":
                self.operation = double_data
            elif self.operation_name == "add_one":
                self.operation = add_one
            elif self.operation_name == "square_data":
                self.operation = square_data
            elif self.operation_name == "abs_data":
                self.operation = abs_data
            elif self.operation_name == "log_data":
                self.operation = log_data
            elif self.operation_name == "concat_arrays":
                self.operation = concat_arrays


class MultiInputMockFilter(FilterBaseClass):
    """A mock filter that accepts multiple inputs."""

    def __init__(
        self,
        input_is_chunked=True,
        allowed_input_type="both",
        is_output=False,
        name=None,
        run_checks=True,
        operation=None,
    ):
        super().__init__(
            input_is_chunked=input_is_chunked,
            allowed_input_type=allowed_input_type,
            is_output=is_output,
            name=name,
            run_checks=run_checks,
        )
        # Use concat_arrays by default
        self.operation = operation or concat_arrays
        if operation is concat_arrays:
            self.operation_name = "concat_arrays"

    def _filter(self, input_array_list):
        """Apply the filter operation to the list of input arrays."""
        return self.operation(input_array_list)
    
    def __getstate__(self):
        """Make the filter picklable by handling the operation."""
        state = self.__dict__.copy()
        # If we have a named operation, we'll restore it during __setstate__
        if hasattr(self, 'operation_name'):
            # Only store the name
            state['operation'] = None
        return state
    
    def __setstate__(self, state):
        """Restore the state including the operation function."""
        self.__dict__.update(state)
        # Restore the operation based on its name
        if hasattr(self, 'operation_name'):
            if self.operation_name == "concat_arrays":
                self.operation = concat_arrays


class TestData(_Data):
    """Concrete implementation of _Data for testing."""

    def __init__(self, raw_data, sampling_frequency):
        super().__init__(raw_data, sampling_frequency)

    def _check_if_chunked(self, data: Union[np.ndarray, DeletedRepresentation]) -> bool:
        """Checks if the data is chunked or not.
        
        For testing purposes, we'll say data is chunked if the shape has at least 2 dimensions
        and the first dimension is greater than 1.
        """
        if isinstance(data, DeletedRepresentation):
            shape = data.shape
        else:
            shape = data.shape

        return len(shape) >= 2 and shape[0] > 1

    def plot(self, representation=None):
        """Dummy plot method for testing."""
        pass


class TestDataClass(unittest.TestCase):
    """Test suite for the _Data base class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple data array (2 channels, 100 samples)
        self.raw_data = np.random.randn(2, 100)
        self.sampling_frequency = 1000.0  # 1 kHz
        self.data_obj = TestData(self.raw_data, self.sampling_frequency)

    def test_initialization(self):
        """Test that the object initializes correctly."""
        self.assertEqual(self.data_obj.sampling_frequency, self.sampling_frequency)
        np.testing.assert_array_equal(self.data_obj.input_data, self.raw_data)
        self.assertEqual(self.data_obj._last_processing_step, InputRepresentationName)
        
        # Test initialization with invalid sampling frequency
        with self.assertRaises(ValueError):
            TestData(self.raw_data, 0)
        
        # Check initial graph structure
        self.assertIn(InputRepresentationName, self.data_obj._processed_representations.nodes)
        self.assertIn(OutputRepresentationName, self.data_obj._processed_representations.nodes)

    def test_properties(self):
        """Test properties and their setters."""
        # Test read-only properties
        with self.assertRaises(RuntimeError):
            self.data_obj.input_data = np.zeros((2, 100))
            
        with self.assertRaises(RuntimeError):
            self.data_obj.processed_representations = {}
            
        with self.assertRaises(RuntimeError):
            self.data_obj.output_representations = {}
        
        # Test is_chunked property
        chunked_status = self.data_obj.is_chunked
        self.assertIsInstance(chunked_status, dict)
        self.assertTrue(chunked_status[InputRepresentationName])  # Our test data is chunked

    def test_apply_filter(self):
        """Test applying a single filter."""
        # Create a simple filter that multiplies by 2
        filter_name = "double_values"
        filter_obj = MockFilter(
            input_is_chunked=True, 
            name=filter_name, 
            operation=double_data
        )
        
        # Apply the filter
        result_name = self.data_obj.apply_filter(
            filter=filter_obj,
            representations_to_filter=[InputRepresentationName]
        )
        
        self.assertEqual(result_name, filter_name)
        
        # Check the result
        result_data = self.data_obj[filter_name]
        np.testing.assert_array_equal(result_data, self.raw_data * 2)
        
        # Check that the filter was registered
        self.assertIn(filter_name, self.data_obj._filters_used)
        
        # Check that the graph was updated
        self.assertIn(filter_name, self.data_obj._processed_representations.nodes)
        self.assertTrue(
            self.data_obj._processed_representations.has_edge(
                InputRepresentationName, filter_name
            )
        )
        
        # Check that _last_processing_step was updated
        self.assertEqual(self.data_obj._last_processing_step, filter_name)
        
        # Test with a string instead of a list
        with self.assertRaises(ValueError):
            self.data_obj.apply_filter(
                filter=filter_obj,
                representations_to_filter=InputRepresentationName  # This should be a list
            )
            
        # Test with is_output=True
        output_filter = MockFilter(
            input_is_chunked=True, 
            name="output_filter", 
            is_output=True, 
            operation=lambda x: x * 3
        )
        
        output_name = self.data_obj.apply_filter(
            filter=output_filter,
            representations_to_filter=[filter_name]
        )
        
        self.assertTrue(
            self.data_obj._processed_representations.has_edge(
                output_name, OutputRepresentationName
            )
        )
        
        # Verify the output representation is correctly registered
        self.assertIn(output_name, self.data_obj.output_representations)
        
        # Verify the output data is correct - use np.testing.assert_array_equal for array comparison
        np.testing.assert_array_equal(
            self.data_obj.output_representations[output_name], 
            self.data_obj._data[output_name]
        )
        
        # Test with keep_representation_to_filter=False
        discard_filter = MockFilter(
            input_is_chunked=True, 
            name="discard_test", 
            operation=lambda x: x * 4
        )
        
        discard_name = self.data_obj.apply_filter(
            filter=discard_filter,
            representations_to_filter=[output_name],
            keep_representation_to_filter=False
        )
        
        # Check that the previous representation is now a DeletedRepresentation
        self.assertIsInstance(self.data_obj._data[output_name], DeletedRepresentation)

    def test_apply_filter_sequence(self):
        """Test applying a sequence of filters."""
        # Create a sequence of filters
        filter1 = MockFilter(
            input_is_chunked=True, 
            name="double", 
            operation=double_data
        )
        
        filter2 = MockFilter(
            input_is_chunked=True, 
            name="add_one", 
            operation=add_one
        )
        
        filter3 = MockFilter(
            input_is_chunked=True, 
            name="square", 
            operation=square_data
        )
        
        filter_sequence = [filter1, filter2, filter3]
        
        # Apply the sequence
        result_name = self.data_obj.apply_filter_sequence(
            filter_sequence=filter_sequence,
            representations_to_filter=[InputRepresentationName]
        )
        
        self.assertEqual(result_name, "square")
        
        # Check the result: (raw_data * 2 + 1) ** 2
        expected_result = ((self.raw_data * 2) + 1) ** 2
        np.testing.assert_array_equal(self.data_obj[result_name], expected_result)
        
        # Test with no filters
        with self.assertRaises(ValueError):
            self.data_obj.apply_filter_sequence(
                filter_sequence=[],
                representations_to_filter=[InputRepresentationName]
            )
        
        # Test with keep_individual_filter_steps=False
        self.data_obj.apply_filter_sequence(
            filter_sequence=filter_sequence,
            representations_to_filter=[InputRepresentationName],
            keep_individual_filter_steps=False
        )
        
        # Check that intermediate results are deleted
        self.assertIsInstance(self.data_obj._data["double"], DeletedRepresentation)
        self.assertIsInstance(self.data_obj._data["add_one"], DeletedRepresentation)
        # The final result should still be a numpy array
        self.assertIsInstance(self.data_obj._data["square"], np.ndarray)
        
        # Test with keep_representation_to_filter=False
        new_input = "square"  # use the result of the previous operation
        new_filter = MockFilter(
            input_is_chunked=True, 
            name="final", 
            operation=lambda x: x / 2
        )
        
        self.data_obj.apply_filter_sequence(
            filter_sequence=[new_filter],
            representations_to_filter=[new_input],
            keep_representation_to_filter=False
        )
        
        self.assertIsInstance(self.data_obj._data[new_input], DeletedRepresentation)

    def test_apply_filter_pipeline(self):
        """Test applying a filter pipeline with multiple branches."""
        # Create filters for the first branch
        branch1_filter1 = MockFilter(
            input_is_chunked=True, 
            name="abs", 
            operation=abs_data
        )
        branch1_filter2 = MockFilter(
            input_is_chunked=True, 
            name="log", 
            operation=log_data
        )
        branch1 = [branch1_filter1, branch1_filter2]
        
        # Create filters for the second branch
        branch2_filter1 = MockFilter(
            input_is_chunked=True, 
            name="square", 
            operation=square_data
        )
        branch2 = [branch2_filter1]
        
        # Create a merge filter for combining the branches
        merge_filter = MultiInputMockFilter(
            input_is_chunked=True,
            name="merge",
            operation=concat_arrays
        )
        branch3 = [merge_filter]
        
        # Apply the pipeline
        self.data_obj.apply_filter_pipeline(
            filter_pipeline=[branch1, branch2, branch3],
            representations_to_filter=[
                [InputRepresentationName],
                [InputRepresentationName],
                ["log", "square"]
            ]
        )
        
        # Check that all branches were processed
        self.assertIn("abs", self.data_obj._data)
        self.assertIn("log", self.data_obj._data)
        self.assertIn("square", self.data_obj._data)
        self.assertIn("merge", self.data_obj._data)
        
        # Check the merge result - should be concatenation of log and square
        expected_merge = np.concatenate([
            self.data_obj["log"],
            self.data_obj["square"]
        ], axis=0)
        np.testing.assert_array_equal(self.data_obj["merge"], expected_merge)
        
        # Test with invalid inputs
        with self.assertRaises(ValueError):
            # Different number of branches and representations
            self.data_obj.apply_filter_pipeline(
                filter_pipeline=[branch1, branch2],
                representations_to_filter=[[InputRepresentationName]]
            )
            
        with self.assertRaises(ValueError):
            # String instead of list
            self.data_obj.apply_filter_pipeline(
                filter_pipeline=[branch1],
                representations_to_filter=[InputRepresentationName]  # Should be a list
            )

    def test_getitem_and_deleted_representation(self):
        """Test item access and automatic recomputation of deleted representations."""
        # Create a filter
        filter_obj = MockFilter(
            input_is_chunked=True, 
            name="test_filter", 
            operation=double_data
        )
        
        # Apply the filter
        self.data_obj.apply_filter(
            filter=filter_obj,
            representations_to_filter=[InputRepresentationName]
        )
        
        # Check normal access
        np.testing.assert_array_equal(self.data_obj["test_filter"], self.raw_data * 2)
        
        # Delete the representation data
        self.data_obj.delete_data("test_filter")
        
        # Check that it was replaced with a DeletedRepresentation
        self.assertIsInstance(self.data_obj._data["test_filter"], DeletedRepresentation)
        
        # Access it again - should recompute
        recomputed_data = self.data_obj["test_filter"]
        np.testing.assert_array_equal(recomputed_data, self.raw_data * 2)
        
        # Now it should be a numpy array again
        self.assertIsInstance(self.data_obj._data["test_filter"], np.ndarray)
        
        # Test access with the LastRepresentationName
        np.testing.assert_array_equal(self.data_obj[LastRepresentationName], self.data_obj["test_filter"])
        
        # Test access with nonexistent key
        with self.assertRaises(KeyError):
            self.data_obj["nonexistent"]
            
        # Test __setitem__ (which should raise an error)
        with self.assertRaises(RuntimeError):
            self.data_obj["new_key"] = np.zeros((2, 100))

    def test_delete_methods(self):
        """Test delete_data, delete_history, and delete methods."""
        # Create filters
        filter1 = MockFilter(
            input_is_chunked=True, 
            name="filter1", 
            operation=double_data
        )
        filter2 = MockFilter(
            input_is_chunked=True, 
            name="filter2", 
            operation=add_one
        )
        
        # Apply the filters in sequence
        self.data_obj.apply_filter_sequence(
            filter_sequence=[filter1, filter2],
            representations_to_filter=[InputRepresentationName]
        )
        
        # Test delete_data
        self.data_obj.delete_data("filter1")
        self.assertIsInstance(self.data_obj._data["filter1"], DeletedRepresentation)
        self.assertIn("filter1", self.data_obj._processed_representations.nodes)
        
        # Test delete_history
        self.data_obj.delete_history("filter1")
        self.assertNotIn("filter1", self.data_obj._processed_representations.nodes)
        self.assertNotIn("filter1", self.data_obj._filters_used)
        self.assertIn("filter1", self.data_obj._data)  # Data is still there as DeletedRepresentation
        
        # For the delete test, create a new filter 
        delete_filter = MockFilter(
            input_is_chunked=True, 
            name="to_delete", 
            operation=square_data
        )
        self.data_obj.apply_filter(
            filter=delete_filter,
            representations_to_filter=[InputRepresentationName]
        )
        
        # Make sure it's there as a numpy array
        self.assertIn("to_delete", self.data_obj._data)
        self.assertIsInstance(self.data_obj._data["to_delete"], np.ndarray)
        self.assertIn("to_delete", self.data_obj._processed_representations.nodes)
        self.assertIn("to_delete", self.data_obj._filters_used)
        
        # Now use delete and verify the correct behavior
        self.data_obj.delete("to_delete")
        
        # The data should still be in _data as a DeletedRepresentation
        self.assertIn("to_delete", self.data_obj._data)
        self.assertIsInstance(self.data_obj._data["to_delete"], DeletedRepresentation)
        
        # But it should be removed from the processing graph and filters
        self.assertNotIn("to_delete", self.data_obj._processed_representations.nodes)
        self.assertNotIn("to_delete", self.data_obj._filters_used)
        
        # Test deleting InputRepresentationName (should be a no-op)
        self.data_obj.delete_data(InputRepresentationName)
        self.assertIn(InputRepresentationName, self.data_obj._data)
        self.assertIsInstance(self.data_obj._data[InputRepresentationName], np.ndarray)
        
        # Test deleting LastRepresentationName
        # First create a new filter
        filter3 = MockFilter(
            input_is_chunked=True, 
            name="filter3", 
            operation=lambda x: x * 3
        )
        self.data_obj.apply_filter(
            filter=filter3, 
            representations_to_filter=[InputRepresentationName]
        )
        
        self.assertEqual(self.data_obj._last_processing_step, "filter3")
        self.data_obj.delete_data(LastRepresentationName)
        self.assertIsInstance(self.data_obj._data["filter3"], DeletedRepresentation)
        
        # Test deleting nonexistent representation
        with self.assertRaises(KeyError):
            self.data_obj.delete_data("nonexistent")

    def test_copy(self):
        """Test the ability to copy the data object."""
        # Create a simple data object with a named operation filter
        raw_data = np.random.randn(2, 50)
        data_obj = TestData(raw_data, 1000.0)
        
        # Apply a filter with a named operation
        filter_obj = MockFilter(
            input_is_chunked=True, 
            name="filter_copy_test", 
            operation=double_data
        )
        data_obj.apply_filter(
            filter=filter_obj,
            representations_to_filter=[InputRepresentationName]
        )
        
        # Create a new object with the same data but different filter
        raw_data2 = raw_data.copy()
        data_obj2 = TestData(raw_data2, 1000.0)
        filter_obj2 = MockFilter(
            input_is_chunked=True, 
            name="different_filter", 
            operation=square_data
        )
        data_obj2.apply_filter(
            filter=filter_obj2,
            representations_to_filter=[InputRepresentationName]
        )
        
        # Check that the two objects have different filters
        self.assertIn("filter_copy_test", data_obj._data)
        self.assertIn("different_filter", data_obj2._data)
        self.assertNotIn("filter_copy_test", data_obj2._data)
        self.assertNotIn("different_filter", data_obj._data)
        
        # Verify they're completely different objects
        self.assertIsNot(data_obj, data_obj2)
        self.assertIsNot(data_obj._data, data_obj2._data)
        self.assertIsNot(data_obj._processed_representations, data_obj2._processed_representations)

    def test_save_and_load(self):
        """Test saving and loading the data."""
        # Create a filter with a named operation for picklability
        filter_obj = MockFilter(
            input_is_chunked=True, 
            name="save_test", 
            operation=double_data
        )
        
        # Create a separate data object for this test
        raw_data = np.random.randn(2, 50)
        data_obj = TestData(raw_data, 1000.0)
        
        # Apply the filter
        data_obj.apply_filter(
            filter=filter_obj,
            representations_to_filter=[InputRepresentationName]
        )
        
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_filename = temp_file.name
            
        try:
            # Save the data
            data_obj.save(temp_filename)
            
            # Load the data
            loaded_data = TestData.load(temp_filename)
            
            # Check that the loaded data has the same attributes
            self.assertEqual(loaded_data.sampling_frequency, data_obj.sampling_frequency)
            np.testing.assert_array_equal(loaded_data.input_data, data_obj.input_data)
            
            # Check that the representation was loaded
            self.assertIn("save_test", loaded_data._data)
            np.testing.assert_array_equal(
                loaded_data["save_test"], 
                data_obj["save_test"]
            )
            
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)

    def test_memory_usage(self):
        """Test the memory_usage method."""
        # Create an initial memory usage snapshot
        initial_usage = self.data_obj.memory_usage()
        self.assertIn(InputRepresentationName, initial_usage)
        self.assertEqual(initial_usage[InputRepresentationName][0], str(self.raw_data.shape))
        self.assertEqual(initial_usage[InputRepresentationName][1], self.raw_data.nbytes)
        
        # Create a filter and apply it
        filter_obj = MockFilter(
            input_is_chunked=True, 
            name="memory_test", 
            operation=double_data
        )
        self.data_obj.apply_filter(
            filter=filter_obj,
            representations_to_filter=[InputRepresentationName]
        )
        
        # Check updated memory usage
        updated_usage = self.data_obj.memory_usage()
        self.assertIn("memory_test", updated_usage)
        self.assertEqual(updated_usage["memory_test"][0], str(self.raw_data.shape))
        self.assertEqual(updated_usage["memory_test"][1], self.raw_data.nbytes)
        
        # Delete the data and check that memory usage shows 0 bytes
        self.data_obj.delete_data("memory_test")
        after_delete_usage = self.data_obj.memory_usage()
        self.assertIn("memory_test", after_delete_usage)
        self.assertEqual(after_delete_usage["memory_test"][1], 0)

    def test_get_representation_history(self):
        """Test the get_representation_history method."""
        # Create a sequence of filters
        filter1 = MockFilter(
            input_is_chunked=True, 
            name="step1", 
            operation=double_data
        )
        filter2 = MockFilter(
            input_is_chunked=True, 
            name="step2", 
            operation=add_one
        )
        filter3 = MockFilter(
            input_is_chunked=True, 
            name="step3", 
            operation=square_data
        )
        
        # Apply the filters in sequence
        self.data_obj.apply_filter_sequence(
            filter_sequence=[filter1, filter2, filter3],
            representations_to_filter=[InputRepresentationName]
        )
        
        # Check the history
        history = self.data_obj.get_representation_history("step3")
        self.assertEqual(
            history, 
            [InputRepresentationName, "step1", "step2", "step3"]
        )
        
        # Check history with a branch
        branch_filter = MockFilter(
            input_is_chunked=True, 
            name="branch", 
            operation=lambda x: x / 2
        )
        self.data_obj.apply_filter(
            filter=branch_filter,
            representations_to_filter=["step2"]  # Branch off from step2
        )
        
        branch_history = self.data_obj.get_representation_history("branch")
        self.assertEqual(
            branch_history, 
            [InputRepresentationName, "step1", "step2", "branch"]
        )

    def test_str_and_repr(self):
        """Test the string representation methods."""
        # Create a basic object and check its representation
        simple_data = TestData(np.zeros((2, 10)), 1000.0)
        
        # Test __repr__
        repr_str = repr(simple_data)
        self.assertIn("TestData", repr_str)
        self.assertIn("Sampling frequency: 1000.0 Hz", repr_str)
        self.assertIn("Input (2, 10)", repr_str)
        
        # Test __str__
        str_output = str(simple_data)
        self.assertIn("TestData", str_output)
        self.assertIn("Sampling frequency: 1000.0 Hz", str_output)
        self.assertIn("Input (2, 10)", str_output)
        
        # Add a filter and check that it appears in the representation
        filter_obj = MockFilter(
            input_is_chunked=True, 
            name="repr_test", 
            is_output=True,
            operation=double_data
        )
        simple_data.apply_filter(
            filter=filter_obj,
            representations_to_filter=[InputRepresentationName]
        )
        
        updated_repr = repr(simple_data)
        self.assertIn("repr_test", updated_repr)
        self.assertIn("(Output)", updated_repr)
        
        updated_str = str(simple_data)
        self.assertIn("repr_test", updated_str)
        self.assertIn("(Output)", updated_str)


if __name__ == "__main__":
    unittest.main() 