from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union, List, Tuple, Callable, Type
import warnings

import zarr
import numpy as np
import lightning as L
from torch.utils.data import DataLoader, Dataset

from myoverse.datasets.filters.generic import (
    IdentityFilter,
    FilterBaseClass,
    IndexDataFilter,
)
from myoverse.datatypes import EMGData, KinematicsData, _Data


class EMGZarrDataset(Dataset):
    """Dataset class for loading EMG data from Zarr files.

    Parameters
    ----------
    zarr_file : Path
        The path to the zarr file
    subset_name : str
        The name of the subset to load (e.g., "training", "validation", "testing")
    target_name : str
        The name of the target data
    emg_dtype : np.dtype, optional
        The data type of the EMG data, by default np.float32
    target_dtype : np.dtype, optional
        The data type of the target data, by default np.float32
    sampling_frequency : float, optional
        The sampling frequency of the EMG data in Hz, by default 2048.0
    input_data_class : Type[_Data], optional
        The class to use for input data, by default EMGData
    target_data_class : Type[_Data], optional
        The class to use for target data, by default KinematicsData
    input_augmentation_pipeline : list[list[FilterBaseClass]], optional
        The augmentation pipeline for the input data
    input_augmentation_probabilities : Sequence[float], optional
        The probabilities for each input augmentation pipeline
    target_augmentation_pipeline : list[list[FilterBaseClass]], optional
        The augmentation pipeline for the target data
    target_augmentation_probabilities : Sequence[float], optional
        The probabilities for each target augmentation pipeline
    cache_size : int, optional
        The maximum number of items to cache, by default 100
    """

    def __init__(
        self,
        zarr_file: Path,
        subset_name: str,
        target_name: str,
        emg_dtype=np.float32,
        target_dtype=np.float32,
        sampling_frequency: float = 2048.0,
        input_data_class: Type[_Data] = EMGData,
        target_data_class: Type[_Data] = KinematicsData,
        input_augmentation_pipeline: list[list[FilterBaseClass]] = None,
        input_augmentation_probabilities: Sequence[float] = None,
        target_augmentation_pipeline: list[list[FilterBaseClass]] = None,
        target_augmentation_probabilities: Sequence[float] = None,
        cache_size: int = 100,
    ):
        self.zarr_file = zarr_file
        self.subset_name = subset_name
        self.target_name = target_name
        self.sampling_frequency = sampling_frequency
        self.input_data_class = input_data_class
        self.target_data_class = target_data_class

        # Validate zarr file exists
        if not Path(zarr_file).exists():
            raise FileNotFoundError(f"Zarr file not found at {zarr_file}")

        # Set default augmentation pipelines if None
        if input_augmentation_pipeline is None:
            self.input_augmentation_pipeline = [
                [IdentityFilter(is_output=True, input_is_chunked=True)]
            ]
        else:
            self.input_augmentation_pipeline = input_augmentation_pipeline

        if input_augmentation_probabilities is None:
            self.input_augmentation_probabilities = (1.0,)
        else:
            self.input_augmentation_probabilities = input_augmentation_probabilities

        if target_augmentation_pipeline is None:
            self.target_augmentation_pipeline = [
                [IndexDataFilter(indices=(0,), is_output=True, input_is_chunked=True)]
            ]
        else:
            self.target_augmentation_pipeline = target_augmentation_pipeline

        if target_augmentation_probabilities is None:
            self.target_augmentation_probabilities = (1.0,)
        else:
            self.target_augmentation_probabilities = target_augmentation_probabilities

        # Validate augmentation probabilities
        self._validate_augmentation_probabilities()

        # Initialize cache
        self._cache = {}
        self.cache_size = cache_size

        # Load data from zarr file
        try:
            zarr_root = zarr.open(str(self.zarr_file))
            if self.subset_name not in zarr_root:
                raise ValueError(f"Subset '{self.subset_name}' not found in Zarr file")

            subset = zarr_root[self.subset_name]
            if "emg" not in subset:
                raise ValueError(f"EMG data not found in subset '{self.subset_name}'")

            self._emg_data = subset["emg"]
            self._emg_data = {
                key: self._emg_data[key] for key in self._emg_data.array_keys()
            }

            if self.target_name not in subset:
                raise ValueError(
                    f"Target data '{self.target_name}' not found in subset '{self.subset_name}'"
                )

            self._target_data = subset[self.target_name]

            try:
                self._target_data = {
                    key: self._target_data[key]
                    for key in self._target_data.array_keys()
                }
            except AttributeError:
                self._target_data = {"temp": self._target_data}

            try:
                self.length = list(self._emg_data.values())[0].shape[0]
                if self.length == 0:
                    warnings.warn(f"Dataset '{self.subset_name}' is empty")
            except IndexError:
                self.length = 0
                warnings.warn(
                    f"Failed to determine dataset length for '{self.subset_name}'"
                )

        except Exception as e:
            raise ValueError(f"Failed to load data from Zarr file: {e}")

        self.emg_dtype = emg_dtype
        self.target_dtype = target_dtype

    def _validate_augmentation_probabilities(self):
        """Validate that augmentation probabilities are valid."""
        # Check input augmentation probabilities
        if len(self.input_augmentation_pipeline) != len(
            self.input_augmentation_probabilities
        ):
            raise ValueError(
                "Number of input augmentation probabilities must match number of pipelines"
            )
        if abs(sum(self.input_augmentation_probabilities) - 1.0) > 1e-6:
            raise ValueError("Sum of input augmentation probabilities must be 1.0")

        # Check target augmentation probabilities
        if len(self.target_augmentation_pipeline) != len(
            self.target_augmentation_probabilities
        ):
            raise ValueError(
                "Number of target augmentation probabilities must match number of pipelines"
            )
        if abs(sum(self.target_augmentation_probabilities) - 1.0) > 1e-6:
            raise ValueError("Sum of target augmentation probabilities must be 1.0")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Check if item is in cache
        if idx in self._cache:
            return self._cache[idx]

        input_data = []
        target_data = []

        # Choose augmentation pipeline based on probabilities
        input_augmentation_chosen = self.input_augmentation_pipeline[
            np.random.choice(
                len(self.input_augmentation_pipeline),
                p=self.input_augmentation_probabilities,
            )
        ]

        target_augmentation_chosen = self.target_augmentation_pipeline[
            np.random.choice(
                len(self.target_augmentation_pipeline),
                p=self.target_augmentation_probabilities,
            )
        ]

        # Process EMG data
        for v in self._emg_data.values():
            try:
                temp = self.input_data_class(
                    v[idx], sampling_frequency=self.sampling_frequency
                )
                temp.apply_filter_sequence(
                    input_augmentation_chosen, representations_to_filter=["Input"]
                )
                input_data.append(list(temp.output_representations.values())[0])
            except Exception as e:
                raise RuntimeError(f"Error processing EMG data at index {idx}: {e}")

        # Process target data
        for v in self._target_data.values():
            try:
                # Use the specified target_data_class without forced np.atleast_2d
                temp = self.target_data_class(
                    v[idx], sampling_frequency=self.sampling_frequency
                )
                temp.apply_filter_sequence(
                    target_augmentation_chosen,
                    representations_to_filter=["Input"],
                )
                target_data.append(list(temp.output_representations.values())[0])
            except Exception as e:
                raise RuntimeError(f"Error processing target data at index {idx}: {e}")

        # Convert to appropriate data types
        result = (
            np.array(input_data).astype(self.emg_dtype),
            np.array(target_data).astype(self.target_dtype),
        )

        # Update cache (with simple LRU-like behavior)
        if len(self._cache) >= self.cache_size:
            # Remove oldest item
            self._cache.pop(next(iter(self._cache)))
        self._cache[idx] = result

        return result


class EMGDatasetLoader(L.LightningDataModule):
    """Dataset loader for the EMG dataset.

    Parameters
    ----------
    data_path : Path
        The path to the zarr file
    seed : Optional[int], optional
        The seed for the random number generator, by default None
    dataloader_params : Optional[Dict[str, Any]], optional
        The parameters for the DataLoader, by default None
    shuffle_train : bool, optional
        Whether to shuffle the training data, by default True
    emg_dtype : np.dtype, optional
        The data type of the EMG data, by default np.float32
    target_dtype : np.dtype, optional
        The data type of the target data, by default np.float32
    target_name : str, optional
        The name of the target data, by default "ground_truth"
    sampling_frequency : float, optional
        The sampling frequency of the EMG data in Hz, by default 2048.0
    input_data_class : Type[_Data], optional
        The class to use for input data, by default EMGData
    target_data_class : Type[_Data], optional
        The class to use for target data, by default KinematicsData
    input_augmentation_pipeline : Optional[List[List[FilterBaseClass]]], optional
        The augmentation pipeline for the input data, by default None (identity filter)
    input_augmentation_probabilities : Optional[Sequence[float]], optional
        The probabilities for the input augmentation pipeline, by default None (1.0)
    target_augmentation_pipeline : Optional[List[List[FilterBaseClass]]], optional
        The augmentation pipeline for the target data, by default None (index filter)
    target_augmentation_probabilities : Optional[Sequence[float]], optional
        The probabilities for the target augmentation pipeline, by default None (1.0)
    cache_size : int, optional
        The maximum number of items to cache, by default 100
    preprocessing_hooks : Optional[Dict[str, Callable]], optional
        Custom preprocessing functions, by default None
    """

    def __init__(
        self,
        data_path: Path,
        seed: Optional[int] = None,
        dataloader_params: Optional[Dict[str, Any]] = None,
        shuffle_train: bool = True,
        emg_dtype=np.float32,
        target_dtype=np.float32,
        target_name: str = "ground_truth",
        sampling_frequency: float = 2048.0,
        input_data_class: Type[_Data] = EMGData,
        target_data_class: Type[_Data] = KinematicsData,
        input_augmentation_pipeline: Optional[List[List[FilterBaseClass]]] = None,
        input_augmentation_probabilities: Optional[Sequence[float]] = None,
        target_augmentation_pipeline: Optional[List[List[FilterBaseClass]]] = None,
        target_augmentation_probabilities: Optional[Sequence[float]] = None,
        cache_size: int = 100,
        preprocessing_hooks: Optional[Dict[str, Callable]] = None,
    ):
        super().__init__()

        # Validate and set data path
        if not isinstance(data_path, Path):
            data_path = Path(data_path)
        self.data_path = data_path

        # Set random seed
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

        # Set default dataloader parameters
        if dataloader_params is None:
            self.dataloader_params = {
                "batch_size": 32,
                "num_workers": 4,
                "pin_memory": True,
            }
        else:
            self.dataloader_params = dataloader_params

        # Set basic parameters
        self.shuffle_train = shuffle_train
        self.emg_dtype = emg_dtype
        self.target_dtype = target_dtype
        self.target_name = target_name
        self.sampling_frequency = sampling_frequency
        self.input_data_class = input_data_class
        self.target_data_class = target_data_class
        self.cache_size = cache_size
        self.preprocessing_hooks = preprocessing_hooks or {}

        # Set augmentation pipelines and probabilities
        self.input_augmentation_pipeline = input_augmentation_pipeline
        self.input_augmentation_probabilities = input_augmentation_probabilities
        self.target_augmentation_pipeline = target_augmentation_pipeline
        self.target_augmentation_probabilities = target_augmentation_probabilities

        # Validate zarr file
        self._validate_zarr_file()

    def _validate_zarr_file(self):
        """Validate that the zarr file exists and has the expected structure."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Zarr file not found at {self.data_path}")

        try:
            zarr_root = zarr.open(str(self.data_path))
            required_subsets = ["training", "validation", "testing"]
            for subset in required_subsets:
                if subset not in zarr_root:
                    warnings.warn(f"Subset '{subset}' not found in Zarr file")
        except Exception as e:
            raise ValueError(f"Failed to validate Zarr file: {e}")

    def _create_dataset(self, subset_name: str, use_augmentation: bool = False):
        """Create a dataset for a specific subset.

        Parameters
        ----------
        subset_name : str
            The name of the subset to load
        use_augmentation : bool, optional
            Whether to use augmentation, by default False

        Returns
        -------
        EMGZarrDataset
            The dataset for the specified subset
        """
        if use_augmentation:
            return EMGZarrDataset(
                zarr_file=self.data_path,
                subset_name=subset_name,
                target_name=self.target_name,
                emg_dtype=self.emg_dtype,
                target_dtype=self.target_dtype,
                sampling_frequency=self.sampling_frequency,
                input_data_class=self.input_data_class,
                target_data_class=self.target_data_class,
                input_augmentation_pipeline=self.input_augmentation_pipeline,
                input_augmentation_probabilities=self.input_augmentation_probabilities,
                target_augmentation_pipeline=self.target_augmentation_pipeline,
                target_augmentation_probabilities=self.target_augmentation_probabilities,
                cache_size=self.cache_size,
            )
        else:
            # Use identity filters for non-training sets (no augmentation)
            return EMGZarrDataset(
                zarr_file=self.data_path,
                subset_name=subset_name,
                target_name=self.target_name,
                emg_dtype=self.emg_dtype,
                target_dtype=self.target_dtype,
                sampling_frequency=self.sampling_frequency,
                input_data_class=self.input_data_class,
                target_data_class=self.target_data_class,
                cache_size=self.cache_size,
            )

    def train_dataloader(self) -> DataLoader:
        """Returns the training set as a DataLoader.

        Returns
        -------
        DataLoader
            The training set
        """
        return DataLoader(
            self._create_dataset("training", use_augmentation=True),
            shuffle=self.shuffle_train,
            **self.dataloader_params.copy(),
        )

    def test_dataloader(self) -> DataLoader:
        """Returns the testing set as a DataLoader.

        Returns
        -------
        DataLoader
            The testing set
        """
        return DataLoader(
            self._create_dataset("testing", use_augmentation=False),
            shuffle=False,
            **self.dataloader_params.copy(),
        )

    def val_dataloader(self) -> DataLoader:
        """Returns the validation set as a DataLoader.

        Returns
        -------
        DataLoader
            The validation set
        """
        # Create a copy of dataloader parameters to avoid side effects
        dataloader_params = self.dataloader_params.copy()
        # Always set drop_last to False for validation to ensure all samples are evaluated
        if "drop_last" in dataloader_params:
            dataloader_params["drop_last"] = False

        return DataLoader(
            self._create_dataset("validation", use_augmentation=False),
            shuffle=False,
            **dataloader_params,
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "EMGDatasetLoader":
        """Create an EMGDatasetLoader from a configuration dictionary.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary

        Returns
        -------
        EMGDatasetLoader
            The dataset loader created from the configuration
        """
        return cls(**config)
