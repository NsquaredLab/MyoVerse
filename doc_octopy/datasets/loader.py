from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import zarr
import numpy as np
import lightning as L
from torch.utils.data import DataLoader, Dataset

from doc_octopy.datasets.filters.generic import IdentityFilter, FilterBaseClass
from doc_octopy.datatypes import EMGData


class EMGDatasetLoader(L.LightningDataModule):
    """Dataset loader for the EMG dataset.

    Parameters
    ----------
    data_path : Path
        The path to the zarr file
    seed : Optional[int], optional
        The seed for the random number generator, by default None
    dataloader_parameters : Dict[str, Any], optional
        The parameters for the DataLoader, by default None
    shuffle_training_data : bool, optional
        Whether to shuffle the training data, by default True
    input_type : numpy.dtype, optional
        The type of the input data, by default np.float32
    ground_truth_type : numpy.dtype, optional
        The type of the ground_truth data, by default np.float32
    ground_truth_name : str, optional
        The name of the ground truth data, by default "ground_truth"
    augmentation_pipeline : list[list[FilterBaseClass]], optional
        The augmentation pipeline, by default [[IdentityFilter(is_output=True)]]
    augmentation_probabilities : Sequence[float], optional
        The probabilities for the augmentation pipeline, by default (1,)
        The sum of the probabilities must be equal to 1 and the number of probabilities must be equal to the number of augmentation sequences.
    """

    class _EMGDatasetLoader(Dataset):
        def __init__(
            self,
            zarr_file: Path,
            subset_name: str,
            ground_truth_name: str,
            input_type=np.float32,
            ground_truth_type=np.float32,
            augmentation_pipeline: list[list[FilterBaseClass]] = [  # noqa
                [IdentityFilter(is_output=True)]
            ],
            augmentation_probabilities: Sequence[float] = (1,),
        ):
            self.zarr_file = zarr_file
            self.subset_name = subset_name
            self.ground_truth_name = ground_truth_name

            self._emg_data = zarr.open(str(self.zarr_file))[self.subset_name]["emg"]
            self._emg_data = {
                key: self._emg_data[key] for key in self._emg_data.array_keys()
            }
            self._ground_truth_data = zarr.open(str(self.zarr_file))[self.subset_name][
                self.ground_truth_name
            ]
            self._ground_truth_data = {
                key: self._ground_truth_data[key]
                for key in self._ground_truth_data.array_keys()
            }

            try:
                self.length = list(self._emg_data.values())[0].shape[0]
            except IndexError:
                self.length = 0

            self.input_type = input_type
            self.ground_truth_type = ground_truth_type

            self.augmentation_pipeline = augmentation_pipeline
            self.augmentation_probabilities = augmentation_probabilities

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            input_data = []

            augmentation_chosen = self.augmentation_pipeline[
                np.random.choice(
                    len(self.augmentation_pipeline), p=self.augmentation_probabilities
                )
            ]

            for v in self._emg_data.values():
                temp = EMGData(v[idx], sampling_frequency=2048)
                temp.apply_filter_sequence(
                    augmentation_chosen, representation_to_filter="Input"
                )
                input_data.append(list(temp.output_representations.values())[0])

            return np.array(input_data).astype(self.input_type), np.array(
                [v[idx] for v in self._ground_truth_data.values()]
            ).astype(self.ground_truth_type)

    def __init__(
        self,
        data_path: Path,
        seed: Optional[int] = None,
        dataloader_parameters: Dict[str, Any] = None,
        shuffle_training_data: bool = True,
        input_type=np.float32,
        ground_truth_type=np.float32,
        ground_truth_name: str = "ground_truth",
        augmentation_pipeline: list[list[FilterBaseClass]] = [  # noqa
            [IdentityFilter(is_output=True)]
        ],
        augmentation_probabilities: Sequence[float] = (1,),
    ):
        """Initializes the dataset.

        Attributes
        ----------
        data_path : Path
            The path to the HDF5 file
        seed : Optional[int], optional
            The seed for the random number generator, by default None
        dataloader_parameters : Dict[str, Any], optional
            The parameters for the DataLoader, by default None
        shuffle_training_data : bool, optional
            Whether to shuffle the training data, by default True
        input_type : np.dtype, optional
            The type of the input data, by default np.float32
        ground_truth_type : np.dtype, optional
            The type of the label data, by default np.float32
        ground_truth_name : bool, optional
            The name of the ground truth data, by default "ground_truth"
        augmentation_pipeline : list[list[FilterBaseClass]], optional
            The augmentation pipeline, by default [[IdentityFilter(is_output=True)]]
        augmentation_probabilities : Sequence[float], optional
            The probabilities for the augmentation pipeline, by default (1,)
            The sum of the probabilities must be equal to 1 and the number of probabilities must be equal to the number of augmentation sequences.
        """
        super().__init__()
        self.data_path = data_path
        self.seed = seed

        if dataloader_parameters is None:
            raise ValueError("DataLoader parameters must be set!")
        self.dataloader_parameters = dataloader_parameters

        self.shuffle_training_data = shuffle_training_data

        self.input_type = input_type
        self.ground_truth_type = ground_truth_type

        self.ground_truth_name = ground_truth_name

        self.augmentation_pipeline = augmentation_pipeline
        self.augmentation_probabilities = augmentation_probabilities

        # check if augmentation probabilities are equal to the number of augmentations filter sequences and that the sum is 1
        if len(self.augmentation_pipeline) != len(self.augmentation_probabilities):
            raise ValueError(
                "The number of augmentation sequences must be equal to the number of probabilities"
            )
        if sum(self.augmentation_probabilities) != 1:
            raise ValueError("The sum of the probabilities must be equal to 1")

    def train_dataloader(self) -> DataLoader:
        """Returns the training set as a DataLoader.

        Returns
        -------
        DataLoader
            The training set

        """
        return DataLoader(
            self._EMGDatasetLoader(
                self.data_path,
                subset_name="training",
                ground_truth_name=self.ground_truth_name,
                input_type=self.input_type,
                ground_truth_type=self.ground_truth_type,
                augmentation_pipeline=self.augmentation_pipeline,
                augmentation_probabilities=self.augmentation_probabilities,
            ),
            shuffle=self.shuffle_training_data,
            **self.dataloader_parameters,
        )

    def test_dataloader(self) -> DataLoader:
        """Returns the testing set as a DataLoader.

        Returns
        -------
        DataLoader
            The testing set

        """
        return DataLoader(
            self._EMGDatasetLoader(
                self.data_path,
                subset_name="testing",
                ground_truth_name=self.ground_truth_name,
                input_type=self.input_type,
                ground_truth_type=self.ground_truth_type,
            ),
            shuffle=False,
            **self.dataloader_parameters,
        )

    def val_dataloader(self) -> DataLoader:
        """Returns the testing set as a DataLoader.

        Returns
        -------
        DataLoader
            The testing set

        """
        dataloader_prams = self.dataloader_parameters.copy()
        if "drop_last" in dataloader_prams:
            dataloader_prams["drop_last"] = False

        return DataLoader(
            self._EMGDatasetLoader(
                self.data_path,
                subset_name="validation",
                ground_truth_name=self.ground_truth_name,
                input_type=self.input_type,
                ground_truth_type=self.ground_truth_type,
            ),
            shuffle=False,
            **dataloader_prams,
        )
