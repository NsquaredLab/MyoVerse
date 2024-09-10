import pickle
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import zarr
from tqdm import tqdm

from doc_octopy.datasets.filters._template import FilterBaseClass, EMGAugmentation
from doc_octopy.datasets.filters.generic import ChunkizeDataFilter
from doc_octopy.datatypes import DATA_TYPES_MAP, _Data


def _split_data(data: np.ndarray, split_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    split_amount = int(data.shape[0] * split_ratio / 2)
    middle_index = data.shape[0] // 2

    mask = np.ones(data.shape[0], dtype=bool)
    mask[middle_index - split_amount : middle_index + split_amount] = False

    return data[mask], data[~mask]


def _add_to_dataset(group: zarr.Group, data: Optional[np.ndarray], name: str):
    if data is None:
        return

    try:
        group[name].append(data)
    except KeyError:
        group.create_dataset(
            name, data=data, shape=data.shape, chunks=(1, *data.shape[1:])
        )


class EMGDataset:
    """
    Class for creating a dataset from EMG and ground truth data.

    Parameters
    ----------
    emg_data_path : pathlib.Path
        Path to the EMG data file. It should be a pickle file containing a dictionary with the keys being the task
        number and the values being a numpy array of shape (n_channels, n_samples).
    ground_truth_data_path : pathlib.Path
        Path to the ground truth data file. It should be a pickle file containing a dictionary with the keys being the
        task number and the values being a numpy array of custom shape (..., n_samples). The custom shape can be
        anything, but the last dimension should be the same as the EMG data.
    tasks_to_use : Sequence[str]
        Sequence of strings containing the task numbers to use. If empty, all tasks will be used.
    save_path : pathlib.Path
        Path to save the dataset to. It should be a zarr file.
    emg_filter_pipeline_before_chunking : list[list[FilterBaseClass]]
        Sequence of filters to apply to the EMG data before chunking. The filters should inherit from
        FilterBaseClass.
    emg_filter_pipeline_after_chunking : list[list[FilterBaseClass]]
        Sequence of filters to apply to the EMG data after chunking. The filters should inherit from
        FilterBaseClass.
    ground_truth_filter_pipeline_before_chunking : list[list[FilterBaseClass]]
        Sequence of filters to apply to the ground truth data before chunking. The filters should inherit from
        FilterBaseClass.
    ground_truth_filter_after_pipeline_chunking : list[list[FilterBaseClass]]
        Sequence of filters to apply to the ground truth data after chunking. The filters should inherit from
        FilterBaseClass.
    chunk_size : int
        Size of the chunks to create from the data.
    chunk_shift : int
        Shift between the chunks.
    testing_split_ratio : float
        Ratio of the data to use for testing. The data will be split in the middle. The first half will be used for
        training and the second half will be used for testing. If 0, no data will be used for testing.
    validation_split_ratio : float
        Ratio of the data to use for validation. The data will be split in the middle. The first half will be used for
        training and the second half will be used for validation. If 0, no data will be used for validation.
    augmentation_pipelines : list[list[EMGAugmentation]]
        Sequence of augmentation_pipelines to apply to the training data. The augmentation_pipelines should inherit from
        EMGAugmentation.
    amount_of_chunks_to_augment_at_once : int
        Amount of chunks to augment at once. This is done to speed up the process.

    Methods
    -------
    create_dataset()
        Creates the dataset.

    """

    def __init__(
        self,
        emg_data_path: Path = Path("REPLACE ME"),
        ground_truth_data_path: Path = Path("REPLACE ME"),
        ground_truth_data_type: str = "kinematics",
        sampling_frequency: float = 0.0,
        tasks_to_use: Sequence[str] = (),
        save_path: Path = Path("REPLACE ME"),
        emg_filter_pipeline_before_chunking: list[list[FilterBaseClass]] = (),
        emg_representations_to_filter_before_chunking: list[str] = (),
        emg_filter_pipeline_after_chunking: list[list[FilterBaseClass]] = (),
        emg_representations_to_filter_after_chunking: list[str] = (),
        ground_truth_filter_pipeline_before_chunking: list[list[FilterBaseClass]] = (),
        ground_truth_representations_to_filter_before_chunking: list[str] = (),
        ground_truth_filter_after_pipeline_chunking: list[list[FilterBaseClass]] = (),
        ground_truth_representations_to_filter_after_pipeline_chunking: list[str] = (),
        chunk_size: int = 192,
        chunk_shift: int = 64,
        testing_split_ratio: float = 0.2,
        validation_split_ratio: float = 0.2,
        augmentation_pipelines: list[list[EMGAugmentation]] = (),
        amount_of_chunks_to_augment_at_once: int = 250,
        debug: bool = False,
    ):
        self.emg_data_path = emg_data_path
        self.ground_truth_data_path = ground_truth_data_path

        self.ground_truth_data_type = ground_truth_data_type

        self.sampling_frequency = sampling_frequency

        self.tasks_to_use = tasks_to_use

        self.save_path = save_path

        self.emg_filter_pipeline_before_chunking = emg_filter_pipeline_before_chunking
        self.emg_representations_to_filter_before_chunking = (
            emg_representations_to_filter_before_chunking
        )
        self.ground_truth_filter_pipeline_before_chunking = (
            ground_truth_filter_pipeline_before_chunking
        )
        self.ground_truth_representations_to_filter_before_chunking = (
            ground_truth_representations_to_filter_before_chunking
        )

        self.emg_filter_pipeline_after_chunking = emg_filter_pipeline_after_chunking
        self.emg_representations_to_filter_after_chunking = (
            emg_representations_to_filter_after_chunking
        )
        self.ground_truth_filter_pipeline_after_chunking = (
            ground_truth_filter_after_pipeline_chunking
        )
        self.ground_truth_representations_to_filter_after_pipeline_chunking = (
            ground_truth_representations_to_filter_after_pipeline_chunking
        )

        self.chunk_size = chunk_size
        self.chunk_shift = chunk_shift

        self.testing_split_ratio = testing_split_ratio
        self.validation_split_ratio = validation_split_ratio

        self.augmentation_pipelines = augmentation_pipelines
        self.amount_of_chunks_to_augment_at_once = amount_of_chunks_to_augment_at_once

        self.debug = debug

        self.__tasks_string_length = 0

    def __add_data_to_dataset(self, data: _Data, groups: list[zarr.Group]):
        for k, v in data.output_representations.items():
            validation_data_from_task = None

            if self.testing_split_ratio > 0:
                training_data_from_task, testing_data_from_task = _split_data(
                    v, self.testing_split_ratio
                )

                if self.validation_split_ratio > 0:
                    testing_data_from_task, validation_data_from_task = _split_data(
                        testing_data_from_task, self.validation_split_ratio
                    )

            else:
                training_data_from_task = v
                testing_data_from_task = None

            for g, data_from_task in zip(
                groups,
                (
                    training_data_from_task,
                    testing_data_from_task,
                    validation_data_from_task,
                ),
            ):
                _add_to_dataset(g, data_from_task, k)

    def create_dataset(self):
        with self.emg_data_path.open("rb") as f:
            emg_data = pickle.load(f)
        with self.ground_truth_data_path.open("rb") as f:
            ground_truth_data = pickle.load(f)

        self.save_path.mkdir(parents=True, exist_ok=True)
        dataset = zarr.open(str(self.save_path), mode="w")

        training_group = dataset.create_group("training")
        testing_group = dataset.create_group("testing")
        validation_group = dataset.create_group("validation")

        self.__tasks_string_length = len(
            max(self.tasks_to_use, key=len)
        )  # need this to know the saving dtype for the labels
        for task in tqdm(self.tasks_to_use, desc="Filtering and splitting data"):
            emg_data_from_task = emg_data[task]
            ground_truth_data_from_task = ground_truth_data[task]

            min_length = min(
                emg_data_from_task.shape[-1], ground_truth_data_from_task.shape[-1]
            )

            emg_data_from_task = emg_data_from_task[..., :min_length]
            ground_truth_data_from_task = ground_truth_data_from_task[..., :min_length]

            emg_data_from_task = DATA_TYPES_MAP["emg"](
                input_data=emg_data_from_task,
                sampling_frequency=self.sampling_frequency,
            )
            ground_truth_data_from_task = DATA_TYPES_MAP[self.ground_truth_data_type](
                input_data=ground_truth_data_from_task,
                sampling_frequency=self.sampling_frequency,
            )

            if emg_data_from_task.is_chunked != ground_truth_data_from_task.is_chunked:
                raise ValueError(
                    f"The EMG and ground truth data should have the same chunking status, but the EMG data is "
                    f"{'' if emg_data_from_task.is_chunked else 'not '}chunked and the ground truth data is "
                    f"{'' if ground_truth_data_from_task.is_chunked else 'not '}chunked."
                )

            if self.debug:
                emg_data_from_task.plot_graph()
                ground_truth_data_from_task.plot_graph()

            if not emg_data_from_task.is_chunked["Input"]:
                emg_data_from_task.apply_filter_pipeline(
                    filter_pipeline=self.emg_filter_pipeline_before_chunking,
                    representations_to_filter=self.emg_representations_to_filter_before_chunking,
                )
                ground_truth_data_from_task.apply_filter_pipeline(
                    filter_pipeline=self.ground_truth_filter_pipeline_before_chunking,
                    representations_to_filter=self.ground_truth_representations_to_filter_before_chunking,
                )

                emg_data_from_task.apply_filter(
                    filter=ChunkizeDataFilter(
                        chunk_size=self.chunk_size,
                        chunk_shift=self.chunk_shift,
                        is_output=len(self.emg_filter_pipeline_after_chunking) == 0,
                    ),
                    representation_to_filter="Last",
                )
                chunked_emg_data_from_task = emg_data_from_task

                ground_truth_data_from_task.apply_filter(
                    filter=ChunkizeDataFilter(
                        chunk_size=self.chunk_size,
                        chunk_shift=self.chunk_shift,
                        is_output=len(self.ground_truth_filter_pipeline_after_chunking)
                        == 0,
                    ),
                    representation_to_filter="Last",
                )
                chunked_ground_truth_data_from_task = ground_truth_data_from_task

                if self.debug:
                    chunked_emg_data_from_task.plot_graph()
                    chunked_ground_truth_data_from_task.plot_graph()
            else:
                chunked_emg_data_from_task = emg_data_from_task  # [:min_length]
                i = 0
                temp = []
                while (
                    i + self.amount_of_chunks_to_augment_at_once
                    <= chunked_emg_data_from_task.shape[0]
                ):
                    temp.append(
                        np.concatenate(
                            chunked_emg_data_from_task[
                                i : i + self.amount_of_chunks_to_augment_at_once
                            ],
                            axis=-1,
                        )
                    )
                    i += self.amount_of_chunks_to_augment_at_once
                chunked_emg_data_from_task = np.stack(temp, axis=1)

                chunked_ground_truth_data_from_task = (
                    ground_truth_data_from_task  # [:min_length]
                )

            chunked_emg_data_from_task.apply_filter_pipeline(
                filter_pipeline=self.emg_filter_pipeline_after_chunking,
                representations_to_filter=self.emg_representations_to_filter_after_chunking,
            )

            chunked_ground_truth_data_from_task.apply_filter_pipeline(
                filter_pipeline=self.ground_truth_filter_pipeline_after_chunking,
                representations_to_filter=self.ground_truth_representations_to_filter_after_pipeline_chunking,
            )

            if self.debug:
                chunked_emg_data_from_task.plot_graph()
                chunked_ground_truth_data_from_task.plot_graph()

            for group_name, chunked_data_from_task in zip(
                ["emg", "ground_truth"],
                [chunked_emg_data_from_task, chunked_ground_truth_data_from_task],
            ):
                self.__add_data_to_dataset(
                    chunked_data_from_task,
                    [
                        (
                            g.create_group(group_name)
                            if group_name not in list(g.group_keys())
                            else g[group_name]
                        )
                        for g in (training_group, testing_group, validation_group)
                    ],
                )

            data_length = list(
                chunked_emg_data_from_task.output_representations.values()
            )[-1].shape[0]

            for g in (training_group, testing_group, validation_group):
                _add_to_dataset(
                    g,
                    np.array([task] * data_length)[..., None].astype(
                        f"<U{self.__tasks_string_length}"
                    ),
                    "label",
                )
                _add_to_dataset(
                    g,
                    np.array([self.tasks_to_use.index(task)] * data_length)[
                        ..., None
                    ].astype(np.int8),
                    "class",
                )
                _add_to_dataset(
                    g,
                    np.repeat(
                        np.array(
                            [
                                np.eye(len(self.tasks_to_use))[
                                    self.tasks_to_use.index(task)
                                ]
                            ]
                        ),
                        data_length,
                        axis=0,
                    ).astype(np.int8),
                    "one_hot_class",
                )

        for augmentation_pipeline in self.augmentation_pipelines:
            emg_to_append = {k: [] for k in dataset["training\emg"]}
            ground_truth_to_append = {k: [] for k in dataset["training/ground_truth"]}
            label_to_append = []
            class_to_append = []
            one_hot_class_to_append = []
            for i in tqdm(
                range(
                    list(chunked_emg_data_from_task.output_representations.values())[
                        -1
                    ].shape[0]
                ),
                desc=f"Augmenting with {str(augmentation_pipeline)}",
            ):
                for k in dataset["training\emg"]:
                    temp = DATA_TYPES_MAP["emg"](
                        input_data=dataset["training/emg"][k][i].astype(np.float32),
                        sampling_frequency=self.sampling_frequency,
                    )
                    temp.apply_filter_pipeline(
                        filter_pipeline=[augmentation_pipeline],
                        representations_to_filter=["Last"],
                    )
                    emg_to_append[k].append(temp["Last"])
                for k in dataset["training/ground_truth"]:
                    ground_truth_to_append[k].append(
                        dataset["training/ground_truth"][k][i]
                    )

                label_to_append.append(dataset["training/label"][i])
                class_to_append.append(dataset["training/class"][i])
                one_hot_class_to_append.append(dataset["training/one_hot_class"][i])

                if i % self.amount_of_chunks_to_augment_at_once == 0:
                    for k, v in emg_to_append.items():
                        _add_to_dataset(training_group["emg"], np.array(v), name=k)
                    for k, v in ground_truth_to_append.items():
                        _add_to_dataset(
                            training_group["ground_truth"], np.array(v), name=k
                        )
                    _add_to_dataset(
                        training_group, np.array(label_to_append), name="label"
                    )
                    _add_to_dataset(
                        training_group, np.array(class_to_append), name="class"
                    )
                    _add_to_dataset(
                        training_group,
                        np.array(one_hot_class_to_append),
                        name="one_hot_class",
                    )
                    emg_to_append = {k: [] for k in dataset["training\emg"]}
                    ground_truth_to_append = {
                        k: [] for k in dataset["training/ground_truth"]
                    }
                    label_to_append = []
                    class_to_append = []
                    one_hot_class_to_append = []

            if len(list(emg_to_append.values())[0]) > 0:
                for k, v in emg_to_append.items():
                    _add_to_dataset(training_group["emg"], np.array(v), name=k)
                for k, v in ground_truth_to_append.items():
                    _add_to_dataset(training_group["ground_truth"], np.array(v), name=k)
                _add_to_dataset(
                    training_group, np.array(class_to_append), name=f"class"
                )
                _add_to_dataset(
                    training_group,
                    np.array(one_hot_class_to_append),
                    name=f"one_hot_class",
                )
