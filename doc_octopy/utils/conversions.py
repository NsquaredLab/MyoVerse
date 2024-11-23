from pathlib import Path

import pickle as pkl
from typing import Optional

import numpy as np
from scipy.io import loadmat
from tqdm import tqdm


def convert_myogestic_to_dococtopy(
    myogestic_data_folder_path: Path, save_folder_path: Path
) -> None:
    """
    Convert a MyoGestic dataset to a DocOctopy dataset.

    Parameters
    ----------
    myogestic_data_folder_path : Path
        Path to the folder holding the MyoGestic recordings.
    save_folder_path : Path
        where to save the DocOctopy format pkl files.
    """
    emg_data = {}
    kinematics_data = {}

    for file in myogestic_data_folder_path.iterdir():
        if file.suffix == ".pkl":
            with open(file, "rb") as f:
                data = pkl.load(f)

            label = data["task"]
            emg_data[label] = data["emg"]
            kinematics_data[label] = data["kinematics"]

    save_folder_path.mkdir(parents=True, exist_ok=True)
    with open(save_folder_path / "emg.pkl", "wb") as f:
        pkl.dump(emg_data, f)
    with open(save_folder_path / "kinematics.pkl", "wb") as f:
        pkl.dump(kinematics_data, f)


def convert_otb_plus_mat_to_pkl(
    mat_files_dir_path: Path | str,
    output_dir_path: Path | str,
    gt_pkl_file_path: Optional[Path | str] = None,
    gt_fsamp: Optional[float] = None,
    mat_to_gt_mapping: Optional[dict[str, str| int]] = None,
) -> None:
    """
    Converts .mat files generated from OTB+ software to pickle files.

    Parameters
    ----------
    mat_files_dir_path : Path | str
        Path to the folder holding the .mat files.
    output_dir_path : Path | str
        Path to the folder where the pickle files will be saved.
    gt_pkl_file_path : Optional[Path | str], optional
        Path to the ground truth pickle file that should be used to link to the .mat files, by default None.
        .. note:: It is assumed that the .mat files and the ground truth pkl file were started at the same time. The recordings will be trimmed to the minimum length of the two.
    gt_fsamp : Optional[float], optional
        Sampling frequency of the ground truth data, by default None.
    mat_to_gt_mapping : Optional[dict[str, str| int]], optional
        Mapping between the .mat file names and the keys in the ground truth pickle file, by default None.

    Raises
    ------
    ValueError
        If gt_pkl_file_path is provided, gt_fsamp must be provided.
    ValueError
        If gt_pkl_file_path is provided, mat_to_gt_mapping must be provided.
    """
    output_data_dict = {}
    gt_interpolated = {}
    for mat_file__path in tqdm(list(Path(mat_files_dir_path).rglob("*.mat"))):
        with mat_file__path.open("rb") as mat_file:
            mat_data = loadmat(mat_file)

        mat_fsamp = np.squeeze(mat_data["SamplingFrequency"])
        task_id = mat_file__path.stem

        mat_data_for_task = mat_data["Data"]

        if gt_pkl_file_path is not None:
            if mat_to_gt_mapping is None:
                raise ValueError("mat_to_gt_mapping must be provided when gt_pkl_file_path is provided.")
            if gt_fsamp is None:
                raise ValueError("gt_fsamp must be provided when gt_pkl_file_path is provided.")

            gt_pkl_file__path = Path(gt_pkl_file_path)
            with gt_pkl_file__path.open("rb") as f:
                gt_data = pkl.load(f)

            task_id = str(mat_to_gt_mapping[mat_file__path.stem])
            length = int(gt_data[task_id].shape[-1] / gt_fsamp * mat_fsamp)

            mat_data_for_task = mat_data_for_task[:length]

            temp = np.empty(gt_data[task_id].shape[:-1] + (length,))
            for compound_index in np.ndindex(gt_data[task_id].shape[:-1]):
                temp[compound_index] = np.interp(
                    np.arange(0, length, 1) / mat_fsamp,
                    np.arange(0, gt_data[task_id].shape[-1], 1) / gt_fsamp,
                    gt_data[task_id][compound_index],
                )
            gt_interpolated[task_id] = temp.astype(np.float32)

        output_data_dict[task_id] = mat_data_for_task.T

    output_dir_path = Path(output_dir_path)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    pkl.dump(output_data_dict, (Path(output_dir_path) / "emg.pkl").open("wb"))

    if gt_pkl_file_path is not None:
        pkl.dump(
            gt_interpolated, (Path(output_dir_path) / "gt_interpolated.pkl").open("wb")
        )
