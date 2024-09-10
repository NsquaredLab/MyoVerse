from pathlib import Path
import pickle as pkl


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
        Path where to save the DocOctopy format pkl files.
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
