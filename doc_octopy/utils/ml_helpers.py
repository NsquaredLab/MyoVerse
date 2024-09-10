import pickle
import platform
import shutil
import time
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import lightning as L
import torch
import torchinfo
import yaml
from lightning.pytorch.loggers import MLFlowLogger


def save_model_lightning(save_path: Path, trainer: L.Trainer, input_example: torch.Tensor):
    Path.mkdir(save_path, parents=True, exist_ok=True)
    torch.save(trainer.model.state_dict(), save_path / "model.pt")

    try:
        with (save_path / "summary.txt").open("w") as f:
            f.write(
                str(torchinfo.summary(trainer.model, input_data=input_example))
                .encode("ascii", "replace")
                .decode("ascii")
            )
    except Exception:
        pass


def set_run_name(logger, run_name: str):
    path = Path(logger.experiment.get_run(logger.run_id).info.artifact_uri[7:]).resolve().parent

    yaml_file = yaml.load((path / "meta.yaml").open("r"), Loader=yaml.FullLoader)
    yaml_file["run_name"] = run_name
    yaml.dump(yaml_file, (path / "meta.yaml").open("w"))

    with (path / "tags" / "mlflow.runName").open("w") as f:
        f.write(run_name)


def save_model_mlflow(logger: MLFlowLogger, trainer: L.Trainer, input_example: torch.Tensor):
    save_model_lightning(  # 7: to remove file:./
        save_path=Path(logger.experiment.get_run(logger.run_id).info.artifact_uri[7:]).resolve(),
        trainer=trainer,
        input_example=input_example,
    )


def save_train_and_def_mlflow(
    logger: MLFlowLogger, train_file_path: Path, def_file_path: Path, dataset_file_path: Path
):
    # 7: to remove file:./
    artifact_path = Path(logger.experiment.get_run(logger.run_id).info.artifact_uri[7:]).resolve()

    shutil.copy(train_file_path, artifact_path / "train.py")
    shutil.copy(def_file_path, artifact_path / "def.py")
    shutil.copy(dataset_file_path, artifact_path / "dataset.py")


def remove_empty_mlflow_runs():
    for file_paths in list(Path("./mlruns").rglob("params")):
        if not any(file_paths.resolve().iterdir()):
            shutil.rmtree(file_paths.resolve().parent)


def save_log_mlflow(logger: MLFlowLogger):
    try:
        artifact_path = Path(logger.experiment.get_run(logger.run_id).info.artifact_uri[7:]).resolve()
        log_file_path = list(Path("../../..").resolve().glob("*.o*"))[0]
        shutil.copy(log_file_path, artifact_path / "log.txt")
    except Exception as e:
        print(e)
    # log_file_path.unlink()
    
def load_model(model, save_path: Path):
    output_model = model(**pickle.load(open(str(list(save_path.rglob("model_hparams.pkl"))[0]), "rb")))
    output_model.load_state_dict(torch.load(str(list(save_path.rglob("model.pt"))[0])))

    return output_model

