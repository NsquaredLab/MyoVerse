from pathlib import Path

import lightning as L
import mlflow
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging
from lightning.pytorch.loggers import MLFlowLogger
from scipy.signal import butter
from tqdm import tqdm

import myoverse
from myoverse.datasets.filters.generic import ApplyFunctionFilter, IdentityFilter, IndexDataFilter
from myoverse.datasets.filters.temporal import SOSFrequencyFilter
from myoverse.datasets.loader import EMGDatasetLoader
from myoverse.datasets.supervised import EMGDataset
from myoverse.datatypes import _Data, KinematicsData
from myoverse.models.definitions.raul_net.online.v16 import RaulNetV16
from myoverse.utils.visualization import plot_predicted_and_ground_truth_kinematics


class Workflow:
    DATA_DIR__PATH = Path(r"/home/oj98yqyk/work/datasets/13_subjects")
    DATA__AI_FORMAT__PATH = DATA_DIR__PATH / "AI Format"
    DATA__AI_FORMAT__SAVE_PATH = Path(r"/home/oj98yqyk/work/datasets/13_subjects_processed/AI Format") 
    MLFLOW__SAVE_PATH = Path(r"/home/oj98yqyk/work/datasets/13_subjects_processed/models_2/mlflow")

    class CustomDataClass(_Data):
        def __init__(self, raw_data, sampling_frequency=None):
            # Initialize parent class with raw data
            super().__init__(raw_data.reshape(1, 60), sampling_frequency, nr_of_dimensions_when_unchunked=2)

    def __init__(self):

        self.workflow()

    def create_dataset(self):
        EMG__PATH = self.DATA__AI_FORMAT__PATH / f"Sub1_emg.pkl"
        KINEMATICS__PATH = self.DATA__AI_FORMAT__PATH / f"Sub1_kinematics.pkl"

        tasks_to_use = ["2", "4", "6", "8", "10", "15", "18", "19"]

        dataset = EMGDataset(
            emg_data_path=EMG__PATH,
            ground_truth_data_path=KINEMATICS__PATH,
            ground_truth_data_type="kinematics",
            sampling_frequency=2048,
            tasks_to_use=tasks_to_use,
            save_path=self.DATA__AI_FORMAT__SAVE_PATH,
            emg_filter_pipeline_before_chunking=[],
            emg_representations_to_filter_before_chunking=[["Input"]],
            emg_filter_pipeline_after_chunking=[
                [
                    IdentityFilter(is_output=True, input_is_chunked=True),
                    SOSFrequencyFilter(
                        sos_filter_coefficients=butter(
                            4, 20, "lowpass", output="sos", fs=2048
                        ),
                        is_output=True,
                        input_is_chunked=True,
                    ),
                ]
            ],
            emg_representations_to_filter_after_chunking=[["Last"]],
            ground_truth_filter_pipeline_before_chunking=[
                [
                    ApplyFunctionFilter(
                        function=np.reshape, name="Reshape", newshape=(63, -1), input_is_chunked=False
                    ),
                    IndexDataFilter(indices=(slice(3, 63),), input_is_chunked=False),
                ]
            ],
            ground_truth_representations_to_filter_before_chunking=[["Input"]],
            ground_truth_filter_pipeline_after_chunking=[
                [
                    ApplyFunctionFilter(
                        function=np.mean, name="Mean", axis=-1, is_output=True, input_is_chunked=True
                    )
                ]
            ],
            ground_truth_representations_to_filter_after_chunking=[["Last"]],
            chunk_shift=64,
            chunk_size=320,
            testing_split_ratio=0.2,
            validation_split_ratio=0.2,
            debug_level=0,
        )

        dataset.create_dataset()

    def train_model(self):
        torch.set_float32_matmul_precision("medium")
        torch.backends.cudnn.benchmark = True

        loader = EMGDatasetLoader(
            self.DATA__AI_FORMAT__SAVE_PATH.resolve(),
            dataloader_params={
                "batch_size": 64,
                "drop_last": True,
                "num_workers": 10,
                "pin_memory": True,
                "persistent_workers": True,
            },
            target_data_class=self.CustomDataClass,
        )

        # Create the model
        model = RaulNetV16(
            learning_rate=1e-4,
            nr_of_input_channels=2,
            input_length__samples=320,
            nr_of_outputs=60,
            nr_of_electrode_grids=5,
            nr_of_electrodes_per_grid=64,
            # Multiply following by 4, 8, 16 to have a useful network
            cnn_encoder_channels=(64, 32, 32),
            mlp_encoder_channels=(128, 128),
            event_search_kernel_length=31,
            event_search_kernel_stride=8,
        )

        logger = MLFlowLogger(
            experiment_name=f"Sub1",
            save_dir=str(self.MLFLOW__SAVE_PATH),
            run_name="all",
            log_model=True,
        )

        trainer = L.Trainer(
            callbacks=[
                ModelCheckpoint(
                    monitor="val_loss", save_top_k=-1, save_last=True, every_n_epochs=5
                ),
                StochasticWeightAveraging(
                    swa_lrs=10 ** (-4), swa_epoch_start=0.5, annealing_epochs=5, device=None
                ),

            ],
            logger=logger,
            accelerator="auto",
            check_val_every_n_epoch=5,
            devices=1,
            precision="16-mixed",
            max_epochs=25,
            log_every_n_steps=50,
            enable_checkpointing=True,
            enable_model_summary=True,
            deterministic=False,
        )

        trainer.fit(model, datamodule=loader)

        logger._mlflow_client.log_artifact(logger._run_id, __file__)
        logger._mlflow_client.log_artifact(
            logger._run_id, myoverse.models.definitions.raul_net.online.v16.__file__
        )

    def visualize_results(self):
        # Placeholder for visualization logic
        mlflow.set_tracking_uri("file:///" + str(self.MLFLOW__SAVE_PATH.as_posix()))

        experiment_id = (
            mlflow.tracking.MlflowClient()
            .get_experiment_by_name("Sub1")
            .experiment_id
        )

        artifact_uri = mlflow.search_runs(
            experiment_id,
            filter_string=f"run_name='all'",
        )["artifact_uri"]

        print(artifact_uri[0].split("/")[-2])

        model = (
            RaulNetV16.load_from_checkpoint(
                list(
                    (
                        self.MLFLOW__SAVE_PATH
                        / Path(*artifact_uri[0].split("/")[-3:-1], "checkpoints")
                    ).rglob("last.ckpt")
                )[0]
            )
            .to("cuda")
            .eval()
        )

        torch.set_float32_matmul_precision("medium")
        torch.backends.cudnn.benchmark = True

        loader = EMGDatasetLoader(
            self.DATA__AI_FORMAT__SAVE_PATH.resolve(),
            shuffle_train=False,
            target_data_class=self.CustomDataClass,
            dataloader_params={
                "batch_size": 64,
                "drop_last": False,
                "num_workers": 10,
                "pin_memory": False,
                "persistent_workers": True,
            },
        )

        predictions = []
        ground_truths = []
        with torch.inference_mode():
            for batch in tqdm(loader.test_dataloader()):
                emg_data, kinematics_data = batch
                emg_data = emg_data.to("cuda")

                predictions.append(model(emg_data).cpu().detach().numpy())
                ground_truths.append(kinematics_data[:, 0, 0].cpu().detach().numpy())

        # predictions = np.concatenate(predictions, axis=0).T
        predictions = np.reshape(np.concatenate(predictions, axis=0).T, (20, 3, -1))
        predictions = np.concatenate(
            [np.zeros((1, 3, predictions.shape[-1])), predictions], axis=0
        )

        # ground_truths = np.concatenate(ground_truths, axis=0).T

        ground_truths = np.reshape(np.concatenate(ground_truths, axis=0).T, (20, 3, -1))
        ground_truths = np.concatenate(
            [np.zeros((1, 3, ground_truths.shape[-1])), ground_truths], axis=0
        )

        predictions = KinematicsData(predictions, 32)
        # predictions.plot("Input", 5, True)
        ground_truths = KinematicsData(ground_truths, 32)
        # ground_truths.plot("Input", 5, True)

        plot_predicted_and_ground_truth_kinematics(
            predictions, ground_truths, "Input", "Input", True, 5
        )

    def workflow(self):
        self.create_dataset()
        self.train_model()
        self.visualize_results()

if __name__ == '__main__':
    Workflow()