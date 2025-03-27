"""
Training a deep learning model
===========================

This example shows how to train a deep learning model using the dataset created in the previous example.
"""

# %%
# Loading the dataset
# -------------------
# To load the dataset we need to use the EMGDatasetLoader class.
#
# Two parameters are required:
#
# - data_path: Path to the dataset file.
#
# - dataloader_parameters: Parameters for the DataLoader.
from pathlib import Path

from myoverse.datasets.loader import EMGDatasetLoader
from myoverse.datatypes import _Data


# Create a class to handle the target data that doesn't enforce a specific shape
class CustomDataClass(_Data):
    def __init__(self, raw_data, sampling_frequency=None):
        # Initialize parent class with raw data
        super().__init__(raw_data.reshape(1, 60), sampling_frequency)

    # Implement required abstract method
    def _check_if_chunked(self, data):
        # For simplicity, we'll assume data is always chunked
        return True

    # Implement abstract method plot (can be a minimal implementation)
    def plot(self, *args, **kwargs):
        pass


# Let's use the built-in IdentityFilter which just passes data through
loader = EMGDatasetLoader(
    Path(r"../data/dataset.zarr").resolve(),
    dataloader_params={"batch_size": 16, "drop_last": True},
    target_data_class=CustomDataClass,
)

# %%
# Training the model
# ------------------
from myoverse.models.definitions.raul_net.online.v16 import RaulNetV16
import lightning as L

# Create the model
model = RaulNetV16(
    learning_rate=1e-4,
    nr_of_input_channels=2,
    input_length__samples=192,
    nr_of_outputs=60,
    nr_of_electrode_grids=5,
    nr_of_electrodes_per_grid=64,
    # Multiply following by 4, 8, 16 to have a useful network
    cnn_encoder_channels=(4, 1, 1),
    mlp_encoder_channels=(8, 8),
    event_search_kernel_length=31,
    event_search_kernel_stride=8,
)

trainer = L.Trainer(
    accelerator="auto",
    devices=1,
    precision="16-mixed",
    max_epochs=1,
    log_every_n_steps=50,
    logger=None,
    enable_checkpointing=False,
    deterministic=False,
)

trainer.fit(model, datamodule=loader)
