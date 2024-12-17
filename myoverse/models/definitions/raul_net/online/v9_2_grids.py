"""Model definition not used in any publication"""
from functools import reduce
from typing import Any, Dict, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch import nn

from myoverse.models.components.activation_functions import SMU


class CircularPad(nn.Module):
    """Circular padding layer"""

    def __init__(self):
        super(CircularPad, self).__init__()

    def forward(self, x) -> torch.Tensor:
        return torch.cat([torch.narrow(x, 3, 48, 16), x, torch.narrow(x, 3, 0, 16)], dim=3)


class RaulNetV9(pl.LightningModule):
    """Model definition used in Sîmpetru et al.

    Attributes
    ----------
    learning_rate : float
        The learning rate.
    nr_of_input_channels : int
        The number of input channels.
    nr_of_outputs : int
        The number of outputs.
    cnn_encoder_channels : Tuple[int, int, int]
        Tuple containing 3 integers defining the cnn encoder channels.
    mlp_encoder_channels : Tuple[int, int]
        Tuple containing 2 integers defining the mlp encoder channels.
    event_search_kernel_length : int
        Integer that sets the length of the kernels searching for action potentials.
    event_search_kernel_stride : int
        Integer that sets the stride of the kernels searching for action potentials.
    """

    def __init__(
        self,
        learning_rate: float,
        nr_of_input_channels: int,
        input_length__samples: int,
        nr_of_outputs: int,
        cnn_encoder_channels: Tuple[int, int, int],
        mlp_encoder_channels: Tuple[int, int],
        event_search_kernel_length: int,
        event_search_kernel_stride: int,
    ):
        super(RaulNetV9, self).__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.nr_of_input_channels = nr_of_input_channels
        self.nr_of_outputs = nr_of_outputs
        self.input_length__samples = input_length__samples

        self.cnn_encoder_channels = cnn_encoder_channels
        self.mlp_encoder_channels = mlp_encoder_channels
        self.event_search_kernel_length = event_search_kernel_length
        self.event_search_kernel_stride = event_search_kernel_stride

        self.criterion = nn.L1Loss()

        self.cnn_encoder = nn.Sequential(
            nn.Conv3d(
                self.nr_of_input_channels,
                self.cnn_encoder_channels[0],
                kernel_size=(1, 1, self.event_search_kernel_length),
                stride=(1, 1, self.event_search_kernel_stride),
            ),
            SMU(),
            nn.BatchNorm3d(self.cnn_encoder_channels[0], track_running_stats=False),
            nn.Dropout3d(p=0.25),
            CircularPad(),
            nn.Conv3d(
                self.cnn_encoder_channels[0], self.cnn_encoder_channels[1], kernel_size=(1, 32, 18), dilation=(1, 2, 1)
            ),
            SMU(),
            nn.BatchNorm3d(self.cnn_encoder_channels[1], track_running_stats=False),
            nn.Conv3d(self.cnn_encoder_channels[1], self.cnn_encoder_channels[2], kernel_size=(2, 9, 1)),
            SMU(),
            nn.BatchNorm3d(self.cnn_encoder_channels[2], track_running_stats=False),
            nn.Flatten(),
            nn.Dropout(p=0.25),
        )

        self.mlp_encoder = nn.Sequential(
            nn.Linear(
                reduce(
                    lambda x, y: x * int(y),
                    self.cnn_encoder(
                        torch.rand((1, self.nr_of_input_channels, 2, 64, self.input_length__samples))
                    ).shape[1:],
                    1,
                ),
                self.mlp_encoder_channels[0],
            ),
            SMU(),
            nn.Linear(self.mlp_encoder_channels[0], self.mlp_encoder_channels[1]),
            SMU(),
            nn.Linear(self.mlp_encoder_channels[1], self.nr_of_outputs),
        )

    def forward(self, inputs) -> torch.Tensor:
        x = torch.stack(inputs[:, [0], 192:].split(64, dim=2), dim=2)

        x = self.cnn_encoder(x)
        x = self.mlp_encoder(x)

        return x

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=0.1)

        lr_scheduler = {
            "scheduler": optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate * (10**1.5),
                total_steps=self.trainer.estimated_stepping_batches,
                anneal_strategy="cos",
                three_phase=False,
                div_factor=10**1.5,
                final_div_factor=1e3,
            ),
            "name": "OncCycleLR",
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [lr_scheduler]

    def training_step(self, train_batch, batch_idx: int) -> Optional[Union[torch.Tensor, Dict[str, Any]]]:
        inputs, ground_truths = train_batch

        prediction = self(inputs)
        scores_dict = {"loss": self.criterion(prediction, ground_truths)}

        if scores_dict["loss"].isnan().item():
            return None

        self.log_dict(scores_dict, prog_bar=True, logger=False, on_epoch=True)
        self.log_dict(
            {f"train/{k}": v for k, v in scores_dict.items()}, prog_bar=False, logger=True, on_epoch=True, on_step=False
        )

        return scores_dict

    def validation_step(self, batch, batch_idx) -> Optional[Union[torch.Tensor, Dict[str, Any]]]:
        inputs, ground_truths = batch

        prediction = self(inputs)
        scores_dict = {"val_loss": self.criterion(prediction, ground_truths)}

        self.log_dict(scores_dict, prog_bar=True, logger=False, on_epoch=True)

        return scores_dict

    def test_step(self, batch, batch_idx) -> Optional[Union[torch.Tensor, Dict[str, Any]]]:
        inputs, ground_truths = batch

        prediction = self(inputs)
        scores_dict = {"loss": self.criterion(prediction, ground_truths)}

        self.log_dict(scores_dict, prog_bar=True, logger=False, on_epoch=True)
        self.log_dict(
            {f"test/{k}": v for k, v in scores_dict.items()}, prog_bar=False, logger=True, on_epoch=False, on_step=True
        )

        return scores_dict
