"""Model definition not used in any publication"""
from functools import reduce
from typing import Any, Dict, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch import nn

from doc_octopy.models.components.activation_functions import SMU
from doc_octopy.models.components.utils import CircularPad


class ElectrodeWiseDropout(torch.nn.Module):
    def __init__(self, p: float):
        super(ElectrodeWiseDropout, self).__init__()
        self.p = p
        if self.p < 0 or self.p > 1:
            raise ValueError("p must be a probability")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            for i in range(x.shape[0]):
                x[i, :, torch.bernoulli(torch.full(x.shape[2:4], self.p, device=x.device)).to(torch.bool)] = 0
        return x


class RaulNetV6(pl.LightningModule):
    """Model definition not used in any publication

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
        super(RaulNetV6, self).__init__()
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

        self.electrode_wise_dropout = ElectrodeWiseDropout(p=0.10)

        self.cnn_encoder = nn.Sequential(
            nn.Conv3d(
                self.nr_of_input_channels,
                self.cnn_encoder_channels[0],
                kernel_size=(1, 1, self.event_search_kernel_length),
                stride=(1, 1, self.event_search_kernel_stride),
            ),
            SMU(),
            nn.BatchNorm3d(self.cnn_encoder_channels[0]),
            # nn.Dropout3d(p=0.25),
            CircularPad(),
            nn.Conv3d(
                self.cnn_encoder_channels[0],
                self.cnn_encoder_channels[1],
                kernel_size=(5, 32, 18),
                dilation=(1, 2, 1),
            ),
            SMU(),
            nn.BatchNorm3d(self.cnn_encoder_channels[1]),
            nn.Conv3d(self.cnn_encoder_channels[1], self.cnn_encoder_channels[2], kernel_size=(5, 9, 1)),
            SMU(),
            nn.BatchNorm3d(self.cnn_encoder_channels[2]),
            nn.Flatten(),
            nn.Dropout(p=0.40),
        )

        self.mlp_encoder = nn.Sequential(
            nn.Linear(
                reduce(
                    lambda x, y: x * int(y),
                    self.cnn_encoder(
                        torch.rand((1, self.nr_of_input_channels, 5, 64, self.input_length__samples))
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
        x = torch.stack(torch.split(inputs, 64, dim=2), dim=2)
        x = self.electrode_wise_dropout(x)
        x = self._normalize_input(x)

        x = self.cnn_encoder(x)
        x = self.mlp_encoder(x)

        return x

    def _normalize_input(self, inputs: torch.Tensor) -> torch.Tensor:
        mins = torch.min(torch.min(inputs, dim=3, keepdim=True)[0], dim=4, keepdim=True)[0].expand(inputs.shape)
        maxs = torch.max(torch.max(inputs, dim=3, keepdim=True)[0], dim=4, keepdim=True)[0].expand(inputs.shape)

        return 2 * torch.div(torch.sub(inputs, mins), torch.sub(maxs, mins)) - 1

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=0.01)

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
