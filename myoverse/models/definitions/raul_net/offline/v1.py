"""Model definition not used in any publication"""

from functools import reduce
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import lightning as L
import torch
import torch.optim as optim
from torch import nn
import warnings

warnings.warn(
    "This model definition is archival only and should not be used for new projects.",
    UserWarning,
)

CRITERION = nn.L1Loss()


class ErfAct_2(nn.Module):
    """ErfAct_2 activation function from Biswas et al.

    References
    ----------
    Biswas, K., Kumar, S., Banerjee, S., Pandey, A.K., 2021. ErfAct and PSerf:
    Non-monotonic smooth trainable Activation Functions. arXiv:2109.04386 [cs].
    """

    def __init__(self, gamma=1.0, sigma=1.25):
        super(ErfAct_2, self).__init__()

        self.gamma = nn.Parameter(torch.tensor(gamma), requires_grad=True)
        self.sigma = nn.Parameter(torch.tensor(sigma), requires_grad=True)

    def forward(self, x) -> torch.Tensor:
        return x * torch.erf(self.gamma * torch.log(1 + torch.exp(self.sigma * x)))


class CircularPad(nn.Module):
    """Circular padding layer"""

    def __init__(self):
        super(CircularPad, self).__init__()

    def forward(self, x) -> torch.Tensor:
        x = torch.cat([torch.narrow(x, 2, 3, 2), x, torch.narrow(x, 2, 0, 2)], dim=2)
        x = torch.cat([torch.narrow(x, 3, 48, 16), x, torch.narrow(x, 3, 0, 16)], dim=3)
        return x


class RaulNetV1(L.LightningModule):
    """Model definition used in Sîmpetru et al. [1]_

    Attributes
    ----------
    example_input_array : torch.Tensor
        Used for creating a summery and checking if the model architecture is valid.
    learning_rate : float
        The learning rate.
    nr_of_input_channels : int
        The number of input channels. In Sîmpetru et al. 2.
    nr_of_outputs : int
        The number of outputs. In Sîmpetru et al. 14 DOFs.
    cnn_encoder_channels : Tuple[int, int, int]
        Tuple containing 3 integers defining the cnn encoder channels.
    mlp_encoder_channels : Tuple[int, int]
        Tuple containing 2 integers defining the mlp encoder channels.
    event_search_kernel_length : int
        Integer that sets the length of the kernels searching for action potentials.
    event_search_kernel_stride : int
        Integer that sets the stride of the kernels searching for action potentials.

    Notes
    -----
    .. [1] Sîmpetru, R.C., Osswald, M., Braun, D.I., Oliveira, D.S., Cakici, A.L., Del Vecchio, A., 2022. Accurate Continuous Prediction of 14 Degrees of Freedom of the Hand from Myoelectrical Signals through Convolutive Deep Learning, in: 2022 44th Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC). Presented at the 2022 44th Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC), pp. 702–706. https://doi.org/10.1109/EMBC48229.2022.9870937

    """

    def __init__(
        self,
        example_input_array: torch.Tensor,
        learning_rate: float,
        nr_of_input_channels: int,
        nr_of_outputs: int,
        cnn_encoder_channels: Tuple[int, int, int],
        mlp_encoder_channels: Tuple[int, int],
        event_search_kernel_length: int,
        event_search_kernel_stride: int,
    ):
        super(RaulNetV1, self).__init__()

        self.example_input_array = example_input_array

        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.nr_of_input_channels = nr_of_input_channels
        self.nr_of_outputs = nr_of_outputs

        # parameters to be searched
        self.cnn_encoder_channels = cnn_encoder_channels
        self.mlp_encoder_channels = mlp_encoder_channels
        self.event_search_kernel_length = event_search_kernel_length
        self.event_search_kernel_stride = event_search_kernel_stride

        self.channels, self.samples = example_input_array.shape[3:]

        # CNN encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(
                self.nr_of_input_channels,
                self.cnn_encoder_channels[0],
                kernel_size=(1, 1, self.event_search_kernel_length),
                stride=(1, 1, self.event_search_kernel_stride),
                bias=False,
            ),
            nn.BatchNorm3d(self.cnn_encoder_channels[0]),
            ErfAct_2(),
            nn.Dropout3d(p=0.25),
            CircularPad(),
            nn.Conv3d(
                self.cnn_encoder_channels[0],
                self.cnn_encoder_channels[1],
                kernel_size=(5, 32, 18),  # was 3 before
                dilation=(1, 2, 1),
                bias=False,
            ),
            nn.BatchNorm3d(self.cnn_encoder_channels[1]),
            ErfAct_2(),
            nn.Conv3d(
                self.cnn_encoder_channels[1],
                self.cnn_encoder_channels[2],
                kernel_size=(5, 9, 1),
                bias=False,
            ),  # was 3 before
            nn.BatchNorm3d(self.cnn_encoder_channels[2]),
            ErfAct_2(),
        )

        # MLP encoder
        self.flat = nn.Flatten()
        self.dropout = nn.Dropout()
        self.l1 = nn.Linear(
            reduce(
                lambda x, y: x * int(y),
                self.encoder(self.example_input_array).shape[1:],
                1,
            ),
            self.mlp_encoder_channels[0],
        )
        self.af1 = ErfAct_2()
        self.l2 = nn.Linear(self.mlp_encoder_channels[0], self.mlp_encoder_channels[1])
        self.af2 = ErfAct_2()
        self.outputs = nn.Linear(self.mlp_encoder_channels[1], self.nr_of_outputs)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs) -> torch.Tensor:
        # CNN encoder
        x = self.encoder(inputs)
        # MLP encoder
        x = self.flat(x)
        x = self.dropout(x)
        x = self.l1(x)
        x = self.af1(x)
        x = self.l2(x)
        x = self.af2(x)
        x = self.outputs(x)
        x = self.sigmoid(x)

        return x

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=1e-4
        )

        lr_scheduler = {
            "scheduler": optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate * (10**1.5),
                steps_per_epoch=int(len(self.trainer.datamodule.train_dataloader())),
                epochs=self.trainer.max_epochs,
                anneal_strategy="cos",
                three_phase=True,
                div_factor=10**1.5,
                final_div_factor=1e2,
            ),
            "name": "learning_rate",
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [lr_scheduler]

    def training_step(
        self, train_batch, batch_idx: int
    ) -> Optional[Union[torch.Tensor, Dict[str, Any]]]:
        inputs, ground_truths = train_batch
        scores_dict = {"loss": CRITERION(self(inputs), ground_truths)}

        if scores_dict["loss"].isnan().item():
            return None

        self.log_dict(scores_dict, prog_bar=True, logger=False, on_epoch=True)
        self.log_dict(
            {f"train/{k}": v for k, v in scores_dict.items()},
            prog_bar=False,
            logger=True,
            on_epoch=True,
            on_step=False,
        )

        return scores_dict

    def validation_step(
        self, batch, batch_idx
    ) -> Optional[Union[torch.Tensor, Dict[str, Any]]]:
        inputs, ground_truths = batch
        scores_dict = {"val_loss": CRITERION(self(inputs), ground_truths)}

        self.log_dict(scores_dict, prog_bar=True, logger=False, on_epoch=True)

        return scores_dict

    def test_step(
        self, batch, batch_idx
    ) -> Optional[Union[torch.Tensor, Dict[str, Any]]]:
        inputs, ground_truths = batch
        scores_dict = {"loss": CRITERION(self(inputs), ground_truths)}

        self.log_dict(scores_dict, prog_bar=True, logger=False, on_epoch=True)
        self.log_dict(
            {f"test/{k}": v for k, v in scores_dict.items()},
            prog_bar=False,
            logger=True,
            on_epoch=False,
            on_step=True,
        )

        return scores_dict
