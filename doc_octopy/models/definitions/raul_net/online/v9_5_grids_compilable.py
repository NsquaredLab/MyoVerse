"""Model definitions used in the Sîmpetru et al. (2023)"""
from functools import reduce
from typing import Any, Dict, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch import nn


class CircularPad(nn.Module):
    """Circular padding layer"""

    def __init__(self):
        super(CircularPad, self).__init__()

    def forward(self, x) -> torch.Tensor:
        x = torch.cat([torch.narrow(x, 2, 3, 2), x, torch.narrow(x, 2, 0, 2)], dim=2)
        x = torch.cat([torch.narrow(x, 3, 48, 16), x, torch.narrow(x, 3, 0, 16)], dim=3)
        return x


class RaulNetV9(torch.nn.Module):
    """Model definition used in Sîmpetru et al. [1]_

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

    Notes
    -----
    .. [1] Sîmpetru, R.C., März, M., Del Vecchio, A., 2023. Proportional and simultaneous real-time control of the full human hand from high-density electromyography. IEEE Trans. Neural Syst. Rehabil. Eng. 31, 3118–3131. https://doi.org/10/gsgk4s

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
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
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
                groups=self.nr_of_input_channels,
            ),
            nn.GELU(),
            nn.InstanceNorm3d(self.cnn_encoder_channels[0], track_running_stats=False),
            nn.Dropout3d(p=0.25),
            CircularPad(),
            nn.Conv3d(
                self.cnn_encoder_channels[0],
                self.cnn_encoder_channels[1],
                kernel_size=(5, 32, 18),
                dilation=(1, 2, 1),
                padding_mode="circular",
                padding=(2, 16, 0),
            ),
            nn.GELU(),
            nn.InstanceNorm3d(self.cnn_encoder_channels[1], track_running_stats=False),
            nn.Conv3d(
                self.cnn_encoder_channels[1],
                self.cnn_encoder_channels[2],
                kernel_size=(5, 9, 1),
            ),
            nn.GELU(),
            nn.InstanceNorm3d(self.cnn_encoder_channels[2], track_running_stats=False),
            nn.Flatten(),
        )

        self.mlp_encoder = nn.Sequential(
            nn.Linear(
                reduce(
                    lambda x, y: x * int(y),
                    self.cnn_encoder(
                        torch.rand(
                            (
                                1,
                                self.nr_of_input_channels,
                                5,
                                64,
                                self.input_length__samples,
                            )
                        )
                    ).shape[1:],
                    1,
                ),
                self.mlp_encoder_channels[0],
            ),
            nn.GELU(),
            nn.Linear(self.mlp_encoder_channels[0], self.mlp_encoder_channels[1]),
            nn.GELU(),
            nn.Linear(self.mlp_encoder_channels[1], self.nr_of_outputs),
        )

    def forward(self, inputs) -> torch.Tensor:
        x = self._reshape_and_normalize(inputs)

        x = self.cnn_encoder(x)
        x = self.mlp_encoder(x)

        return x

    @staticmethod
    def _reshape_and_normalize(inputs):
        x = torch.stack(inputs.split(64, dim=2), dim=2)
        return (x - x.mean(dim=(3, 4), keepdim=True)) / (
            x.std(dim=(3, 4), keepdim=True, unbiased=True) + 1e-15
        )


class RaulNetV9_Compilable(pl.LightningModule):
    """This model is exactly the same as the RaulNetV9 model, but it is required for the old pytorch compile method.
    Some tricks were involved to make it work.
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
        super(RaulNetV9_Compilable, self).__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.nr_of_input_channels = nr_of_input_channels
        self.nr_of_outputs = nr_of_outputs
        self.input_length__samples = input_length__samples

        self.cnn_encoder_channels = cnn_encoder_channels
        self.mlp_encoder_channels = mlp_encoder_channels
        self.event_search_kernel_length = event_search_kernel_length
        self.event_search_kernel_stride = event_search_kernel_stride

        self.model = RaulNetV9(
            learning_rate=learning_rate,
            nr_of_input_channels=nr_of_input_channels,
            input_length__samples=input_length__samples,
            nr_of_outputs=nr_of_outputs,
            cnn_encoder_channels=cnn_encoder_channels,
            mlp_encoder_channels=mlp_encoder_channels,
            event_search_kernel_length=event_search_kernel_length,
            event_search_kernel_stride=event_search_kernel_stride,
        )

        self.criterion = self.model.criterion

        # backends
        # ['aot_ts_nvfuser', 'cudagraphs', 'inductor', 'ipex', 'nvprims_nvfuser', 'onnxrt', 'tensorrt', 'tvm']
        self.model = torch.compile(self.model, backend="inductor", fullgraph=False)

    def forward(self, inputs) -> torch.Tensor:
        return self.model(inputs)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=0.32
        )

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
            "name": "OneCycleLR",
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [lr_scheduler]

    def training_step(
        self, train_batch, batch_idx: int
    ) -> Optional[Union[torch.Tensor, Dict[str, Any]]]:
        inputs, ground_truths = train_batch

        prediction = self.forward(inputs)
        scores_dict = {"loss": self.criterion(prediction, ground_truths)}

        if scores_dict["loss"].isnan().item():
            return None

        self.log_dict(
            scores_dict, prog_bar=True, logger=False, on_epoch=True, sync_dist=True
        )
        self.log_dict(
            {f"train/{k}": v for k, v in scores_dict.items()},
            prog_bar=False,
            logger=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

        return scores_dict

    def validation_step(
        self, batch, batch_idx
    ) -> Optional[Union[torch.Tensor, Dict[str, Any]]]:
        inputs, ground_truths = batch

        prediction = self.forward(inputs)
        scores_dict = {"val_loss": self.criterion(prediction, ground_truths)}

        self.log_dict(
            scores_dict, prog_bar=True, logger=False, on_epoch=True, sync_dist=True
        )

        return scores_dict

    def test_step(
        self, batch, batch_idx
    ) -> Optional[Union[torch.Tensor, Dict[str, Any]]]:
        inputs, ground_truths = batch

        prediction = self.forward(inputs)
        scores_dict = {"loss": self.criterion(prediction, ground_truths)}

        self.log_dict(
            scores_dict, prog_bar=True, logger=False, on_epoch=True, sync_dist=True
        )
        self.log_dict(
            {f"test/{k}": v for k, v in scores_dict.items()},
            prog_bar=False,
            logger=True,
            on_epoch=False,
            on_step=True,
            sync_dist=True,
        )

        return scores_dict
