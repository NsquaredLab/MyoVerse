"""Model definition not used in any publication"""
from typing import Any, Dict, Optional, Sequence, Union

import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch import nn


class Transpose(nn.Module):
    """Circular padding layer"""

    def __init__(self, dim0: int, dim1: int):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x) -> torch.Tensor:
        return torch.transpose(x, dim0=self.dim0, dim1=self.dim1).contiguous(memory_format=torch.contiguous_format)


class RaulNetV5(pl.LightningModule):
    """Model definition not used in any publication

    Attributes
    ----------
    example_input_array : torch.Tensor
        Used for creating a summery and checking if the model architecture is valid.
    learning_rate : float
        The learning rate.
    """

    def __init__(
        self,
        learning_rate: float,
        cnn_layer_sizes: Sequence[int],
        mlp_layer_sizes: Sequence[int],
        nr_of_kernels_for_extraction: int,
        nr_of_outputs: int,
    ):
        super(RaulNetV5, self).__init__()

        self.save_hyperparameters()

        self.learning_rate = learning_rate

        self.cnn_layer_sizes = cnn_layer_sizes
        self.mlp_layer_sizes = mlp_layer_sizes
        self.nr_of_kernels_for_extraction = nr_of_kernels_for_extraction

        self.nr_of_outputs = nr_of_outputs

        self.criterion = nn.L1Loss()

        self.raw_ap_extractor = nn.Conv2d(
            5, 5 * self.nr_of_kernels_for_extraction, kernel_size=(1, 61), stride=(1, 8), bias=False, groups=5
        )
        self.filtered_ap_extractor = nn.Conv2d(
            5, 5 * self.nr_of_kernels_for_extraction, kernel_size=(1, 61), stride=(1, 8), bias=False, groups=5
        )

        # --------------------------------------------------------------------------------------------------------------
        self.encoder = nn.Sequential(
            nn.GELU(),
            nn.BatchNorm3d(2),
            nn.Conv3d(
                2,
                self.cnn_layer_sizes[0],
                kernel_size=(self.nr_of_kernels_for_extraction // 4, 16, 5),
                padding=(self.nr_of_kernels_for_extraction // 4 // 2, 8, 0),
                dilation=(1, 2, 1),
                stride=(self.nr_of_kernels_for_extraction // 4 // 2, 2, 1),
                padding_mode="circular",
                bias=False,
            ),
            nn.GELU(),
            nn.BatchNorm3d(self.cnn_layer_sizes[0]),
            nn.Dropout3d(p=0.25),
            nn.Conv3d(
                self.cnn_layer_sizes[0], self.cnn_layer_sizes[1], kernel_size=(7, 9, 5), dilation=(2, 1, 1), bias=False
            ),
            nn.GELU(),
            nn.BatchNorm3d(self.cnn_layer_sizes[1]),
            nn.Conv3d(
                self.cnn_layer_sizes[1], self.cnn_layer_sizes[2], kernel_size=(7, 9, 5), dilation=(2, 1, 1), bias=False
            ),
            nn.GELU(),
            nn.BatchNorm3d(self.cnn_layer_sizes[2]),
            nn.Conv3d(self.cnn_layer_sizes[2], self.cnn_layer_sizes[3], kernel_size=(17, 9, 5)),
            nn.GELU(),
            nn.BatchNorm3d(self.cnn_layer_sizes[3]),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(self.cnn_layer_sizes[3], self.mlp_layer_sizes[0]),
            nn.GELU(),
            nn.BatchNorm1d(self.mlp_layer_sizes[0]),
            nn.Linear(self.mlp_layer_sizes[0], self.mlp_layer_sizes[1]),
            nn.GELU(),
            nn.BatchNorm1d(self.mlp_layer_sizes[1]),
            nn.Linear(self.mlp_layer_sizes[1], self.nr_of_outputs),
            nn.Tanh(),
        )

    def forward(self, inputs) -> torch.Tensor:
        x = torch.stack(torch.split(inputs, 64, dim=2), dim=1).contiguous(memory_format=torch.contiguous_format)
        x = self._normalize_input(x)

        x = torch.stack([self.raw_ap_extractor(x[:, :, 0]), self.filtered_ap_extractor(x[:, :, 1])], dim=1)

        return self.encoder(x)

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
                final_div_factor=1e2,
            ),
            "name": "OncCycleLR",
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [lr_scheduler]

    def training_step(self, train_batch, batch_idx: int) -> Optional[Union[torch.Tensor, Dict[str, Any]]]:
        inputs, ground_truths = train_batch

        scores_dict = {"loss": self.criterion(self(inputs), ground_truths)}

        self.log_dict(scores_dict, prog_bar=True, logger=False, on_epoch=True)
        self.log_dict(
            {f"train/{k}": v for k, v in scores_dict.items()}, prog_bar=False, logger=True, on_epoch=True, on_step=False
        )

        return scores_dict

    def validation_step(self, batch, batch_idx) -> Optional[Union[torch.Tensor, Dict[str, Any]]]:
        inputs, ground_truths = batch

        scores_dict = {"val_loss": self.criterion(self(inputs), ground_truths)}

        self.log_dict(scores_dict, prog_bar=True, logger=False, on_epoch=True)

        return scores_dict

    def test_step(self, batch, batch_idx) -> Optional[Union[torch.Tensor, Dict[str, Any]]]:
        inputs, ground_truths = batch

        scores_dict = {"loss": self.criterion(self(inputs), ground_truths)}

        self.log_dict(scores_dict, prog_bar=True, logger=False, on_epoch=True)
        self.log_dict(
            {f"test/{k}": v for k, v in scores_dict.items()}, prog_bar=False, logger=True, on_epoch=False, on_step=True
        )

        return scores_dict
