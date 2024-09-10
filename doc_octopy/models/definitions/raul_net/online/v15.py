"""Model definition not used in any publication"""
import sys
from functools import reduce
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    from bayesian_torch.layers import get_kernel_size
except ImportError:
    sys.exit(
        """You need bayesian_torch!
                install it from https://github.com/IntelLabs/bayesian-torch
                or run pip install bayesian-torch."""
    )


class BaseVariationalLayer_(nn.Module):
    def __init__(self):
        super().__init__()

    def kl_div(self, mu_q, sigma_q, mu_p, sigma_p):
        """
        Calculates kl divergence between two gaussians (Q || P)

        Parameters:
             * mu_q: torch.Tensor -> mu parameter of distribution Q
             * sigma_q: torch.Tensor -> sigma parameter of distribution Q
             * mu_p: float -> mu parameter of distribution P
             * sigma_p: float -> sigma parameter of distribution P

        returns torch.Tensor of shape 0
        """
        kl = (
            torch.log(sigma_p)
            - torch.log(sigma_q)
            + (sigma_q**2 + (mu_q - mu_p) ** 2) / (2 * (sigma_p**2))
            - 0.5
        )
        return kl.mean()


class Conv3dFlipout(BaseVariationalLayer_):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        prior_mean=0,
        prior_variance=1,
        posterior_mu_init=0,
        posterior_rho_init=-3.0,
    ):
        """
        Implements Conv3d layer with Flipout reparameterization trick.

        Inherits from bayesian_torch.layers.BaseVariationalLayer_

        Parameters:
            in_channels: int -> number of channels in the input image,
            out_channels: int -> number of channels produced by the convolution,
            kernel_size: int -> size of the convolving filter,
            stride: int -> stride of the convolution. Default: 1,
            padding: int -> zero-padding added to both sides of the input. Default: 0,
            dilation: int -> spacing between filter elements. Default: 1,
            groups: int -> number of blocked connections from input channels to output channels,
            prior_mean: float -> mean of the prior arbitrary distribution to be used on the
                                 complexity cost,
            prior_variance: float -> variance of the prior arbitrary distribution to be used on
                                     the complexity cost,
            posterior_mu_init: float -> init trainable mu parameter representing mean
                                        of the approximate posterior,
            posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate
                                         posterior through softplus function,
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init

        kernel_size = get_kernel_size(kernel_size, 3)

        self.mu_kernel = nn.Parameter(
            torch.Tensor(
                out_channels,
                in_channels // groups,
                kernel_size[0],
                kernel_size[1],
                kernel_size[2],
            )
        )
        self.rho_kernel = nn.Parameter(
            torch.Tensor(
                out_channels,
                in_channels // groups,
                kernel_size[0],
                kernel_size[1],
                kernel_size[2],
            )
        )

        self.register_buffer(
            "eps_kernel",
            torch.Tensor(
                out_channels,
                in_channels // groups,
                kernel_size[0],
                kernel_size[1],
                kernel_size[2],
            ),
            persistent=False,
        )
        self.register_buffer(
            "prior_weight_mu",
            torch.Tensor(
                out_channels,
                in_channels // groups,
                kernel_size[0],
                kernel_size[1],
                kernel_size[2],
            ),
            persistent=False,
        )
        self.register_buffer(
            "prior_weight_sigma",
            torch.Tensor(
                out_channels,
                in_channels // groups,
                kernel_size[0],
                kernel_size[1],
                kernel_size[2],
            ),
            persistent=False,
        )

        self.mu_bias = nn.Parameter(torch.Tensor(out_channels))
        self.rho_bias = nn.Parameter(torch.Tensor(out_channels))
        self.register_buffer("eps_bias", torch.Tensor(out_channels), persistent=False)
        self.register_buffer(
            "prior_bias_mu", torch.Tensor(out_channels), persistent=False
        )
        self.register_buffer(
            "prior_bias_sigma", torch.Tensor(out_channels), persistent=False
        )

        self.init_parameters()

    def init_parameters(self):
        # prior values
        self.prior_weight_mu.data.fill_(self.prior_mean)
        self.prior_weight_sigma.data.fill_(self.prior_variance)

        # init our weights for the deterministic and perturbated weights
        self.mu_kernel.data.normal_(mean=self.posterior_mu_init, std=0.1)
        self.rho_kernel.data.normal_(mean=self.posterior_rho_init, std=0.1)

        self.mu_bias.data.normal_(mean=self.posterior_mu_init, std=0.1)
        self.rho_bias.data.normal_(mean=self.posterior_rho_init, std=0.1)
        self.prior_bias_mu.data.fill_(self.prior_mean)
        self.prior_bias_sigma.data.fill_(self.prior_variance)

    def kl_loss(self):
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        kl = self.kl_div(
            self.mu_kernel, sigma_weight, self.prior_weight_mu, self.prior_weight_sigma
        )

        sigma_bias = torch.log1p(torch.exp(self.rho_bias))
        kl += self.kl_div(
            self.mu_bias, sigma_bias, self.prior_bias_mu, self.prior_bias_sigma
        )
        return kl

    def forward(self, x):
        # linear outputs
        outputs = F.conv3d(
            x,
            weight=self.mu_kernel,
            bias=self.mu_bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        # sampling perturbation signs
        sign_input = torch.rand_like(x, device=x.device) * 2 - 1
        sign_input = sign_input.sign()

        sign_output = torch.rand_like(outputs, device=outputs.device) * 2 - 1
        sign_output = sign_output.sign()

        # gettin perturbation weights
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        eps_kernel = self.eps_kernel.data.normal_()

        delta_kernel = sigma_weight * eps_kernel

        kl = self.kl_div(
            self.mu_kernel, sigma_weight, self.prior_weight_mu, self.prior_weight_sigma
        )

        sigma_bias = torch.log1p(torch.exp(self.rho_bias))
        eps_bias = self.eps_bias.data.normal_()
        bias = sigma_bias * eps_bias
        kl = kl + self.kl_div(
            self.mu_bias, sigma_bias, self.prior_bias_mu, self.prior_bias_sigma
        )

        # perturbed feedforward
        perturbed_outputs = (
            F.conv3d(
                x * sign_input,
                weight=delta_kernel,
                bias=bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
            * sign_output
        )
        # returning outputs + perturbations

        return outputs + perturbed_outputs, kl


class RaulNetV15(pl.LightningModule):
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
        nr_of_electrode_grids: int = 5,
        nr_of_electrodes_per_grid: int = 64,
        inference_only: bool = False,
    ):
        super(RaulNetV15, self).__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.nr_of_input_channels = nr_of_input_channels
        self.nr_of_outputs = nr_of_outputs
        self.input_length__samples = input_length__samples

        self.cnn_encoder_channels = cnn_encoder_channels
        self.mlp_encoder_channels = mlp_encoder_channels
        self.event_search_kernel_length = event_search_kernel_length
        self.event_search_kernel_stride = event_search_kernel_stride

        self.nr_of_electrode_grids = nr_of_electrode_grids
        self.nr_of_electrodes_per_grid = nr_of_electrodes_per_grid

        self.inference_only = inference_only

        self.criterion = nn.L1Loss()

        self.bayesian_conv3d = Conv3dFlipout(
            self.nr_of_input_channels,
            self.cnn_encoder_channels[0],
            kernel_size=(1, 1, self.event_search_kernel_length),
            stride=(1, 1, self.event_search_kernel_stride),
            groups=self.nr_of_input_channels,
        )

        self._dummy_conv3d = nn.Conv3d(
            self.nr_of_input_channels,
            self.cnn_encoder_channels[0],
            kernel_size=(1, 1, self.event_search_kernel_length),
            stride=(1, 1, self.event_search_kernel_stride),
            groups=self.nr_of_input_channels,
        )

        self.cnn_encoder = nn.Sequential(
            nn.GELU(approximate="tanh"),
            nn.InstanceNorm3d(self.cnn_encoder_channels[0]),
            nn.Dropout3d(p=0.20),
            nn.Conv3d(
                self.cnn_encoder_channels[0],
                self.cnn_encoder_channels[1],
                kernel_size=(
                    self.nr_of_electrode_grids,
                    int(np.floor(self.nr_of_electrodes_per_grid / 2)),
                    18,
                ),
                dilation=(1, 2, 1),
                padding=(
                    int(np.floor(self.nr_of_electrode_grids / 2)),
                    int(np.floor(self.nr_of_electrodes_per_grid / 4)),
                    0,
                ),
                padding_mode="circular",
            ),
            nn.GELU(approximate="tanh"),
            nn.InstanceNorm3d(self.cnn_encoder_channels[1]),
            nn.Conv3d(
                self.cnn_encoder_channels[1],
                self.cnn_encoder_channels[2],
                kernel_size=(
                    self.nr_of_electrode_grids,
                    int(np.floor(self.nr_of_electrodes_per_grid / 7)),
                    1,
                ),
            ),
            nn.GELU(approximate="tanh"),
            nn.InstanceNorm3d(self.cnn_encoder_channels[2]),
            nn.Flatten(),
        )

        self.mlp = nn.Sequential(
            nn.Linear(
                self.cnn_encoder(
                    self._dummy_conv3d(
                        torch.rand(
                            (
                                1,
                                self.nr_of_input_channels,
                                self.nr_of_electrode_grids,
                                self.nr_of_electrodes_per_grid,
                                self.input_length__samples,
                            )
                        )
                    )
                )
                .detach()
                .shape[1],
                self.mlp_encoder_channels[0],
            ),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.mlp_encoder_channels[0], self.mlp_encoder_channels[1]),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.mlp_encoder_channels[1], self.nr_of_outputs),
        )

        del self._dummy_conv3d

    def forward(self, inputs) -> Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        x = self._reshape_and_normalize(inputs)

        x, kl = self.bayesian_conv3d(x)
        x = self.cnn_encoder(x)
        x = self.mlp(x)

        if self.inference_only:
            return x

        return x, kl

    def _reshape_and_normalize(self, inputs):
        x = torch.stack(inputs.split(self.nr_of_electrodes_per_grid, dim=2), dim=2)
        return (x - x.mean(dim=(3, 4), keepdim=True)) / (
            x.std(dim=(3, 4), keepdim=True, unbiased=True) + 1e-15
        )

    def _reshape_and_normalize_v2(self, inputs):
        x = torch.stack(inputs.split(self.nr_of_electrodes_per_grid, dim=2), dim=2)
        return (x - x.median(dim=4, keepdim=True)[0].median(dim=3, keepdim=True)[0]) / (
            torch.quantile(
                torch.quantile(x, q=0.75, dim=4, keepdim=True),
                q=0.75,
                dim=3,
                keepdim=True,
            )
            - torch.quantile(
                torch.quantile(x, q=0.25, dim=4, keepdim=True),
                q=0.25,
                dim=3,
                keepdim=True,
            )
            + 1e-15
        )

    def _reshape_and_normalize_v3(self, inputs):
        x = torch.stack(inputs.split(self.nr_of_electrodes_per_grid, dim=2), dim=2)

        positive_mask = x >= 0
        negative_mask = x < 0

        positive_data = torch.log1p(x[positive_mask] + 1e-15)
        negative_data = -torch.log1p(-x[negative_mask] + 1e-15)

        x[positive_mask] = positive_data
        x[negative_mask] = negative_data

        return (x - x.mean(dim=(3, 4), keepdim=True)) / (
            x.std(dim=(3, 4), keepdim=True, unbiased=True) + 1e-15
        )

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
        ground_truths = ground_truths[:, 0]

        predictions, kls = [], []
        for i in range(7):
            prediction, kl = self(inputs)
            predictions.append(prediction)
            kls.append(kl)

        prediction = torch.stack(predictions, dim=0).mean(dim=0)
        kl = torch.stack(kls, dim=0).mean(dim=0)

        scores_dict = {
            "loss": self.criterion(prediction, ground_truths)
            + (kl / prediction.shape[0]),
            "kl": kl,
        }

        if scores_dict["loss"].isnan().item():
            return None

        self.log_dict(
            scores_dict, prog_bar=True, logger=False, on_epoch=True, rank_zero_only=True
        )
        self.log_dict(
            {f"train/{k}": v for k, v in scores_dict.items()},
            prog_bar=False,
            logger=True,
            on_epoch=True,
            on_step=False,
            rank_zero_only=True,
        )

        return scores_dict

    def validation_step(
        self, batch, batch_idx
    ) -> Optional[Union[torch.Tensor, Dict[str, Any]]]:
        inputs, ground_truths = batch
        ground_truths = ground_truths[:, 0]

        prediction, kl = self(inputs)
        scores_dict = {
            "val_loss": self.criterion(prediction, ground_truths),
            "val_kl": kl,
        }

        self.log_dict(
            scores_dict, prog_bar=True, logger=False, on_epoch=True, sync_dist=True
        )

        return scores_dict

    def test_step(
        self, batch, batch_idx
    ) -> Optional[Union[torch.Tensor, Dict[str, Any]]]:
        inputs, ground_truths = batch
        ground_truths = ground_truths[:, 0]

        prediction, kl = self(inputs)
        scores_dict = {"loss": self.criterion(prediction, ground_truths), "kl": kl}

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
