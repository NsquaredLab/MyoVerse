from torch import nn
import torch


class EuclidianDistance(nn.Module):
    def __init__(self):
        super(EuclidianDistance, self).__init__()

    def forward(self, prediction, ground_truth) -> torch.Tensor:
        return torch.mean(
            torch.mean(
                torch.sqrt(
                    torch.sum(
                        torch.square(
                            prediction.reshape(-1, 20, 3)
                            - ground_truth.reshape(-1, 20, 3)
                        ),
                        dim=-1,
                    )
                ),
                dim=-1,
            )
        )
