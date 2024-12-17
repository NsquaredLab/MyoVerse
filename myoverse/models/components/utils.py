import torch
from torch import nn


class WeightedSum(nn.Module):
    def __init__(self, alpha=0.5):
        super(WeightedSum, self).__init__()

        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=True)

    def forward(self, x, y) -> torch.Tensor:
        return self.alpha * x + (1 - self.alpha) * y


class CircularPad(nn.Module):
    """Circular padding layer used in the paper [1].

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        Padded tensor.

    References
    ----------
    [1] Sîmpetru, R.C., Osswald, M., Braun, D.I., Oliveira, D.S., Cakici, A.L., Del Vecchio, A., 2022. Accurate Continuous
    Prediction of 14 Degrees of Freedom of the Hand from Myoelectrical Signals through Convolutive Deep Learning, in:
    2022 44th Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC),
    pp. 702–706. https://doi.org/10.1109/EMBC48229.2022.9870937
    """

    def __init__(self):
        super(CircularPad, self).__init__()

    def forward(self, x) -> torch.Tensor:
        x = torch.cat([torch.narrow(x, 2, 3, 2), x, torch.narrow(x, 2, 0, 2)], dim=2)
        x = torch.cat([torch.narrow(x, 3, 48, 16), x, torch.narrow(x, 3, 0, 16)], dim=3)
        return x
