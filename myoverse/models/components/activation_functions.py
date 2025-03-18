import torch
from torch import nn


class PSerf(nn.Module):
    """PSerf activation function from Biswas et al.

    Parameters
    ----------
    gamma : float, optional
        The gamma parameter, by default 1.0.
    sigma : float, optional
        The sigma parameter, by default 1.25.
    stabilisation_term : float, optional
        The stabilisation term, by default 1e-12.

    References
    ----------
    Biswas, K., Kumar, S., Banerjee, S., Pandey, A.K., 2021.
    ErfAct and PSerf: Non-monotonic smooth trainable Activation Functions. arXiv:2109.04386 [cs].
    """

    def __init__(
        self, gamma: float = 1.0, sigma: float = 1.25, stabilisation_term: float = 1e-12
    ):
        super(PSerf, self).__init__()

        self.gamma = nn.Parameter(torch.tensor(gamma), requires_grad=True)
        self.sigma = nn.Parameter(torch.tensor(sigma), requires_grad=True)

        self.stabilisation_term = torch.tensor(stabilisation_term)

    def forward(self, x) -> torch.Tensor:
        return (
            x * torch.erf(self.gamma * torch.log(1 + torch.exp(self.sigma * x)))
            + self.stabilisation_term
        )


class SAU(nn.Module):
    """SAU activation function from Biswas et al.

    Parameters
    ----------
    alpha : float, optional
        The alpha parameter, by default 0.15.
    n : int, optional
        The n parameter, by default 20000.

    References
    ----------
    Biswas, K., Kumar, S., Banerjee, S., Pandey, A.K., 2021.
    SAU: Smooth activation function using convolution with approximate identities. arXiv:2109.13210 [cs].

    """

    def __init__(self, alpha=0.15, n=20000):
        super(SAU, self).__init__()

        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=True)
        self.n = torch.tensor(n)

    def forward(self, x) -> torch.Tensor:
        return (
            torch.sqrt(torch.tensor(2 / torch.pi))
            * torch.exp(-(torch.pow(self.n, 2) * torch.pow(x, 2)) / 2)
            / (2 * self.n)
            + (1 + self.alpha) / 2 * x
            + (1 - self.alpha)
            / 2
            * x
            * torch.erf(self.n * x / torch.sqrt(torch.tensor(2)))
        )


class SMU(nn.Module):
    """SMU activation function from Biswas et al.

    Parameters
    ----------
    alpha : float, optional
        The alpha parameter, by default 0.01.
    mu : float, optional
        The mu parameter, by default 2.5.

    References
    ----------
    Biswas, K., Kumar, S., Banerjee, S., Pandey, A.K., 2022.
    SMU: smooth activation function for deep networks using smoothing maximum technique. arXiv:2111.04682 [cs].

    Notes
    -----
    This version also make alpha trainable.
    """

    def __init__(self, alpha=0.01, mu=2.5):
        super(SMU, self).__init__()

        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=True)
        self.mu = nn.Parameter(torch.tensor(mu), requires_grad=True)

    def forward(self, x) -> torch.Tensor:
        return (
            (1 + self.alpha) * x
            + (1 - self.alpha) * x * torch.erf(self.mu * (1 - self.alpha) * x)
        ) / 2


class SMU_old(nn.Module):
    """SMU activation function from Biswas et al. This is an older version of the SMU activation function and should not
    be used.

    Warning
    -------
    This is an older version of the SMU activation function and should not be used.

    Parameters
    ----------
    alpha : float, optional
        The alpha parameter, by default 0.01.
    mu : float, optional
    The mu parameter, by default 2.5.

    References
    ----------
    Biswas, K., Kumar, S., Banerjee, S., Pandey, A.K., 2022.
    SMU: smooth activation function for deep networks using smoothing maximum technique. arXiv:2111.04682 [cs].
    """

    def __init__(self, alpha=0.01, mu=2.5):
        super(SMU_old, self).__init__()

        self.alpha = torch.tensor(alpha)
        self.mu = nn.Parameter(torch.tensor(mu), requires_grad=True)

    def forward(self, x) -> torch.Tensor:
        return (
            (1 + self.alpha) * x
            + (1 - self.alpha) * x * torch.erf(self.mu * (1 - self.alpha) * x)
        ) / 2
