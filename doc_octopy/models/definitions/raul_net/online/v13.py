"""Model definition not used in any publication"""
import pickle
from pathlib import Path

import torch
from torch import nn

from doc_octopy.models.definitions.raul_net.online.v9_5_grids_compilable import RaulNetV9_Compilable


class RaulNetV13(nn.Module):
    """Wrapper to make one RaulNetV9_Compilable per finger

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

    def __init__(self, models_path: Path):
        super(RaulNetV13, self).__init__()

        self.cnns = nn.ModuleList()
        self.mlps = nn.ModuleList()

        model_paths = {}
        for name_file_path in list(models_path.rglob("mlflow.runName")):
            model_paths[name_file_path.open("r").read().split(" ")[-1]] = name_file_path

        for name_file_path in [
            model_paths["thumb"],
            model_paths["index"],
            model_paths["middle"],
            model_paths["ring"],
            model_paths["pinky"],
        ]:
            temp = RaulNetV9_Compilable(
                **pickle.load((Path(name_file_path).parent.parent / "artifacts" / "model_hparams.pkl").open("rb"))
            )
            temp.load_state_dict(
                {
                    k.replace("model._orig_mod.", ""): v
                    for k, v in torch.load(name_file_path.parent.parent / "artifacts" / "model.pt").items()
                }
            )

            self.cnns.append(temp.cnn_encoder)
            self.mlps.append(temp.mlp_encoder)

    def forward(self, inputs) -> torch.Tensor:
        input_tensor = self._reshape_and_normalize(inputs)
        return torch.cat([mlp(cnn(input_tensor)) for cnn, mlp in zip(self.cnns, self.mlps)], dim=1)

    def _reshape_and_normalize(self, inputs):
        x = torch.stack(inputs.split(64, dim=2), dim=2)
        return (x - x.mean(dim=(3, 4), keepdim=True)) / (x.std(dim=(3, 4), keepdim=True, unbiased=True) + 1e-15)
