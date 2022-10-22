from typing import List

from torch import nn
import torch.nn.functional as F


class MLPNet(nn.Module):
    def __init__(
        self,
        dropout: float,
        input_size: int,
        output_size: int,
        output_dims: List[int]
    ):
        super().__init__()

        layers: List[nn.Module] = []

        input_dim = input_size
        for output_dim in output_dims:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = output_dim

        layers.append(nn.Linear(input_dim, output_size))

        self.layers: nn.Module = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.layers(x)

        return logits