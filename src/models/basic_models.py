from typing import List, Tuple

import torch
from torch import nn


class MLPBase(nn.Module):
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

        self.classification_layers: nn.Module = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.classification_layers(x)

        return logits


class MLPNet(nn.Module):
    def __init__(
        self,
        dropout: float,
        input_size: int,
        output_size: int,
        output_dims: List[int]
    ):
        super().__init__()

        self.classification_layers = MLPBase(
            dropout=dropout,
            input_size=input_size,
            output_size=output_size,
            output_dims=output_dims
        )

    def forward(self, x):
        # features = torch.mean(x, dim=1)
        features = x
        logits = self.classification_layers(features)

        return logits


class CNN1DNet(nn.Module):
    def __init__(
        self,
        dropout: float,
        in_channels: int,
        mlp_input: int,
        mlp_output_dims: List[int],
        output_size: int,
        conv_layers: List[Tuple],
        global_pooling: str = "mean",
    ):
        super().__init__()


        self.global_pooling = global_pooling

        def block(n_in, n_out, k, stride):
            return nn.Sequential(
                nn.Conv1d(n_in, n_out, k, stride=stride, bias=False),
                nn.BatchNorm1d(n_out),
                nn.ReLU(),
                nn.MaxPool1d(2, 2),
                nn.Dropout(dropout)
        )

        self.conv_layers = nn.ModuleList()

        in_d = in_channels
        for dim, k, stride in conv_layers:
            self.conv_layers.append(block(in_d, dim, k, stride))
            in_d = dim

        self.classification_layers = MLPBase(
            dropout=dropout,
            input_size=mlp_input,
            output_size=output_size,
            output_dims=mlp_output_dims
        )

    def forward(self, x):
        # x = torch.mean(x, dim=1).unsqueeze(1)
        x = x.unsqueeze(1)
        for conv in self.conv_layers:
            x = conv(x)

        if self.global_pooling == "mean":
            x = x.mean(-1)
        elif self.global_pooling == "flat":
            x = x.flatten(start_dim=1)
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these: ['mean', 'flat']"
            )

        logits = self.classification_layers(x)

        return logits


class CNN2DNet(nn.Module):
    def __init__(
        self,
        dropout: float,
        in_channels: int,
        mlp_input: int,
        mlp_output_dims: List[int],
        output_size: int,
        conv_layers: List[Tuple],
        global_pooling: str = "mean"
    ):
        super().__init__()

        self.global_pooling = global_pooling

        def block(n_in, n_out, k, stride):
            return nn.Sequential(
                nn.Conv2d(n_in, n_out, k, stride=stride, bias=False),
                nn.BatchNorm2d(n_out),
                nn.ReLU(),
                nn.AvgPool2d(2, 2),
                nn.Dropout(dropout)
        )

        self.conv_layers = nn.ModuleList()

        in_d = in_channels
        for dim, k, stride in conv_layers:
            self.conv_layers.append(block(in_d, dim, k, stride))
            in_d = dim

        self.classification_layers = MLPBase(
            dropout=dropout,
            input_size=mlp_input,
            output_size=output_size,
            output_dims=mlp_output_dims
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        for conv in self.conv_layers:
            x = conv(x)

        if self.global_pooling == "mean":
            x = torch.mean(x, dim=3)
        elif self.global_pooling == "flat":
            x = x.flatten(start_dim=1)
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these: ['mean', 'flat']"
            )

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        logits = self.classification_layers(x)

        return logits