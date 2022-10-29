from typing import List, Tuple

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
        # print(x.shape)
        x = x.squeeze()
        logits = self.layers(x)

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
        global_pooling: str = "mean"
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

        self.dense_layers = MLPNet(
            dropout=dropout,
            input_size=mlp_input,
            output_size=output_size,
            output_dims=mlp_output_dims
        )

    def forward(self, x):
        # print(x.shape)
        for conv in self.conv_layers:
            x = conv(x)

        if self.global_pooling == "mean":
            x = x.mean(-1)
        elif self.global_pooling == "flat":
            x = x.flatten(start_dim=1)
        else:
            print("Pooling is not availabel... exiting...")
            exit()
        # print(x.shape)
        logits = self.dense_layers(x)

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
                nn.MaxPool2d(2, 2),
                nn.Dropout(dropout)
        )

        self.conv_layers = nn.ModuleList()

        in_d = in_channels
        for dim, k, stride in conv_layers:
            self.conv_layers.append(block(in_d, dim, k, stride))
            in_d = dim

        self.dense_layers = MLPNet(
            dropout=dropout,
            input_size=mlp_input,
            output_size=output_size,
            output_dims=mlp_output_dims
        )

    def forward(self, x):
        # print(x.shape)
        for conv in self.conv_layers:
            x = conv(x)

        if self.global_pooling == "mean":
            x = x.mean(-1)
        if self.global_pooling == "flat":
            x = x.flatten(start_dim=1)
        else:
            print("Pooling is not availabel... exiting...")
            exit()

        # print(x.shape)

        logits = self.dense_layers(x)

        return logits