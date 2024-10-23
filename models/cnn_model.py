from typing import List

import numpy.typing as npt
import torch
from base_model import BaseModel
from torch import nn


def conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    dilation: int = 1,
) -> nn.Conv1d:
    """
    Convolution Block

    Args:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        stride: The stride.
        dilation: The dilation.
    Returns:
        The convolutional layer.
    """
    return nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=1,
        dilation=dilation,
    )


def mlp(
    in_features: int,
    out_features: int,
    layer_sizes: List[int],
    out_act: nn.Module = nn.Identity(),
    activation: nn.Module = nn.ReLU(),
) -> nn.Sequential:
    """
    Multilayer perceptron.

    Args:
        in_features: The number of input features.
        out_features: The number of output features.
        layer_sizes: The sizes of the layers.
        out_act: The output activation function.
        activation: The activation function.
    Returns:
        The multilayer perceptron.
    """
    layers: List[nn.Module] = []
    prev_size = in_features
    for size in layer_sizes:
        layers.append(nn.Linear(prev_size, size))
        layers.append(activation)
        prev_size = size
    layers.append(nn.Linear(prev_size, out_features))
    layers.append(out_act)
    return nn.Sequential(*layers)


class CNNArchitecture(nn.Module):
    def __init__(self, num_histones: int, seq_len: int) -> None:
        self.num_histones = num_histones
        self.seq_len = seq_len

        self.cnn = nn.Sequential(
            conv_block(num_histones, 16, dilation=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            conv_block(16, 32, dilation=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten(),
        )

        with torch.no_grad():
            n_flatten = self.cnn(
                torch.zeros(1, num_histones, seq_len, dtype=torch.float32)
            ).shape[1]

        assert n_flatten == 32 * (
            seq_len // 4
        ), f"Expected {32 * (seq_len // 4)}"

        self.fc = mlp(
            in_features=32 * (seq_len // 4),
            out_features=1,
            layer_sizes=[64, 32],
            out_act=nn.Identity(),  # TODO: maybe RELU or Sigmoid?
        )

        print(self.cnn)
        print(self.fc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        return self.fc(x)


class CNNModel(BaseModel):
    def __init__(self) -> None:
        pass

    def fit(
        self,
        X_train: npt.NDArray,
        y_train: npt.NDArray,
        X_val: npt.NDArray,
        y_val: npt.NDArray,
    ) -> None:
        pass

    def predict(self, X: npt.NDArray) -> npt.NDArray:
        pass

    def save_weights(self, path: str) -> None:
        pass

    def load_weights(self, path: str) -> None:
        pass
