from typing import Dict, List

import numpy.typing as npt
import torch
from torch import nn

from base_model import BaseModel


def conv_layer(
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
        padding=dilation,
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
        super().__init__()
        self.num_histones = num_histones
        self.seq_len = seq_len

        # original shape: (batch_size, num_histones, seq_len)

        self.cnn = nn.Sequential(
            conv_layer(num_histones, 16, dilation=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # shape is now (batch_size, 32, seq_len // 2)
            conv_layer(16, 32, dilation=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # shape is now (batch_size, 32, seq_len // 4)
            nn.Flatten(),
        )

        with torch.no_grad():
            n_flatten = self.cnn(
                torch.zeros(1, num_histones, seq_len, dtype=torch.float32)
            ).shape[1]

        self.fc = mlp(
            in_features=n_flatten,
            out_features=1,
            layer_sizes=[256, 64],
            out_act=nn.Identity(),  # TODO: maybe RELU or Sigmoid?
        )

        print(self.cnn)
        print(self.fc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        return self.fc(x)


class CNNModel(BaseModel):
    def __init__(self, run_dir: Dict[str, str]) -> None:
        super().__init__(run_dir)

        self.num_histones = len(self.cfg.histones)
        self.seq_len = self.cfg.seq_len
        self.model = CNNArchitecture(
            num_histones=self.num_histones, seq_len=self.seq_len
        ).to(self.cfg.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.cfg.learning_rate
        )

    def fit(
        self,
    ) -> None:
        self.torch_train_pipeline(
            model=self.model,
            optimizer=self.optimizer,
            criterion=nn.MSELoss(),
        )

    def predict(self, X: npt.NDArray) -> npt.NDArray:
        assert len(X.shape) == 3, "Unexpected number of dimensions"
        assert X.shape[1] == self.num_histones, "Unexpected number of histones"
        assert X.shape[2] == self.seq_len, "Unexpected sequence length"

        X_torch = torch.tensor(X, dtype=torch.float32).to(self.cfg.device)

        with torch.no_grad():
            return self.model(X_torch).detach().cpu().numpy()

    def save_weights(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load_weights(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path))
