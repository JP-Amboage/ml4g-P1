from abc import ABC, abstractmethod
from typing import Tuple

import numpy.typing as npt
import torch
from torch.utils.data import DataLoader, TensorDataset

from utils import Config, CustomLogger


class BaseModel(ABC):
    def __init__(self) -> None:
        self.cfg = Config()
        self.cfg.parse("hyperparams")
        self.logger = CustomLogger()
        pass

    @abstractmethod
    def fit(
        self,
        X_train: npt.NDArray,
        y_train: npt.NDArray,
        X_val: npt.NDArray,
        y_val: npt.NDArray,
    ) -> None:
        pass

    def preprocess_data(
        self,
        X_train: npt.NDArray,
        y_train: npt.NDArray,
        X_val: npt.NDArray,
        y_val: npt.NDArray,
    ) -> Tuple[DataLoader, DataLoader]:
        # X_train has shape (n_samples, n_histones, seq_len), idem for X_val
        # y_train has shape (n_samples, 1), idem for y_val

        X_train_th = torch.tensor(X_train, dtype=torch.float32)
        y_train_th = torch.tensor(y_train, dtype=torch.float32)
        X_val_th = torch.tensor(X_val, dtype=torch.float32)
        y_val_th = torch.tensor(y_val, dtype=torch.float32)

        train_dataset = TensorDataset(X_train_th, y_train_th)
        val_dataset = TensorDataset(X_val_th, y_val_th)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=4,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=4,
        )

        return train_loader, val_loader

    @abstractmethod
    def predict(self, X: npt.NDArray) -> npt.NDArray:
        pass

    @abstractmethod
    def save_weights(self, path: str) -> None:
        pass

    @abstractmethod
    def load_weights(self, path: str) -> None:
        pass
