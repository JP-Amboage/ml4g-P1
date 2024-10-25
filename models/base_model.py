from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader, TensorDataset

from utils import Config, CustomLogger


class BaseModel(ABC):
    def __init__(self) -> None:
        self.cfg = Config()
        self.cfg.parse("hyperparams")
        self.logger = CustomLogger()

    def load_data(self) -> None:
        # base path is self.cfg.data_path
        # then we want X, and y numpy arrays, that are the result of concatenating:
        # {X1. X2}_{train, val}_{X, y}.npy
        # dont mind about the split and just merge everything into big X and y numpy arrays

        X1_train_X = np.load(f"{self.cfg.data_path}/X1_train_X.npy")
        X1_train_y = np.load(f"{self.cfg.data_path}/X1_train_y.npy")
        X1_val_X = np.load(f"{self.cfg.data_path}/X1_val_X.npy")
        X1_val_y = np.load(f"{self.cfg.data_path}/X1_val_y.npy")
        X2_train_X = np.load(f"{self.cfg.data_path}/X2_train_X.npy")
        X2_train_y = np.load(f"{self.cfg.data_path}/X2_train_y.npy")
        X2_val_X = np.load(f"{self.cfg.data_path}/X2_val_X.npy")
        X2_val_y = np.load(f"{self.cfg.data_path}/X2_val_y.npy")

        self.X_train = np.concatenate((X1_train_X, X2_train_X), axis=1)
        self.y_train = np.concatenate((X1_train_y, X2_train_y), axis=1)
        self.X_val = np.concatenate((X1_val_X, X2_val_X), axis=1)
        self.y_val = np.concatenate((X1_val_y, X2_val_y), axis=1)

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
