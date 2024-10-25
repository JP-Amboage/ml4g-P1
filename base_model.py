from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from utils import (
    AverageMeter,
    Config,
    CustomLogger,
    pearson_correlation,
    r_squared,
    spearman_correlation,
)


class BaseModel(ABC):
    def __init__(self, run_dir: Dict[str, str]) -> None:
        self.cfg = Config()
        self.cfg.parse("hyperparams")
        self.logger = CustomLogger()
        self.run_dir = run_dir

        # Tensorboard writer
        self.summary_writer = SummaryWriter(log_dir=self.run_dir["tensorboard"])

        # Load data
        self.load_data()

        # Preprocess data for Torch-based models
        self.train_loader, self.val_loader = self.preprocess_data(
            self.X_train, self.y_train, self.X_val, self.y_val
        )

    def load_data(self) -> None:
        # base path is self.cfg.data_path
        # then we want X, and y numpy arrays, that are the result of concatenating:
        # {X1. X2}_{train, val}_{X, y}.npy
        # dont mind about the split and just merge everything into big X and y numpy arrays
        self.logger.log.info("Loading data...")

        X1_train_X = np.load(f"{self.cfg.numpy_data_dir}/X1_train_X.npy")
        X1_train_y = np.load(f"{self.cfg.numpy_data_dir}/X1_train_y.npy")
        X1_val_X = np.load(f"{self.cfg.numpy_data_dir}/X1_val_X.npy")
        X1_val_y = np.load(f"{self.cfg.numpy_data_dir}/X1_val_y.npy")
        X2_train_X = np.load(f"{self.cfg.numpy_data_dir}/X2_train_X.npy")
        X2_train_y = np.load(f"{self.cfg.numpy_data_dir}/X2_train_y.npy")
        X2_val_X = np.load(f"{self.cfg.numpy_data_dir}/X2_val_X.npy")
        X2_val_y = np.load(f"{self.cfg.numpy_data_dir}/X2_val_y.npy")

        self.X_train = np.concatenate((X1_train_X, X2_train_X), axis=0)
        self.y_train = np.concatenate((X1_train_y, X2_train_y), axis=0)
        self.y_train = np.expand_dims(self.y_train, axis=1)
        self.X_val = np.concatenate((X1_val_X, X2_val_X), axis=0)
        self.y_val = np.concatenate((X1_val_y, X2_val_y), axis=0)
        self.y_val = np.expand_dims(self.y_val, axis=1)

        self.logger.log.info(f"X_train shape: {self.X_train.shape}")
        self.logger.log.info(f"y_train shape: {self.y_train.shape}")
        self.logger.log.info(f"X_val shape: {self.X_val.shape}")
        self.logger.log.info(f"y_val shape: {self.y_val.shape}")

    def preprocess_data(
        self,
        X_train: npt.NDArray,
        y_train: npt.NDArray,
        X_val: npt.NDArray,
        y_val: npt.NDArray,
    ) -> Tuple[DataLoader, DataLoader]:
        # X_train has shape (n_samples, n_histones, seq_len), idem for X_val
        # y_train has shape (n_samples, 1), idem for y_val

        self.logger.log.info(
            "Preprocessing data. Converting Numpy arrays to PyTorch DataLoader..."
        )

        # Remove samples having any NaN value
        original_n_samples = X_train.shape[0]
        nan_samples = np.isnan(X_train).any(axis=(1, 2))
        X_train = X_train[~nan_samples]
        y_train = y_train[~nan_samples]
        self.logger.log.info(
            f"Removed {original_n_samples - X_train.shape[0]} samples "
            f"out of {original_n_samples} due to NaN values"
        )

        # Normalize data
        mean = np.mean(X_train, axis=(0, 2), keepdims=True)
        std = np.std(X_train, axis=(0, 2), keepdims=True)

        # TODO: log histone-wise mean and std

        std[std == 0] = 1  # Avoid division by zero

        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std

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

    def torch_train_pipeline(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
    ) -> None:
        best_val_loss = float("inf")

        epochs_without_improvement = 0

        for epoch in range(self.cfg.epochs):
            model.train()

            epoch_loss = AverageMeter()
            epoch_spearman_corr = AverageMeter()
            epoch_pearson_corr = AverageMeter()
            epoch_r_squared = AverageMeter()

            for batch_idx, (X_batch, y_batch) in enumerate(self.train_loader):
                X_batch, y_batch = (
                    X_batch.to(self.cfg.device),
                    y_batch.to(self.cfg.device),
                )

                # Update model based on the batch
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss.update(loss.item(), self.cfg.batch_size)

                # Compute metrics
                spearman_corr, pearson_corr, r_squared = self.compute_metrics(
                    y_pred.detach().cpu().numpy(),
                    y_batch.detach().cpu().numpy(),
                )

                epoch_spearman_corr.update(spearman_corr, self.cfg.batch_size)
                epoch_pearson_corr.update(pearson_corr, self.cfg.batch_size)
                epoch_r_squared.update(r_squared, self.cfg.batch_size)

            # Log training metrics
            self.logger.log.info(
                f"Epoch [{epoch}] -> "
                f"Loss: {epoch_loss.avg}, Spearman Correlation "
                f"{epoch_spearman_corr.avg}, Pearson Correlation: "
                f"{epoch_pearson_corr.avg}, R^2: {epoch_r_squared.avg}"
            )

            # Update tensorboard
            self.summary_writer.add_scalar("Loss/train", epoch_loss.avg, epoch)
            self.summary_writer.add_scalar(
                "Spearman Correlation/train", epoch_spearman_corr.avg, epoch
            )
            self.summary_writer.add_scalar(
                "Pearson Correlation/train", epoch_pearson_corr.avg, epoch
            )
            self.summary_writer.add_scalar(
                "R^2/train", epoch_r_squared.avg, epoch
            )

            # Validation
            (
                val_loss,
                val_spearman_corr,
                val_pearson_corr,
                val_r_squared,
            ) = self.validate(model, criterion, epoch)

            self.summary_writer.add_scalar("Loss/val", val_loss, epoch)
            self.summary_writer.add_scalar(
                "Spearman Correlation/val", val_spearman_corr, epoch
            )
            self.summary_writer.add_scalar(
                "Pearson Correlation/val", val_pearson_corr, epoch
            )
            self.summary_writer.add_scalar("R^2/val", val_r_squared, epoch)

            if val_loss < best_val_loss:
                self.logger.log.info("IMPROVED! ******")
                best_val_loss = val_loss
                epochs_without_improvement = 0
                self.save_weights(
                    f"{self.run_dir['checkpoints']}/best_model.pth"
                )
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.cfg.patience:
                self.logger.log.info(
                    f"Early stopping at epoch {epoch} with val_loss: {val_loss}"
                )
                break

    @torch.no_grad()
    def validate(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        epoch: int,
    ) -> Tuple[float, float, float, float]:
        """
        Evaluate the model on the validation set.

        Args:
            model(nn.Module): The model to evaluate.
            criterion(nn.Module): The loss function.
            epoch(int): The current epoch.

        Returns:
            The validation loss, Spearman correlation, Pearson correlation, and R^2.
        """
        model.eval()

        y_preds: torch.Tensor = torch.tensor([]).to(self.cfg.device)
        y_true: torch.Tensor = torch.tensor([]).to(self.cfg.device)

        for X_batch, y_batch in self.val_loader:
            X_batch, y_batch = (
                X_batch.to(self.cfg.device),
                y_batch.to(self.cfg.device),
            )

            y_pred = model(X_batch)
            y_preds = torch.cat((y_preds, y_pred), dim=0)
            y_true = torch.cat((y_true, y_batch), dim=0)

        val_loss = criterion(y_preds, y_true).item()
        (
            val_spearman_corr,
            val_pearson_corr,
            val_r_squared,
        ) = self.compute_metrics(
            y_preds.detach().cpu().numpy(), y_true.detach().cpu().numpy()
        )

        self.logger.log.info(
            f"Epoch [{epoch}] -> "
            f"Validation loss: {val_loss}, Spearman Correlation: "
            f"{val_spearman_corr}, Pearson Correlation: {val_pearson_corr}, "
            f"R^2: {val_r_squared}"
        )

        return val_loss, val_spearman_corr, val_pearson_corr, val_r_squared

    def compute_metrics(
        self, y_preds: npt.NDArray, y_true: npt.NDArray
    ) -> Tuple[float, float, float]:
        """
        Compute Spearman correlation, Pearson correlation, and R^2.

        Args:
            y_preds(npt.NDArray): The predicted values.
            y_true(npt.NDArray): The true values.

        Returns:
            The Spearman correlation
            The Pearson correlation
            The R^2
        """

        y_preds = y_preds.flatten()
        y_true = y_true.flatten()

        spearman_corr = spearman_correlation(y_preds, y_true)
        pearson_corr = pearson_correlation(y_preds, y_true)
        r2 = r_squared(y_preds, y_true)

        return spearman_corr, pearson_corr, r2

    @abstractmethod
    def fit(
        self,
    ) -> None:
        pass

    @abstractmethod
    def predict(self, X: npt.NDArray) -> npt.NDArray:
        pass

    @abstractmethod
    def save_weights(self, path: str) -> None:
        """
        Save the model weights to a file.

        Args:
            path(str): The path to save the weights.

        Returns:
            None
        """
        pass

    @abstractmethod
    def load_weights(self, path: str) -> None:
        pass
