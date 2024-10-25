import logging
import os
import shutil
from datetime import datetime
from typing import Any, Callable, Dict, Type, TypeVar

import numpy as np
import numpy.typing as npt
import orjson
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score


class dotdict(dict):
    """
    A dictionary that allows access to its keys as attributes.
    """

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(f"Attribute {name} not found")
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"Attribute {name} not found")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __setitem__(self, key: str, value: Any) -> None:
        if isinstance(value, dict):
            value = dotdict(value)
        super().__setitem__(key, value)

    def copy(self) -> Any:
        data = super().copy()
        return self._class_(data)


class CustomLogger(object):
    instance: "CustomLogger"

    def __new__(
        cls: Type["CustomLogger"], *args: Any, **kwargs: Any
    ) -> "CustomLogger":
        # Singleton pattern
        if not hasattr(cls, "instance"):
            cls.instance = super(CustomLogger, cls).__new__(cls)
        return cls.instance

    def __init__(
        self,
        log_file: str | None = None,
        log_level: int = logging.INFO,
    ) -> None:
        self.log = logging.getLogger(__name__)
        if not self.log.hasHandlers():
            self.log.setLevel(log_level)

            # Stream handler
            stream_handler = logging.StreamHandler()
            # stream_handler.setLevel(log_level)
            stream_format = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s"
            )
            stream_handler.setFormatter(stream_format)
            self.log.addHandler(stream_handler)

            # File handler
            if log_file is not None:
                file_handler = logging.FileHandler(
                    log_file, mode="a", encoding="utf-8"
                )
                file_handler.setLevel(log_level)
                file_format = logging.Formatter(
                    "%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s"
                )
                file_handler.setFormatter(file_format)
                self.log.addHandler(file_handler)

    def getLogger(self) -> logging.Logger:
        return self.log

    def addFileHandler(self, log_file: str) -> None:
        # check first that the file handler is not already added
        for handler in self.log.handlers:
            if isinstance(handler, logging.FileHandler):
                self.log.warning(
                    "File handler already added, please remove it first"
                )

        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(self.log.level)
        file_format = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s"
        )
        file_handler.setFormatter(file_format)
        self.log.addHandler(file_handler)

    def setLogFilePath(self, log_file: str) -> None:
        if not os.path.exists(os.path.dirname(log_file)):
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

        for handler in self.log.handlers:
            if isinstance(handler, logging.FileHandler):
                self.log.removeHandler(handler)

        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_format = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s"
        )
        file_handler.setFormatter(file_format)
        self.log.addHandler(file_handler)

    def setLogLevel(self, log_level: int | str) -> None:
        self.log.setLevel(log_level)
        for handler in self.log.handlers:
            handler.setLevel(log_level)


F = TypeVar("F", bound=Callable[..., Any])


def log_exceptions_decorator(logger: logging.Logger) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    "Exception occurred in %s", func.__name__, exc_info=True
                )
                raise  # Re-raise the exception after logging it

        return wrapper  # type: ignore

    return decorator


class AverageMeter:
    """
    Computes and stores the average and current value.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """
        Reset the meter.
        """
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """
        Update the meter.

        Args:
            val (float): The value to update.
            n (int): The number of samples.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def pearson_correlation(
    predictions: npt.NDArray, targets: npt.NDArray
) -> float:
    """
    Calculate the Pearson correlation

    Args:
        predictions(npt.NDArray): The predicted values.
        targets(npt.NDArray): The target values.

    Returns:
        The Pearson correlation
    """
    assert (
        predictions.shape == targets.shape
    ), "Predictions and targets must have the same shape."
    assert (
        len(predictions.shape) == 1
    ), "Predictions and targets must be 1D arrays."

    # Manually calculate the Pearson correlation
    mean_predictions = np.mean(predictions)
    mean_targets = np.mean(targets)

    numerator = np.sum(
        (predictions - mean_predictions) * (targets - mean_targets)
    )
    denominator = np.sqrt(
        np.sum((predictions - mean_predictions) ** 2)
        * np.sum((targets - mean_targets) ** 2)
    )

    manual_corr = numerator / denominator

    # Calculate the Pearson correlation
    corr, _ = pearsonr(predictions, targets)

    if not np.isclose(manual_corr, corr):
        print(f"[Pearson] Manual calculation: {manual_corr}, scipy calculation: {corr}")

    # assert np.isclose(
    #     manual_corr, corr
    # ), f"Manual calculation: {manual_corr}, scipy calculation: {corr}"

    return corr


def spearman_correlation(
    predictions: npt.NDArray, targets: npt.NDArray
) -> float:
    """
    Calculate the Spearman correlation

    Args:
        predictions(npt.NDArray): The predicted values.
        targets(npt.NDArray): The target values.

    Returns:
        The Spearman correlation
    """
    assert (
        predictions.shape == targets.shape
    ), "Predictions and targets must have the same shape."
    assert (
        len(predictions.shape) == 1
    ), "Predictions and targets must be 1D arrays."
    # Get the ranks of predictions and targets
    pred_ranks = np.argsort(np.argsort(predictions))
    target_ranks = np.argsort(np.argsort(targets))

    # Calculate the differences in ranks
    d = pred_ranks - target_ranks

    # Calculate the squared differences
    d_squared = d**2

    # Number of observations
    n = len(predictions)

    # Compute Spearman's correlation using the formula
    rho = 1 - (6 * np.sum(d_squared)) / (n * (n**2 - 1))

    # Calculate the Spearman correlation
    spearman_corr = spearmanr(predictions, targets).correlation

    if not np.isclose(rho, spearman_corr):
        print(f"[Spearman] Manual calculation: {rho}, scipy calculation: {spearman_corr}")

    # assert np.isclose(
    #     rho, spearman_corr
    # ), f"Manual calculation: {rho}, scipy calculation: {spearman_corr}"

    return rho


def r_squared(predictions: npt.NDArray, targets: npt.NDArray) -> float:
    """
    Calculate the R^2 score

    Args:
        predictions(npt.NDArray): The predicted values.
        targets(npt.NDArray): The target values.

    Returns:
        The R^2 score
    """
    assert (
        predictions.shape == targets.shape
    ), "Predictions and targets must have the same shape."
    assert (
        len(predictions.shape) == 1
    ), "Predictions and targets must be 1D arrays."

    # Calculate the mean of the target values
    mean_targets = np.mean(targets)

    # Calculate the total sum of squares
    total_sum_squares = np.sum((targets - mean_targets) ** 2)

    # Calculate the residual sum of squares
    residual_sum_squares = np.sum((targets - predictions) ** 2)

    # Calculate the R^2 score
    manual_r_squared = 1 - (residual_sum_squares / total_sum_squares)

    # Use  sklearn's r2_score function
    sklearn_r_squared = r2_score(targets, predictions)

    if not np.isclose(manual_r_squared, sklearn_r_squared):
        print(
            f"[R2] Manual calculation: {manual_r_squared}, sklearn calculation: {sklearn_r_squared}"
        )

    # assert np.isclose(
    #     manual_r_squared, sklearn_r_squared
    # ), f"Manual calculation: {manual_r_squared}, sklearn calculation: {sklearn_r_squared}"

    return manual_r_squared


def setup_run(run_folder: str) -> Dict[str, str]:
    """
    Creates a new run folder under the specified run folder.

    Args:
        run_folder (str): The run folder path.

    Returns:
        Dict[str, str]: The paths to the subfolders.
    """

    date_and_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    run = f"{run_folder}/{date_and_time}"
    if not os.path.exists(run):
        os.makedirs(run, exist_ok=True)

    # Create subfolders -> models, logs, outputs, tensorboard
    checkpoints = f"{run}/checkpoints"
    logs = f"{run}/logs"
    outputs = f"{run}/outputs"
    tensorboard = f"{run}/tensorboard"

    os.makedirs(checkpoints, exist_ok=True)
    os.makedirs(logs, exist_ok=True)
    os.makedirs(outputs, exist_ok=True)
    os.makedirs(tensorboard, exist_ok=True)

    return {
        "checkpoints": checkpoints,
        "logs": logs,
        "outputs": outputs,
        "tensorboard": tensorboard,
    }


class Config(dotdict):
    """
    Interface for the configuration parsing.

    Attributes:
        env (str): The environment.
        verbose (str): The verbosity.
        log_folder (str): The log folder.
        config_file (str): The config file.
    """

    def __init__(
        self,
        log_folder: str = "logs",
        config_file: str = "configuration/config.json",
        verbose: bool = True,
    ) -> None:
        super().__init__()

        self.verbose = verbose
        self.log_folder = log_folder
        self.config_file = config_file
        self.parse("general")

        if self.cuda and torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

    def parse(self, module: str) -> None:
        with open(self.config_file, "r") as f:
            config = orjson.loads(f.read())

        try:
            config = config[module]
        except KeyError:
            raise KeyError(
                f"{module} configuration not found "
                f"in the config file ({self.config_file})"
            )

        for key, value in config.items():
            self[key] = value

    def persist(self, output_dir: str) -> None:
        """
        Persist the configuration to disk.

        Args:
            output_dir (str): The output path.
        Returns:
            None
        """
        dst = os.path.join(output_dir, "config.json")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(self.config_file, dst)

    # boolean function to check for existence of a key in the config
    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            raise TypeError(f"key must be a string, not {type(key)}")
        return key in self.keys()


# Example usage
# netconfig = Config()
# netconfig.parse("NNetWrapper")
# print(netconfig.verbose)
# print(netconfig.optimizer)
# print(netconfig.optimizer_args)
# print(netconfig.optimizer_args.weight_decay)
