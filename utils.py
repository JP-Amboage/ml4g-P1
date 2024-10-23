import logging
import os
import shutil
from typing import Any, Callable, Type, TypeVar

import orjson
import torch


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

    def setLogLevel(self, log_level: int) -> None:
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
