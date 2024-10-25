import os
from typing import Dict

from base_model import BaseModel
from cnn_model import CNNModel
from utils import Config, CustomLogger, setup_run

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


logger = CustomLogger()


def get_model(model_name: str, run_dir: Dict[str, str]) -> BaseModel:
    """
    Returns the model instance given the model name

    Args:
        model_name (str): The model name.
        run_dir (Dict[str, str]): The run directory.

    Returns:
        The model instance.
    """
    logger.log.info(f"Using model: {model_name}")
    if model_name == "cnn":
        return CNNModel(run_dir)

    raise ValueError(f"Model {model_name} not found")


def main() -> None:
    # Setup run directory
    os.makedirs("runs", exist_ok=True)
    run_dir = setup_run("runs")

    # Configuration object
    cfg = Config()
    cfg.persist(output_dir=run_dir["outputs"])

    # Setup logger
    log_file = f"{run_dir['logs']}/train.log"
    log_level = "DEBUG" if cfg.debug else "INFO"
    logger.setLogFilePath(log_file)
    logger.setLogLevel(log_level)
    logger.log.debug(
        "Verbose mode activated"
    )  # Will only print if debug is True

    logger.log.info(
        f"Run folder {run_dir['checkpoints'].split('/')[1]} setup successfully"
    )

    # Launch training
    model = get_model(cfg.model_name, run_dir)
    model.fit()


if __name__ == "__main__":
    main()
