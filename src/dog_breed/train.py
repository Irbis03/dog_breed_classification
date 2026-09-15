import os
import subprocess

import hydra
import mlflow
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.loggers import MLFlowLogger

from dog_breed.data.datamodule import DogDataModule
from dog_breed.models.module import DogClassifierModule
from dog_breed.utils.dvc_utils import ensure_data_downloaded
from dog_breed.utils.plotting import save_training_plots


def get_git_commit_id() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train")
def main(cfg: DictConfig):
    # 1. Загрузка данных через DVC при необходимости
    ensure_data_downloaded(os.path.dirname(cfg.data.csv_path))

    # 2. Инициализация MLflow
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name,
        tracking_uri=cfg.mlflow.tracking_uri,
    )

    # 3. Модель и данные
    datamodule = DogDataModule(**cfg.data)
    model = DogClassifierModule(**cfg.model)

    # 4. Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        logger=mlflow_logger,
    )

    # Запись git commit id в MLflow
    commit_id = get_git_commit_id()
    mlflow_logger.experiment.log_param(mlflow_logger.run_id, "git_commit_id", commit_id)

    # 5. Запуск обучения
    trainer.fit(model, datamodule=datamodule)

    # 6. Сохранение графиков метрик
    metrics = {
        "train_loss": [
            float(v) for k, v in trainer.logged_metrics.items() if "train_loss" in k
        ],
        "val_loss": [
            float(v) for k, v in trainer.logged_metrics.items() if "val_loss" in k
        ],
        "train_acc": [
            float(v) for k, v in trainer.logged_metrics.items() if "train_acc" in k
        ],
        "val_acc": [
            float(v) for k, v in trainer.logged_metrics.items() if "val_acc" in k
        ],
    }
    save_training_plots(metrics, output_dir=cfg.plots_dir)


if __name__ == "__main__":
    main()
