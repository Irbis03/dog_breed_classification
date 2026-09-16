import os
import subprocess
from pathlib import Path

import hydra
import mlflow
import pytorch_lightning as pl
from dvc.repo import Repo
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from dog_breed.data.datamodule import DogDataModule
from dog_breed.models.module import DogClassifierModule
from dog_breed.utils.ensure_datasets_download import ensure_datasets_downloaded
from dog_breed.utils.plotting import PlotMetricsCallback


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
    # 1. Загрузка данных при необходимости
    # Достаем параметры DVC из Hydra-конфига data
    dvc_config = cfg.get("dvc", {})
    if dvc_config != {}:
        data_dir_path = dvc_config.get("data_dir_path")
        data_remote_name = dvc_config.get("data_remote_name")

    ensure_datasets_downloaded(Path(data_dir_path))

    repo = Repo()

    repo.add(data_dir_path)
    repo.push(targets=[data_dir_path], remote=data_remote_name)

    # 2. Инициализация MLflow
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    # Запускаем явный контекст MLflow Run
    with mlflow.start_run() as run:
        mlflow_logger = MLFlowLogger(
            experiment_name=cfg.mlflow.experiment_name,
            tracking_uri=cfg.mlflow.tracking_uri,
            run_id=run.info.run_id,
        )

        # Запись git commit id в MLflow
        commit_id = get_git_commit_id()
        mlflow.log_param("git_commit_id", commit_id)

        # 3. Модель и данные
        datamodule = DogDataModule(**cfg.data)
        model = DogClassifierModule(**cfg.model)

        # 4.1. Коллбек для графиков
        plot_callback = PlotMetricsCallback(output_dir=cfg.plots_dir)

        # 4.2 Создаем коллбек с нужной папкой для сохранения
        checkpoint_callback = ModelCheckpoint(
            # Все чекпоинты будут сохраняться в tmp/checkpoints/
            dirpath="tmp/checkpoints",
            filename="best",
            save_top_k=1,
            monitor="val_loss",
            mode="min",
        )

        # 5. Trainer
        trainer = pl.Trainer(
            max_epochs=cfg.trainer.max_epochs,
            accelerator=cfg.trainer.accelerator,
            devices=cfg.trainer.devices,
            logger=mlflow_logger,
            callbacks=[plot_callback, checkpoint_callback],
        )

        # 6. Запуск обучения
        trainer.fit(model, datamodule=datamodule)

        # 7. Отправка графика в MLflow как артефакта
        plot_path = os.path.join(cfg.plots_dir, "metrics_plot.png")
        if os.path.exists(plot_path):
            mlflow.log_artifact(plot_path)

        # 8. Сохраняем модель в формате MLflow для удобного Serving
        mlflow.pytorch.log_model(
            model,  # или model целиком
            artifact_path="model",
            serialization_format="pickle",
        )


if __name__ == "__main__":
    main()
