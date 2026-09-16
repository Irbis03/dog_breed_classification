import os

import matplotlib.pyplot as plt
import pytorch_lightning as pl


class PlotMetricsCallback(pl.Callback):
    def __init__(self, output_dir: str = "plots"):
        super().__init__()
        self.output_dir = output_dir
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        # Пропускаем «проверочный» прогон сан-чек перед обучением
        if trainer.sanity_checking:
            return

        metrics = trainer.callback_metrics

        if "train_loss" in metrics:
            self.history["train_loss"].append(metrics["train_loss"].item())
        if "val_loss" in metrics:
            self.history["val_loss"].append(metrics["val_loss"].item())
        if "train_acc" in metrics:
            self.history["train_acc"].append(metrics["train_acc"].item())
        if "val_acc" in metrics:
            self.history["val_acc"].append(metrics["val_acc"].item())

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        os.makedirs(self.output_dir, exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Отрисовка Loss
        if self.history["train_loss"] or self.history["val_loss"]:
            if self.history["train_loss"]:
                ax1.plot(self.history["train_loss"], label="Train Loss", marker="o")
            if self.history["val_loss"]:
                ax1.plot(self.history["val_loss"], label="Val Loss", marker="o")
            ax1.set_title("Loss over Epochs")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.legend()
            ax1.grid(True)

        # Отрисовка Accuracy
        if self.history["train_acc"] or self.history["val_acc"]:
            if self.history["train_acc"]:
                ax2.plot(self.history["train_acc"], label="Train Acc", marker="o")
            if self.history["val_acc"]:
                ax2.plot(self.history["val_acc"], label="Val Acc", marker="o")
            ax2.set_title("Accuracy over Epochs")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Accuracy")
            ax2.legend()
            ax2.grid(True)

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, "metrics_plot.png")
        plt.savefig(plot_path)
        plt.close()
