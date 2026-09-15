import os
import matplotlib.pyplot as plt


def save_training_plots(
    metrics_history: dict, output_dir: str = "plots"
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss plot
    if "train_loss" in metrics_history and "val_loss" in metrics_history:
        ax1.plot(metrics_history["train_loss"], label="Train Loss")
        ax1.plot(metrics_history["val_loss"], label="Val Loss")
        ax1.set_title("Training & Validation Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()

    # Accuracy plot
    if "train_acc" in metrics_history and "val_acc" in metrics_history:
        ax2.plot(metrics_history["train_acc"], label="Train Acc")
        ax2.plot(metrics_history["val_acc"], label="Val Acc")
        ax2.set_title("Training & Validation Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "metrics_plot.png")
    plt.savefig(plot_path)
    plt.close()