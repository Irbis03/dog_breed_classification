import os

import torch
from fire import Fire

from dog_breed.models.module import DogClassifierModule


def export_to_onnx(
    ckpt_path: str = "tmp/checkpoints/best.ckpt",
    output_path: str = "model_repository/dog_breed_onnx/model.onnx",
    img_size: int = 224,
):
    """Экспортирует PyTorch Lightning чекпоинт в ONNX формат."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 1. Загрузка модели
    model = DogClassifierModule.load_from_checkpoint(ckpt_path)
    model.eval()

    # 2. Фейковый тензор
    # Нужен для того, чтобы ONNX «понял» форму и тип данных на входе сети.
    dummy_input = torch.randn(1, 3, img_size, img_size, device="cpu")

    # 3. Экспорт
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"Модель успешно экспортирована в ONNX: {output_path}")


if __name__ == "__main__":
    Fire(export_to_onnx)
