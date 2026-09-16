#!/usr/bin/env bash
set -e

# Пути к моделям
ONNX_MODEL="model_repository/dog_breed_onnx/model.onnx"
TRT_ENGINE="model_repository/dog_breed_trt/model.plan"

# Создаем директорию под TensorRT план, если она отсутствует
mkdir -p $(dirname $TRT_ENGINE)

echo "🚀 Запуск конвертации ONNX в TensorRT..."
echo "Входной файл: $ONNX_MODEL"
echo "Выходной файл: $TRT_ENGINE"

# Проверяем наличие утилиты trtexec в системе
if command -v trtexec &> /dev/null; then
    # Конвертируем с включением FP16 оптимизации для ускорения работы на GPU
    trtexec \
        --onnx=$ONNX_MODEL \
        --saveEngine=$TRT_ENGINE \
        --fp16 \
        --workspace=4096

    echo "✅ Модель успешно конвертирована в TensorRT: $TRT_ENGINE"
else
    echo "❌ Ошибка: утилита 'trtexec' не найдена."
    echo "Запустите этот скрипт внутри официального контейнера NVIDIA Triton или TensorRT (например, nvcr.io/nvidia/tritonserver)."
    exit 1
fi
