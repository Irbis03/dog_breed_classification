import json

import requests
from fire import Fire

from dog_breed.infer import preprocess  # или локальный препроцесс


def send_to_mlflow_serving(
    image_path: str, url: str = "http://127.0.0.1:5000/invocations"
):
    """Отправляет изображение на локальный эндпоинт MLflow Model Serving."""
    # 1. Подготовка изображения
    tensor_img = preprocess(image_path)

    # 2. Тело запроса (MLflow tensor format)
    payload = {"inputs": tensor_img.tolist()}
    headers = {"Content-Type": "application/json"}

    # 3. Отправка POST-запроса
    response = requests.post(url, data=json.dumps(payload), headers=headers)

    if response.status_code == 200:
        result = response.json()
        print(f"[MLflow Serving Success] Ответ сервера: {result}")
        return result
    else:
        print(f"[Error {response.status_code}]: {response.text}")
        return None


if __name__ == "__main__":
    Fire(send_to_mlflow_serving)
