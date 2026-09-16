import cv2
import numpy as np
import onnxruntime as ort
from fire import Fire


def preprocess(img_path: str, img_size: int = 224) -> np.ndarray:
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Файл не найден: {img_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_size, img_size))

    # Нормализация ImageNet
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std

    # HWC -> CHW -> NCHW
    image = image.transpose(2, 0, 1)
    return np.expand_dims(image, axis=0).astype(np.float32)


def run_inference(
    image_path: str,
    onnx_path: str = "model_repository/dog_breed_onnx/1/model.onnx",
):
    """Выполняет инференс изображения с помощью ONNX Runtime."""
    tensor_input = preprocess(image_path)
    session = ort.InferenceSession(onnx_path)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    outputs = session.run([output_name], {input_name: tensor_input})
    probs = outputs[0][0]
    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])

    print(f"Предсказанный класс ID: {pred_class} (Уверенность: {confidence:.4f})")
    return {"class_id": pred_class, "confidence": confidence}


if __name__ == "__main__":
    Fire(run_inference)
