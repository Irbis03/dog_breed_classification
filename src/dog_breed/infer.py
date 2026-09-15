import sys

import albumentations as A
import cv2
import torch

from dog_breed.models.module import DogClassifierModule


def predict(image_path: str, model_path: str = "models/model.ckpt"):
    transform = A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    augmented = transform(image=image)["image"]
    tensor_img = torch.tensor(
        augmented.transpose(2, 0, 1) / 255.0, dtype=torch.float32
    ).unsqueeze(0)

    model = DogClassifierModule.load_from_checkpoint(model_path)
    model.eval()

    with torch.no_grad():
        logits = model(tensor_img)
        pred_class = logits.argmax(dim=1).item()

    print(f"Предсказанный класс: {pred_class}")
    return pred_class


def main():
    if len(sys.argv) > 1:
        predict(sys.argv[1])
    else:
        print("Использование: dog-infer <путь_к_картинке>")


if __name__ == "__main__":
    main()
