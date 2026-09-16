import logging
from pathlib import Path

import gdown

logger = logging.getLogger(__name__)


def ensure_datasets_downloaded(data_dir: Path = Path("data/raw")) -> None:
    """Проверяет наличие данных.
    Если данных нет, скачивает их из google drive."""
    labels_file = data_dir / "labels.csv"
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    if labels_file.exists() and train_dir.exists():
        logger.info("Данные уже присутствуют локально.")
        return

    logger.info("Данные не найдены. Выполняется загрузку...")
    try:
        folder_train_url = "https://drive.google.com/drive/folders/1hMvw66rl_Mg-1IBCZQiYhDEi6wDm9yfu?usp=sharing"
        folder_test_url = "https://drive.google.com/drive/folders/1bjJDHgg7Yb0QKMWhyaVBcLykfUHBPSYz?usp=sharing"
        labels_csv_url = "https://drive.google.com/file/d/1PhbrzRpR_Qnc_0jRnaebEqMUfgd7HHp7/view?usp=sharing"

        gdown.download_folder(url=folder_train_url, output=str(train_dir), quiet=False)
        gdown.download_folder(url=folder_test_url, output=str(test_dir), quiet=False)
        gdown.download(url=labels_csv_url, output=str(labels_file), quiet=False)

        logger.info("Данные успешно загружены.")
    except Exception as e:
        logger.warning(f"Ошибка : {e}")
