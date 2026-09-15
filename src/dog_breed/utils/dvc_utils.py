import logging
import os
import subprocess

logger = logging.getLogger(__name__)


def ensure_data_downloaded(data_dir: str = "data/raw") -> None:
    """Проверяет наличие данных.
    Если данных нет, скачивает через DVC или мок-генератор."""
    labels_file = os.path.join(data_dir, "labels.csv")
    train_dir = os.path.join(data_dir, "train")

    if os.path.exists(labels_file) and os.path.exists(train_dir):
        logger.info("Данные уже присутствуют локально.")
        return

    logger.info("Данные не найдены. Выполняется загрузка через DVC...")
    try:
        subprocess.run(["dvc", "pull"], check=True)
        logger.info("Данные успешно загружены через DVC.")
    except Exception as e:
        logger.warning(f"Ошибка при вызове dvc pull: {e}. Проверьте remote хранилище.")
