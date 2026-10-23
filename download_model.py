from pathlib import Path
from urllib.request import urlretrieve


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/1/"
    "pose_landmarker_full.task"
)

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_DIR = PROJECT_DIR / "models"

MODEL_PATH = MODEL_DIR / "pose_landmarker_full.task"


def download_model():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000:
        print("MediaPipe model already downloaded.")
        print(f"Location: {MODEL_PATH}")
        return

    print("Downloading MediaPipe Pose Landmarker...")
    print("This may take a little while.")

    temporary_path = MODEL_PATH.with_suffix(".download")

    try:
        urlretrieve(MODEL_URL, temporary_path)

        if temporary_path.stat().st_size < 1_000_000:
            raise RuntimeError(
                "Downloaded model appears incomplete."
            )

        temporary_path.replace(MODEL_PATH)

        print("Download complete!")
        print(f"Model saved to: {MODEL_PATH}")

    finally:
        temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    download_model()
