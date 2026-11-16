"""Shared configuration for data preparation, training, and webcam inference."""
from pathlib import Path

IMAGE_SIZE = (128, 128)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}
DEFAULT_DATASET_DIR = Path("dataset")
DEFAULT_MODEL_PATH = Path("model.pkl")
CNN_MODEL_PATH = Path("model_cnn.keras")
RESULTS_PATH = Path("model_results.json")

# Same face detector settings in training and webcam.
HAAR_SCALE_FACTOR = 1.08
HAAR_MIN_NEIGHBORS = 6
HAAR_MIN_FACE_SIZE = (70, 70)
FACE_MARGIN_RATIO = 0.12

# Diagnostic thresholds only; not a guarantee of recognizing unknown people.
CONFIDENCE_THRESHOLD = 0.70
MARGIN_THRESHOLD = 0.15
DEBUG_INTERVAL_SECONDS = 2.0
