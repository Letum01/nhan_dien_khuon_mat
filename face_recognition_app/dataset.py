"""Read face datasets and apply the SAME preprocessing used by webcam inference."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import (
    IMAGE_SIZE, SUPPORTED_EXTENSIONS, HAAR_SCALE_FACTOR,
    HAAR_MIN_NEIGHBORS, HAAR_MIN_FACE_SIZE, FACE_MARGIN_RATIO,
)

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_DETECTOR = cv2.CascadeClassifier(CASCADE_PATH)
if FACE_DETECTOR.empty():
    raise RuntimeError(f"Không tải được Haar Cascade: {CASCADE_PATH}")


def detect_faces(rgb: np.ndarray, detector=None) -> np.ndarray:
    """Detect faces in RGB, using identical Haar parameters for train and webcam."""
    detector = FACE_DETECTOR if detector is None else detector
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return detector.detectMultiScale(
        gray,
        scaleFactor=HAAR_SCALE_FACTOR,
        minNeighbors=HAAR_MIN_NEIGHBORS,
        minSize=HAAR_MIN_FACE_SIZE,
    )


def crop_with_margin(
    rgb: np.ndarray,
    box: tuple[int, int, int, int],
    margin_ratio: float = FACE_MARGIN_RATIO,
) -> np.ndarray:
    """Crop a detected face with the same margin in training and webcam."""
    x, y, width, height = map(int, box)
    margin_x = int(width * margin_ratio)
    margin_y = int(height * margin_ratio)
    left, top = max(0, x - margin_x), max(0, y - margin_y)
    right = min(rgb.shape[1], x + width + margin_x)
    bottom = min(rgb.shape[0], y + height + margin_y)
    return rgb[top:bottom, left:right]


def apply_clahe(rgb: np.ndarray) -> np.ndarray:
    """Cân bằng độ sáng thích ứng (CLAHE) trên kênh L trong không gian màu LAB."""
    if rgb.dtype != np.uint8:
        rgb_uint8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    else:
        rgb_uint8 = rgb
    lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def normalize_face_crop(rgb_crop: np.ndarray, align_face: bool = True) -> np.ndarray:
    """Chuẩn hóa ảnh mặt: Căn chỉnh 2 mắt nằm ngang -> Cân bằng sáng CLAHE -> Resize 128x128 -> float32 [0, 1]."""
    if rgb_crop.size == 0:
        raise ValueError("Vùng mặt sau khi crop rỗng")
    resized = cv2.resize(rgb_crop, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    if align_face:
        from .geometry import extract_landmarks, align_face_horizontal
        le, re, _, _ = extract_landmarks(resized)
        resized = align_face_horizontal(resized, le, re)

    enhanced = apply_clahe(resized)
    return enhanced.astype(np.float32) / 255.0


def prepare_face(image: Image.Image) -> np.ndarray | None:
    """For a training image: use the largest Haar detection, then normalize."""
    rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    faces = detect_faces(rgb)
    if len(faces) == 0:
        return None
    box = max(faces, key=lambda b: int(b[2]) * int(b[3]))
    return normalize_face_crop(crop_with_margin(rgb, tuple(map(int, box))))


try:
    from pillow_heif import register_heif_opener
except ImportError:
    register_heif_opener = None


def load_dataset(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray, list[Path], np.ndarray]:
    """Accept either flat folders or independent session folders, never mixed."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục dataset: {dataset_dir}")
    if register_heif_opener:
        register_heif_opener()
    elif any(p.suffix.lower() in {".heic", ".heif"} for p in dataset_dir.rglob("*")):
        raise ImportError("Dataset có ảnh HEIC; chạy: pip install pillow-heif")

    images, labels, image_paths, sessions = [], [], [], []
    for class_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        direct = [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
        nested_dirs = [p for p in class_dir.iterdir() if p.is_dir()]
        if direct and nested_dirs:
            raise ValueError(
                f"{class_dir.name}: đang trộn ảnh trực tiếp với thư mục session. "
                "Hãy thống nhất một kiểu cho từng người."
            )
        paths = sorted(p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
        count, skipped = 0, 0
        for path in paths:
            try:
                with Image.open(path) as source:
                    face = prepare_face(source)
                if face is None:
                    skipped += 1
                    print(f"Không phát hiện mặt, bỏ qua: {path}")
                    continue
                images.append(face)
                labels.append(class_dir.name)
                image_paths.append(path)
                sessions.append(path.parent.name if path.parent != class_dir else "__legacy_flat__")
                count += 1
            except (OSError, UnidentifiedImageError, ValueError, cv2.error) as error:
                skipped += 1
                print(f"Không đọc/xử lý được {path}: {error}")
        print(f"{class_dir.name}: dùng {count}/{len(paths)} ảnh, bỏ qua {skipped}")
        if count < 10:
            raise ValueError(f"{class_dir.name}: cần ít nhất 10 ảnh có mặt được Haar phát hiện, hiện có {count}.")
    if not images:
        raise ValueError("Không có ảnh hợp lệ trong dataset")
    return np.asarray(images), np.asarray(labels), image_paths, np.asarray(sessions)
