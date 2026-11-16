
import argparse
import hashlib
import shutil
from datetime import datetime
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from face_recognition_app.config import DEFAULT_DATASET_DIR, SUPPORTED_EXTENSIONS


def session_for(path: Path) -> str:
    """Use EXIF date when available, otherwise filesystem mtime (only an estimate)."""
    try:
        with Image.open(path) as image:
            date = image.getexif().get(36867) or image.getexif().get(306)
        if date:
            parsed = datetime.strptime(str(date), "%Y:%m:%d %H:%M:%S")
            return parsed.strftime("%Y-%m-%d")
    except (OSError, UnidentifiedImageError, ValueError):
        pass
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output", type=Path, default=Path("dataset_sessions"))
    args = parser.parse_args()
    if not args.source.is_dir():
        raise FileNotFoundError(f"Không tìm thấy dataset nguồn: {args.source}")
    if args.source.resolve() == args.output.resolve():
        raise ValueError("Output phải khác source để giữ nguyên dataset gốc.")
    copied = 0
    for person in sorted(p for p in args.source.iterdir() if p.is_dir()):
        for image in sorted(p for p in person.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS):
            session = session_for(image)
            digest = hashlib.sha1(str(image.resolve()).encode()).hexdigest()[:8]
            destination = args.output / person.name / session / f"{image.stem}_{digest}{image.suffix.lower()}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, destination)
            copied += 1
    print(f"Copied {copied} images to {args.output.resolve()}")
    print("LƯU Ý: nhóm fallback theo ngày sửa tệp không xác nhận phiên chụp thực tế.")
    for person in sorted(p for p in args.output.iterdir() if p.is_dir()):
        sessions = sorted(p.name for p in person.iterdir() if p.is_dir())
        print(f"{person.name}: {len(sessions)} sessions: {', '.join(sessions)}")


if __name__ == "__main__":
    main()