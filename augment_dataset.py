
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from face_recognition_app.config import DEFAULT_DATASET_DIR, SUPPORTED_EXTENSIONS


def augment_and_save(image_path: Path, output_dir: Path) -> int:
    try:
        with Image.open(image_path) as img:
            base_img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception as e:
        print(f"Bỏ qua ảnh lỗi {image_path}: {e}")
        return 0

    person_name = image_path.parent.name
    target_person_dir = output_dir / person_name
    target_person_dir.mkdir(parents=True, exist_ok=True)

    # 1. Lưu ảnh gốc
    stem = image_path.stem
    base_img.save(target_person_dir / f"{stem}_orig.jpg", quality=95)

    # 2. Xoay nhẹ -8 độ
    rot_left = base_img.rotate(-8, resample=Image.Resampling.BILINEAR, expand=False)
    rot_left.save(target_person_dir / f"{stem}_rot_l8.jpg", quality=95)

    # 3. Xoay nhẹ +8 độ
    rot_right = base_img.rotate(8, resample=Image.Resampling.BILINEAR, expand=False)
    rot_right.save(target_person_dir / f"{stem}_rot_r8.jpg", quality=95)

    # 4. Tăng nhẹ độ sáng và tương phản
    bright = ImageEnhance.Brightness(base_img).enhance(1.15)
    contrast = ImageEnhance.Contrast(bright).enhance(1.10)
    contrast.save(target_person_dir / f"{stem}_bright.jpg", quality=95)

    return 4


def main() -> None:
    parser = argparse.ArgumentParser(description="Nhân bản tập dữ liệu khuôn mặt tự động")
    parser.add_argument("--source", type=Path, default=DEFAULT_DATASET_DIR, help="Thư mục dataset gốc")
    parser.add_argument("--output", type=Path, default=Path("dataset_augmented"), help="Thư mục lưu dataset nhân bản")
    args = parser.parse_args()

    if not args.source.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục nguồn: {args.source}")

    print(f"Đang nhân bản dữ liệu từ '{args.source}' sang '{args.output}'...")
    total_generated = 0
    for person_dir in sorted(p for p in args.source.iterdir() if p.is_dir()):
        images = [p for p in person_dir.glob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
        count = 0
        for img_path in images:
            count += augment_and_save(img_path, args.output)
        total_generated += count
        print(f"  {person_dir.name}: {len(images)} ảnh gốc -> {count} ảnh nhân bản.")

    print(f"\n[HOÀN THÀNH] Đã tạo thành công {total_generated} ảnh tại: {args.output.resolve()}")
    print("Để huấn luyện mô hình trên tập dữ liệu mở rộng này, chạy lệnh:")
    print(f"  python train_model.py --dataset {args.output}")


if __name__ == "__main__":
    main()
