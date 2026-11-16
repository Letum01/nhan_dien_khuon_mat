"""Điểm chạy CLI để huấn luyện mô hình nhận diện."""
import argparse
from pathlib import Path

from face_recognition_app.config import DEFAULT_DATASET_DIR, DEFAULT_MODEL_PATH
from face_recognition_app.training import train_and_save


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CNN embeddings and an SVM face classifier.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--baseline", type=Path, help="Existing model bundle to compare on the same test images")
    parser.add_argument(
        "--split-method", choices=["session", "random", "auto"], default="auto",
        help="auto: tự động nhận diện 1 phiên hoặc nhiều phiên (khuyến nghị); session: chia theo phiên; random: chia ngẫu nhiên",
    )
    args = parser.parse_args()
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size phải nằm giữa 0 và 1")
    if args.test_size != 0.3:
        print(f"Tỷ lệ tùy chỉnh: train {1 - args.test_size:.0%}, test {args.test_size:.0%}")
    train_and_save(args.dataset, args.output, args.test_size, args.baseline, split_method=args.split_method)


if __name__ == "__main__":
    main()
