import argparse
import time
from pathlib import Path

import cv2

from face_recognition_app.config import (
    DEFAULT_DATASET_DIR,
    HAAR_MIN_FACE_SIZE,
    HAAR_MIN_NEIGHBORS,
    HAAR_SCALE_FACTOR,
)
from face_recognition_app.dataset import CASCADE_PATH, FACE_DETECTOR, detect_faces

parser = argparse.ArgumentParser(description="Chụp ảnh khuôn mặt từ webcam theo phiên")
parser.add_argument("--person", required=True, help="Tên người cần thu thập dữ liệu")
parser.add_argument("--session", required=True, help="Tên phiên chụp (ví dụ: session_1, 2026-11-15)")
parser.add_argument("--count", type=int, default=100, help="Số lượng ảnh cần chụp")
args = parser.parse_args()

if args.count <= 0:
    parser.error("--count phải lớn hơn 0")

folder = DEFAULT_DATASET_DIR / args.person / args.session
folder.mkdir(parents=True, exist_ok=True)

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not camera.isOpened():
    camera.release()
    camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise RuntimeError("Khong mo duoc webcam!")

detector = FACE_DETECTOR

saved = 0
last_capture = 0
started = False

print("Nhan S de bat dau chup, Q de thoat.")

try:
    while saved < args.count:
        ok, frame = camera.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detect_faces(rgb, detector=detector)

        preview = frame.copy()
        for x, y, w, h in faces:
            cv2.rectangle(
                preview, (x, y), (x + w, y + h),
                (0, 255, 0), 2
            )

        if started and len(faces) == 1:
            if time.monotonic() - last_capture >= 0.4:
                filename = folder / f"face_{int(time.time() * 1000)}.jpg"
                if cv2.imwrite(str(filename), frame):
                    saved += 1
                    last_capture = time.monotonic()
                    print(f"Da chup: {saved}/{args.count}")

        cv2.putText(
            preview,
            f"Captured: {saved}/{args.count} {'[RECORDING]' if started else '[PAUSED - Press S]'}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 0) if started else (0, 165, 255), 2
        )

        cv2.imshow("Face Capture - S: Start, Q: Quit", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            started = True
        elif key == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()

print(f"Hoan thanh: {saved} anh moi trong {folder}")
