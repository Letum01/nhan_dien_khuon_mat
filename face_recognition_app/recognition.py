"""One-person webcam diagnostic: shared Haar preprocessing -> CNN embedding -> SVM.

The SVM class probability maximum is used consistently for the on-screen label.
Do not interpret the probability as proof that the input is a known real person.
"""
import collections
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
from keras.models import load_model

from .cnn import L2Normalization
from .config import (
    CNN_MODEL_PATH, DEFAULT_MODEL_PATH, IMAGE_SIZE,
    CONFIDENCE_THRESHOLD, MARGIN_THRESHOLD, DEBUG_INTERVAL_SECONDS,
)
from .dataset import CASCADE_PATH, detect_faces, crop_with_margin, normalize_face_crop
from .geometry import extract_face_geometry, draw_facial_geometry

PREVIEW_WINDOW = "Anh khuon mat dua vao CNN (128x128)"


def run_webcam(model_path: Path = DEFAULT_MODEL_PATH, cnn_path: Path = CNN_MODEL_PATH) -> None:
    model_path = Path(model_path).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy model.pkl: {model_path}")
    # Load only a .pkl created by your own trusted project.
    with model_path.open("rb") as file:
        bundle = pickle.load(file)
    svm = bundle["svm"]
    class_names = np.asarray(bundle["class_names"])
    use_geometry = bundle.get("use_geometry", False)
    if tuple(bundle["image_size"]) != tuple(IMAGE_SIZE):
        raise ValueError(f"Sai kích thước model: {bundle['image_size']} != {IMAGE_SIZE}")
    valid_preprocessors = {
        "haar_1.08_8_80_margin_0.12_rgb",
        "haar_1.08_8_80_margin_0.12_rgb_geometry_v1",
        "haar_1.08_6_70_margin_0.12_rgb_geometry_v2",
    }
    if bundle.get("preprocessing_version") not in valid_preprocessors:
        print("CẢNH BÁO: model.pkl có thể được train bằng cấu hình cắt mặt cũ; cần train lại với dataset.py mới.")
    stored_path = Path(bundle.get("cnn_path", cnn_path))
    if not stored_path.is_absolute():
        stored_path = model_path.parent / stored_path
    stored_path = stored_path.resolve()
    if not stored_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy CNN tương ứng với model.pkl: {stored_path}")

    extractor = load_model(stored_path, compile=False, safe_mode=False)
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    if detector.empty():
        raise RuntimeError(f"Không thể mở Haar Cascade: {CASCADE_PATH}")
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Không mở được webcam. Đóng ứng dụng khác đang dùng camera.")

    print(f"Đang dùng SVM: {model_path}")
    print(f"Đang dùng CNN: {stored_path}")
    print("Danh tính đã học:", ", ".join(map(str, class_names)))
    print("Chế độ thử 1 người; Q: thoát. Cửa sổ phụ là ảnh CNN nhận vào.")
    last_debug = 0.0
    preview = np.zeros((256, 256, 3), dtype=np.uint8)
    history_window = collections.deque(maxlen=6)
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Không đọc được webcam")
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detect_faces(rgb, detector=detector)
            if len(faces):
                crops = []
                valid_boxes = []
                for box in faces:
                    bx, by, bw, bh = map(int, box)
                    c = crop_with_margin(rgb, (bx, by, bw, bh))
                    if c.size:
                        crops.append(normalize_face_crop(c))
                        valid_boxes.append((bx, by, bw, bh))

                if crops:
                    # Update preview window with the largest face crop
                    largest_idx = max(range(len(valid_boxes)), key=lambda i: valid_boxes[i][2] * valid_boxes[i][3])
                    primary_crop = crops[largest_idx]
                    view_rgb = np.clip(primary_crop * 255, 0, 255).astype(np.uint8)
                    preview = cv2.resize(
                        cv2.cvtColor(view_rgb, cv2.COLOR_RGB2BGR),
                        (256, 256), interpolation=cv2.INTER_NEAREST,
                    )

                    # Extract geometric features (Eyes, Nose, Mouth) for each detected face
                    all_geo_features = []
                    all_landmarks = []
                    for c in crops:
                        geo_vec, lms = extract_face_geometry(c)
                        all_geo_features.append(geo_vec)
                        all_landmarks.append(lms)

                    # Draw geometric wireframe on the preview window for the primary face
                    primary_lms = all_landmarks[largest_idx]
                    scale_x = preview.shape[1] / float(IMAGE_SIZE[0])
                    scale_y = preview.shape[0] / float(IMAGE_SIZE[1])
                    scaled_primary_lms = {
                        k: (v[0] * scale_x, v[1] * scale_y) for k, v in primary_lms.items()
                    }
                    preview = draw_facial_geometry(preview, scaled_primary_lms)

                    # Fast batch inference: directly call extractor without Keras predict overhead
                    batch_tensor = np.asarray(crops, dtype=np.float32)
                    embeddings = extractor(batch_tensor, training=False).numpy()

                    # Combine CNN embeddings with geometric features if model was trained with fusion
                    if use_geometry:
                        geo_tensor = np.asarray(all_geo_features, dtype=np.float32)
                        features_fused = np.hstack([embeddings, geo_tensor])
                        all_probabilities = np.asarray(svm.predict_proba(features_fused))
                    else:
                        all_probabilities = np.asarray(svm.predict_proba(embeddings))

                    classes = np.asarray(svm.classes_, dtype=int)

                    now = time.monotonic()
                    debug_log = (now - last_debug >= DEBUG_INTERVAL_SECONDS)
                    if debug_log:
                        last_debug = now

                    for idx, (bx, by, bw, bh) in enumerate(valid_boxes):
                        probs = all_probabilities[idx]
                        best_pos = int(np.argmax(probs))
                        predicted_id = int(classes[best_pos])
                        if not 0 <= predicted_id < len(class_names):
                            raise ValueError(f"Nhãn SVM không hợp lệ: {predicted_id}")
                        name = str(class_names[predicted_id])
                        confidence = float(probs[best_pos])
                        second = float(np.partition(probs, -2)[-2]) if len(probs) > 1 else 0.0
                        margin = confidence - second
                        uncertain = confidence < CONFIDENCE_THRESHOLD or margin < MARGIN_THRESHOLD

                        # Áp dụng Temporal Smoothing (Bỏ phiếu đa số 6 frame) cho khuôn mặt chính
                        if idx == largest_idx:
                            history_window.append(name if not uncertain else "Uncertain")
                            counts = collections.Counter(history_window)
                            most_common_name, occurrences = counts.most_common(1)[0]
                            # Nếu đa số trong 6 frame gần nhất cùng một người, ưu tiên hiển thị ổn định
                            if occurrences >= 3 and most_common_name != "Uncertain":
                                name = most_common_name
                                uncertain = False

                        label = f"{'Chua chac: ' if uncertain else ''}{name} ({confidence:.0%})"
                        color = (0, 165, 255) if uncertain else (0, 255, 0)
                        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, 2)
                        cv2.putText(frame, label, (bx, max(30, by - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

                        # Draw the 4 biometric points (eyes, nose, mouth) directly on the camera frame
                        lms = all_landmarks[idx]
                        sx = float(bw) / float(IMAGE_SIZE[0])
                        sy = float(bh) / float(IMAGE_SIZE[1])
                        for pt_name, (lx, ly) in lms.items():
                            fx = int(bx + lx * sx)
                            fy = int(by + ly * sy)
                            pt_color = (0, 255, 0) if "eye" in pt_name else ((0, 0, 255) if pt_name == "nose" else (255, 0, 0))
                            cv2.circle(frame, (fx, fy), 3, pt_color, -1, cv2.LINE_AA)

                        if debug_log and idx == largest_idx:
                            print(f"\n--- Mặt lớn nhất: x={bx}, y={by}, w={bw}, h={bh} ---")
                            for label_id, probability in zip(classes, probs):
                                print(f"  {class_names[int(label_id)]}: {probability:.2%}")
                            print(f"  Theo predict_proba: {name}; Confidence={confidence:.2%}; Delta={margin:.2%}")
                else:
                    preview = np.zeros((256, 256, 3), dtype=np.uint8)
                    cv2.putText(preview, "Vung mat rong", (38, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            else:
                preview = np.zeros((256, 256, 3), dtype=np.uint8)
                cv2.putText(preview, "Khong co mat", (38, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(frame, "Khong phat hien mat", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            cv2.imshow("Face Recognition - CNN + SVM", frame)
            cv2.imshow(PREVIEW_WINDOW, preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
