"""Train CNN + SVM, evaluate and save reproducible diagnostics.

Splitting strategies:
- session: hold out whole, independently collected capture sessions.
- random: stratified image-level split, for smoke testing only.
"""
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .cnn import build_cnn
from .config import CNN_MODEL_PATH, DEFAULT_MODEL_PATH, IMAGE_SIZE
from .dataset import load_dataset
from .geometry import extract_face_geometry


def _session_split(y: np.ndarray, sessions: np.ndarray, test_size: float, seed: int = 42, min_sessions: int = 2, reserve_train_sessions: int = 1):
    """Select whole capture sessions for test, maintaining every identity in both sets."""
    by_class: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, (label, session) in enumerate(zip(y, sessions)):
        by_class[int(label)][str(session)].append(index)
    rng = np.random.default_rng(seed)
    train_indices, test_indices = [], []
    assignments = {}
    for label, groups in sorted(by_class.items()):
        keys = list(groups)
        if len(keys) < 2:
            raise ValueError(
                f"Lớp {label} chỉ có {len(keys)} phiên chụp. Cần ít nhất 2 phiên riêng biệt "
                "trong mỗi lớp (dataset/<người>/<phiên>/<ảnh>)."
            )
        if len(keys) < min_sessions:
            raise ValueError(
                f"Lớp {label} chỉ có {len(keys)} phiên; yêu cầu tối thiểu {min_sessions} "
                "phiên cho phép tách dữ liệu không rò rỉ."
            )
        rng.shuffle(keys)
        target = sum(len(groups[key]) for key in keys) * test_size
        selected, count = [keys[0]], len(groups[keys[0]])
        for key in keys[1:]:
            # Preserve enough training sessions for a separate validation session.
            if len(keys) - len(selected) <= reserve_train_sessions:
                break
            if abs(count + len(groups[key]) - target) < abs(count - target):
                selected.append(key)
                count += len(groups[key])
        for key in keys:
            (test_indices if key in selected else train_indices).extend(groups[key])
        assignments[str(label)] = {
            "test_sessions": selected,
            "train_sessions": [k for k in keys if k not in selected],
        }
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return np.asarray(train_indices, dtype=int), np.asarray(test_indices, dtype=int), assignments


def _evaluate(y_true, y_pred, probabilities, class_names, paths, sessions, output_dir: Path, prefix: str, save_mistakes: bool = False):
    labels = np.arange(len(class_names))
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names.tolist(), output_dict=True, zero_division=0
    )
    mistakes = []
    mistakes_dir = output_dir / f"{prefix}_misclassified"
    for index, (actual, predicted) in enumerate(zip(y_true, y_pred)):
        if actual == predicted:
            continue
        source = Path(paths[index])
        confidence = float(probabilities[index, predicted])
        saved_path = None
        if save_mistakes:
            copied_to = mistakes_dir / str(class_names[actual]) / f"pred_{class_names[predicted]}__{source.name}"
            copied_to.parent.mkdir(parents=True, exist_ok=True)
            try:
                copied_to.write_bytes(source.read_bytes())
                saved_path = str(copied_to.resolve())
            except OSError:
                pass
        mistakes.append({
            "image": str(source.resolve()),
            "saved_copy": saved_path,
            "true_label": str(class_names[actual]),
            "predicted_label": str(class_names[predicted]),
            "confidence": confidence,
            "session": str(sessions[index]),
        })
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "misclassified_count": len(mistakes),
        "misclassified_images": mistakes,
    }


def train_and_save(
    dataset_dir: Path,
    model_path: Path,
    test_size: float,
    baseline_model: Path | None = None,
    split_method: str = "session",
) -> None:
    """Train using a session holdout or a stratified random image-level split."""
    from tensorflow import keras

    model_path = Path(model_path)
    if model_path.name != DEFAULT_MODEL_PATH.name:
        raise ValueError("Chỉ lưu model.pkl; không tạo tên model khác.")
    if baseline_model and Path(baseline_model).resolve() == model_path.resolve():
        raise ValueError("Baseline không được là chính model.pkl đang ghi đè.")
    if split_method not in {"session", "random", "auto"}:
        raise ValueError("split_method phải là 'session', 'random' hoặc 'auto'")
    X, names, paths, sessions = load_dataset(dataset_dir)
    # Convert Python lists to NumPy arrays for indexing with train/test indices.
    names = np.asarray(names)
    paths = np.asarray(paths, dtype=object)
    sessions = np.asarray(sessions, dtype=object)
    class_names, y = np.unique(names, return_inverse=True)
    if len(class_names) < 2:
        raise ValueError("Cần ít nhất 2 danh tính để huấn luyện.")

    has_insufficient_sessions = (
        "__legacy_flat__" in sessions
        or any(len(set(sessions[names == name])) < 2 for name in class_names)
    )

    resolved_split = split_method
    if split_method == "auto":
        resolved_split = "random" if has_insufficient_sessions else "session"
        if resolved_split == "random":
            print("[THÔNG BÁO] Phát hiện dataset có 1 phiên chụp (flat structure). Tự động dùng phân chia ngẫu nhiên kết hợp Data Augmentation.")
        else:
            print("[THÔNG BÁO] Phát hiện dataset có đủ nhiều phiên chụp. Tự động dùng phân chia theo phiên (session holdout).")

    if resolved_split == "random":
        indices = np.arange(len(y))
        counts = np.bincount(y, minlength=len(class_names))
        if counts.min() < 3:
            raise ValueError("Chế độ random cần ít nhất 3 ảnh hợp lệ cho mỗi danh tính.")
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=42, stratify=y
        )
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        fit_local, val_local = train_test_split(
            np.arange(len(y_train)), test_size=0.15, random_state=43, stratify=y_train
        )
        split_info = {}
        validation_sessions = {}
        split_name = "stratified_random_image_split"
        print("LƯU Ý: Đang huấn luyện với 1 phiên duy nhất. Đã tự động kích hoạt Data Augmentation (lật, xoay, tương phản, độ sáng) để tối ưu độ chính xác.")
    else:
        if has_insufficient_sessions:
            raise ValueError(
                "Không thể đánh giá theo phiên: mỗi lớp cần tối thiểu 2 phiên độc lập. "
                "Dùng cấu trúc dataset/<người>/<phiên>/<ảnh> và không suy phiên thật chỉ từ mtime."
            )
        train_idx, test_idx, split_info = _session_split(y, sessions, test_size, min_sessions=4, reserve_train_sessions=2)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        fit_local, val_local, validation_sessions = _session_split(
            y_train, sessions[train_idx], 0.15, seed=43, min_sessions=3, reserve_train_sessions=2
        )
        split_name = "capture_session_holdout"

    if np.bincount(y_train[fit_local], minlength=len(class_names)).min() < 2:
        raise ValueError("Mỗi danh tính cần ít nhất 2 ảnh fit sau khi tách validation.")

    X_fit, X_val = X_train[fit_local], X_train[val_local]
    y_fit, y_val = y_train[fit_local], y_train[val_local]

    cnn, _ = build_cnn(len(class_names))
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=16, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
    ]
    cnn.fit(X_fit, y_fit, validation_data=(X_val, y_val), epochs=80, batch_size=16, callbacks=callbacks, verbose=1)
    extractor = keras.Model(cnn.input, cnn.get_layer("embedding").output)
    train_embeddings = extractor.predict(X_train, verbose=0)
    test_embeddings = extractor.predict(X_test, verbose=0)

    print("Đang trích xuất đặc trưng hình học 3 điểm (Mắt - Mũi - Miệng)...")
    geo_train = np.asarray([extract_face_geometry(img)[0] for img in X_train], dtype=np.float32)
    geo_test = np.asarray([extract_face_geometry(img)[0] for img in X_test], dtype=np.float32)

    # Ghép nối vector đặc trưng CNN (128-d) + Đặc trưng hình học (10-d) => 138-d
    train_fused = np.hstack([train_embeddings, geo_train])
    test_fused = np.hstack([test_embeddings, geo_test])

    print("Đang tự động dò tìm siêu tham số tối ưu cho SVM (GridSearchCV)...")
    param_grid = {
        "svm__C": [0.5, 1.0, 5.0, 10.0, 20.0, 50.0],
        "svm__gamma": ["scale", "auto", 0.05, 0.01, 0.005],
    }
    base_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", probability=True, random_state=42)),
    ])
    grid = GridSearchCV(base_pipeline, param_grid, cv=3, n_jobs=-1)
    grid.fit(train_fused, y_train)
    svm = grid.best_estimator_
    print(f"Tham số SVM tối ưu nhất: {grid.best_params_} (CV Score: {grid.best_score_:.2%})")
    probabilities = svm.predict_proba(test_fused)
    # Use the same probability-based class decision in evaluation and webcam.
    predictions = np.asarray(svm.classes_, dtype=int)[np.argmax(probabilities, axis=1)]

    model_path.parent.mkdir(parents=True, exist_ok=True)
    cnn_path = model_path.with_name(CNN_MODEL_PATH.name)
    extractor.save(cnn_path)
    bundle = {
        "svm": svm,
        "class_names": class_names,
        "image_size": IMAGE_SIZE,
        "preprocessing_version": "haar_1.08_6_70_margin_0.12_rgb_geometry_v2",
        "cnn_path": str(cnn_path.resolve()),
        "use_geometry": True,
    }
    with model_path.open("wb") as file:
        pickle.dump(bundle, file)

    results_path = model_path.with_name("model_results.json")
    evaluation = _evaluate(
        y_test, predictions, probabilities, class_names, paths[test_idx],
        sessions[test_idx], results_path.parent, model_path.stem,
    )
    results = {
        "dataset": str(dataset_dir.resolve()),
        "split": {
            "method": split_name,
            "requested_test_fraction": test_size,
            "actual_train_fraction": float(len(train_idx) / len(X)),
            "actual_test_fraction": float(len(test_idx) / len(X)),
            "train_samples": len(train_idx),
            "test_samples": len(test_idx),
            "sessions_by_class": split_info,
            "validation_sessions_by_class": validation_sessions,
        },
        "classes": class_names.tolist(),
        "cnn_test_accuracy": float(accuracy_score(y_test, np.argmax(cnn.predict(X_test, verbose=0), axis=1))),
        "svm": evaluation,
    }
    if baseline_model and Path(baseline_model).is_file():
        # Only load pickle files that you created yourself and trust.
        with Path(baseline_model).open("rb") as file:
            baseline = pickle.load(file)
        baseline_cnn_path = Path(baseline.get("cnn_path", CNN_MODEL_PATH))
        if not baseline_cnn_path.is_absolute():
            baseline_cnn_path = Path(baseline_model).resolve().parent / baseline_cnn_path
        if baseline_cnn_path.is_file() and list(baseline["class_names"]) == list(class_names):
            baseline_extractor = keras.models.load_model(baseline_cnn_path, compile=False, safe_mode=False)
            baseline_embeddings = baseline_extractor.predict(X_test, verbose=0)
            baseline_prob = baseline["svm"].predict_proba(baseline_embeddings)
            baseline_pred = np.asarray(baseline["svm"].classes_, dtype=int)[np.argmax(baseline_prob, axis=1)]
            results["baseline_comparison"] = {
                "baseline_model": str(Path(baseline_model).resolve()),
                "evaluation_note": "Baseline evaluated on the same session-held-out images; its training data may overlap this test set." if split_method == "session" else "Baseline tested on the same random-held-out images; its original training set may overlap.",
                "baseline": _evaluate(y_test, baseline_pred, baseline_prob, class_names, paths[test_idx], sessions[test_idx], results_path.parent, "baseline", save_mistakes=False),
                "new_model": {"accuracy": evaluation["accuracy"]},
                "accuracy_delta": float(evaluation["accuracy"] - accuracy_score(y_test, baseline_pred)),
            }
        else:
            results["baseline_comparison"] = {
                "baseline_model": str(Path(baseline_model).resolve()),
                "error": "Không tìm thấy CNN baseline hoặc danh sách lớp khác dataset mới; không thể so sánh.",
            }

    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SVM {split_name} accuracy: {evaluation['accuracy']:.2%}")
    print(f"Đã lưu mô hình: {model_path.resolve()} | CNN: {cnn_path.resolve()}")
    print(f"Đã lưu báo cáo: {results_path.resolve()} (không tạo thư mục ảnh nhận sai)")
