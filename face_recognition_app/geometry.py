"""Trích xuất điểm mốc hình học (Mắt, Mũi, Miệng) và tính toán khoảng cách sinh trắc học bất biến."""
import cv2
import numpy as np


def extract_landmarks(rgb_crop: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Xác định tọa độ 4 điểm mốc sinh trắc học: (Mắt trái, Mắt phải, Đỉnh mũi, Tâm miệng).
    
    Đầu vào là ảnh mặt đã cắt kích thước (128, 128, 3), float32 [0..1] hoặc uint8 [0..255].
    """
    if rgb_crop.dtype == np.float32 or rgb_crop.max() <= 1.0:
        gray = cv2.cvtColor((rgb_crop * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = cv2.cvtColor(rgb_crop.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    h, w = gray.shape

    # 1. Vùng Mắt trái (theo hướng nhìn vào ảnh): x: [16%, 46%], y: [22%, 46%]
    le_y1, le_y2 = int(0.22 * h), int(0.46 * h)
    le_x1, le_x2 = int(0.16 * w), int(0.46 * w)
    le_roi = gray[le_y1:le_y2, le_x1:le_x2]
    le_blurred = cv2.GaussianBlur(le_roi, (7, 7), 0)
    _, _, min_loc, _ = cv2.minMaxLoc(le_blurred)
    left_eye = (float(le_x1 + min_loc[0]), float(le_y1 + min_loc[1]))

    # 2. Vùng Mắt phải: x: [54%, 84%], y: [22%, 46%]
    re_y1, re_y2 = int(0.22 * h), int(0.46 * h)
    re_x1, re_x2 = int(0.54 * w), int(0.84 * w)
    re_roi = gray[re_y1:re_y2, re_x1:re_x2]
    re_blurred = cv2.GaussianBlur(re_roi, (7, 7), 0)
    _, _, re_min_loc, _ = cv2.minMaxLoc(re_blurred)
    right_eye = (float(re_x1 + re_min_loc[0]), float(re_y1 + re_min_loc[1]))

    # 3. Vùng Đỉnh mũi: x: [36%, 64%], y: [46%, 68%]
    n_y1, n_y2 = int(0.46 * h), int(0.68 * h)
    n_x1, n_x2 = int(0.36 * w), int(0.64 * w)
    n_roi = gray[n_y1:n_y2, n_x1:n_x2]
    n_blurred = cv2.GaussianBlur(n_roi, (7, 7), 0)
    _, _, _, max_loc = cv2.minMaxLoc(n_blurred)
    nose = (float(n_x1 + max_loc[0]), float(n_y1 + max_loc[1]))

    # 4. Vùng Tâm miệng: x: [28%, 72%], y: [66%, 88%]
    m_y1, m_y2 = int(0.66 * h), int(0.88 * h)
    m_x1, m_x2 = int(0.28 * w), int(0.72 * w)
    m_roi = gray[m_y1:m_y2, m_x1:m_x2]
    m_blurred = cv2.GaussianBlur(m_roi, (9, 9), 0)
    _, _, m_min_loc, _ = cv2.minMaxLoc(m_blurred)
    mouth = (float(m_x1 + m_min_loc[0]), float(m_y1 + m_min_loc[1]))

    return left_eye, right_eye, nose, mouth


def compute_geometric_features(
    left_eye: tuple[float, float],
    right_eye: tuple[float, float],
    nose: tuple[float, float],
    mouth: tuple[float, float],
) -> np.ndarray:
    """Tính toán 10 tỷ lệ khoảng cách hình học bất biến theo tỷ lệ phóng to/thu nhỏ."""
    le = np.asarray(left_eye, dtype=np.float32)
    re = np.asarray(right_eye, dtype=np.float32)
    n = np.asarray(nose, dtype=np.float32)
    m = np.asarray(mouth, dtype=np.float32)

    # Khoảng cách cơ sở chuẩn hóa: Khoảng cách giữa 2 mắt
    d_eyes = float(np.linalg.norm(re - le))
    if d_eyes < 1.0:
        d_eyes = 1.0

    # Trung điểm giữa hai mắt
    mid_eyes = 0.5 * (le + re)

    # 1. Các khoảng cách chuẩn hóa theo D_eyes
    f0 = float(np.linalg.norm(le - n) / d_eyes)       # Mắt trái đến mũi
    f1 = float(np.linalg.norm(re - n) / d_eyes)       # Mắt phải đến mũi
    f2 = float(np.linalg.norm(mid_eyes - n) / d_eyes) # Chiều dài sống mũi
    f3 = float(np.linalg.norm(n - m) / d_eyes)        # Khoảng cách Mũi - Miệng
    f4 = float(np.linalg.norm(mid_eyes - m) / d_eyes) # Trục dọc từ mắt đến miệng
    f5 = float(np.linalg.norm(le - m) / d_eyes)       # Mắt trái đến miệng
    f6 = float(np.linalg.norm(re - m) / d_eyes)       # Mắt phải đến miệng

    # 2. Tỷ lệ tương quan
    f7 = float(f2 / (f3 + 1e-6))                      # Tỷ lệ sống mũi / mũi-miệng
    f8 = float((f0 + f1) / (2.0 * f4 + 1e-6))         # Tỷ lệ cân bằng tam giác mặt

    # 3. Diện tích tam giác Mắt trái - Mắt phải - Mũi
    triangle_area = 0.5 * abs(
        le[0] * (re[1] - n[1]) + re[0] * (n[1] - le[1]) + n[0] * (le[1] - re[1])
    )
    f9 = float(triangle_area / (d_eyes ** 2))

    return np.asarray([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9], dtype=np.float32)


def extract_face_geometry(rgb_crop: np.ndarray) -> tuple[np.ndarray, dict[str, tuple[float, float]]]:
    """Hàm bao trích xuất cả vector hình học và từ điển điểm mốc."""
    left_eye, right_eye, nose, mouth = extract_landmarks(rgb_crop)
    geo_vector = compute_geometric_features(left_eye, right_eye, nose, mouth)
    landmarks = {
        "left_eye": left_eye,
        "right_eye": right_eye,
        "nose": nose,
        "mouth": mouth,
    }
    return geo_vector, landmarks


def align_face_horizontal(rgb_crop: np.ndarray, left_eye: tuple[float, float], right_eye: tuple[float, float]) -> np.ndarray:
    """Xoay phẳng khuôn mặt sao cho đường nối 2 mắt nằm ngang tuyệt đối (0 độ)."""
    dx = float(right_eye[0] - left_eye[0])
    dy = float(right_eye[1] - left_eye[1])
    if dx < 10.0:
        return rgb_crop
    angle = float(np.degrees(np.arctan2(dy, dx)))

    # Chỉ xoay nếu góc lệch hợp lý (0.5 đến 12.0 độ).
    # Các góc quá lớn thường do heuristic bắt nhầm tóc/bóng tối thành mắt, xoay sẽ làm hỏng ảnh.
    if abs(angle) < 0.5 or abs(angle) > 12.0:
        return rgb_crop

    h, w = rgb_crop.shape[:2]
    eye_center = (float((left_eye[0] + right_eye[0]) / 2.0), float((left_eye[1] + right_eye[1]) / 2.0))
    rot_mat = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    aligned = cv2.warpAffine(rgb_crop, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return aligned


def draw_facial_geometry(bgr_image: np.ndarray, landmarks: dict[str, tuple[float, float]]) -> np.ndarray:
    """Vẽ minh họa 4 điểm mốc và các đường nối hình học sinh trắc lên ảnh."""
    annotated = bgr_image.copy()
    le = tuple(map(int, landmarks["left_eye"]))
    re = tuple(map(int, landmarks["right_eye"]))
    n = tuple(map(int, landmarks["nose"]))
    m = tuple(map(int, landmarks["mouth"]))

    # Các đường nối tam giác và trục nhân trắc
    cv2.line(annotated, le, re, (255, 200, 0), 1, cv2.LINE_AA)
    cv2.line(annotated, le, n, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.line(annotated, re, n, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.line(annotated, n, m, (0, 165, 255), 1, cv2.LINE_AA)

    mid_eyes = ((le[0] + re[0]) // 2, (le[1] + re[1]) // 2)
    cv2.line(annotated, mid_eyes, n, (255, 100, 255), 1, cv2.LINE_AA)

    # Điểm mốc
    cv2.circle(annotated, le, 3, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.circle(annotated, re, 3, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.circle(annotated, n, 3, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(annotated, m, 3, (255, 0, 0), -1, cv2.LINE_AA)

    return annotated
