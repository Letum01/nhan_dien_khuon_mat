
## 📁 Cấu Trúc Dự Án

```
python/
├── dataset/                     # Dữ liệu ảnh khuôn mặt gốc (phân theo thư mục từng người)
│   ├── Hoai_Phuong/
│   ├── Hoang_Anh/
│   ├── Huu_Thang/
│   ├── Minh_Hy/
│   └── Thanh_Trinh/
├── face_recognition_app/        # Package mã nguồn ứng dụng lõi
│   ├── __init__.py              # Khởi tạo package
│   ├── config.py                # Cấu hình tập trung (kích thước ảnh, tham số Haar, ngưỡng tin cậy)
│   ├── cnn.py                   # Kiến trúc CNN tự học (kèm Data Augmentation, BatchNorm, Dropout)
│   ├── dataset.py               # Xử lý ảnh: Haar Cascade, Face Alignment, Cân bằng sáng CLAHE
│   ├── geometry.py              # Trích xuất 4 điểm mốc (Mắt-Mũi-Miệng), tính 10 tỷ lệ & vẽ wireframe
│   ├── recognition.py           # Vòng lặp webcam đa đối tượng, Temporal Voting, suy luận Batch
│   └── training.py              # Huấn luyện CNN, Feature Fusion 138-d, GridSearchCV SVM
├── augment_dataset.py           # Script tự động nhân bản dữ liệu ảnh (xoay, chỉnh sáng)
├── capture_faces.py             # Script chụp ảnh thu thập khuôn mặt từ webcam theo phiên
├── model.pkl                    # Model bundle lưu SVM, danh sách lớp, siêu tham số
├── model_cnn.keras              # Trọng số mạng CNN trích xuất đặc trưng 128-d
├── model_results.json           # Báo cáo đánh giá chi tiết (Classification report, Confusion matrix)
├── organize_sessions.py         # Tiện ích tự động gom nhóm ảnh theo ngày chụp từ EXIF/mtime
├── requirements.txt             # Danh sách thư viện phụ thuộc
├── train_model.py               # Điểm chạy CLI huấn luyện mô hình
├── webcam_recognition.py        # Điểm chạy CLI nhận diện webcam trực tiếp
└── README.md                    # Tài liệu hướng dẫn dự án
```

---

## 🛠️ Cài Đặt Môi Trường

Dự án yêu cầu **Python 3.10 trở lên** (khuyến nghị Python 3.11).

---

## 🚀 Hướng Dẫn Sử Dụng Nhanh

### 1. Thu thập ảnh khuôn mặt mới (Nếu cần thêm người)
Chạy script chụp ảnh tự động qua webcam:
```powershell
python capture_faces.py --person <Ten_Nguoi> --session session_1 --count 100
```
- Phím **S**: Bắt đầu chụp.
- Phím **Q**: Thoát.
- Ảnh sẽ tự động được lưu vào `dataset/<Ten_Nguoi>/<session_1>/`.

### 2. Nhân bản dữ liệu (Data Expansion)
Tạo thêm các biến thể tự nhiên giúp mô hình học sâu hơn mà không cần chụp thêm:
```powershell
python augment_dataset.py --source dataset --output dataset_augmented
```

### 3. Huấn luyện mô hình (Training)
Huấn luyện mô hình CNN kết hợp Đặc trưng hình học và SVM:

- **Huấn luyện trên dataset hiện tại**:
  ```powershell
  python train_model.py
  ```
  *(Hệ thống mặc định sử dụng chế độ `--split-method auto`, tự động kích hoạt Data Augmentation và Feature Fusion).*

- **Huấn luyện trên dataset đã nhân bản**:
  ```powershell
  python train_model.py --dataset dataset_augmented
  ```

- **Huấn luyện chia theo phiên độc lập (nếu có từ 2-4 phiên chụp trở lên)**:
  ```powershell
  python train_model.py --split-method session
  ```

Sau khi huấn luyện, kết quả đánh giá chi tiết (`precision`, `recall`, `f1-score`) sẽ được lưu vào `model_results.json`.

### 4. Nhận diện qua Webcam (Inference)
Chạy nhận diện thời gian thực:
```powershell
python webcam_recognition.py
```
- **Khung hình chính**: Nhận diện đồng thời các khuôn mặt trong khung hình, hiển thị tên, độ tin cậy và 4 chấm sinh trắc học:
  - 🟢 Xanh lá: Mắt trái, Mắt phải
  - 🔴 Đỏ: Đỉnh mũi
  - 🔵 Xanh dương: Tâm miệng
- **Cửa sổ phụ (`PREVIEW_WINDOW`)**: Hiển thị ảnh khuôn mặt chính sau khi căn chỉnh và lưới tam giác sinh trắc học.
- Phím **Q**: Thoát ứng dụng.

---

## 🔬 Chi Tiết Kỹ Thuật

### 1. Vector Hình Học Sinh Trắc Học 10 Chiều (`geometry.py`)
Lấy khoảng cách hai mắt $D_{\text{eyes}} = ||P_{\text{eye\_right}} - P_{\text{eye\_left}}||_2$ làm chuẩn quy đổi:
- $f_0, f_1$: Khoảng cách từ từng mắt đến mũi $/ D_{\text{eyes}}$
- $f_2$: Chiều dài sống mũi (Trung điểm mắt đến Mũi) $/ D_{\text{eyes}}$
- $f_3$: Khoảng cách Mũi - Miệng $/ D_{\text{eyes}}$
- $f_4$: Trục dọc khuôn mặt (Trung điểm mắt đến Miệng) $/ D_{\text{eyes}}$
- $f_5, f_6$: Khoảng cách từ từng mắt đến Miệng $/ D_{\text{eyes}}$
- $f_7$: Tỷ lệ sống mũi / khoảng cách mũi-miệng ($f_2 / f_3$)
- $f_8$: Chỉ số đối xứng tam giác mặt
- $f_9$: Diện tích tam giác sinh trắc học Mắt trái - Mắt phải - Mũi $/ D_{\text{eyes}}^2$

### 2. Mạng CNN Trích Xuất Đặc Trưng (`cnn.py`)
- Kích thước đầu vào: $(128, 128, 3)$.
- Khối tích chập: 3 tầng `Conv2D` $(32 \rightarrow 64 \rightarrow 128)$, mỗi tầng đi kèm `BatchNormalization` và kích hoạt `ReLU`.
- Regularization: `GlobalAveragePooling2D` kết hợp `Dropout(0.3)`.
- Lớp biểu diễn: `Dense(128, activation="relu", name="embedding")`.

### 3. Bộ Phân Loại Kết Hợp (`training.py`)
- Vector hợp nhất: $V_{\text{fused}} = [V_{\text{cnn}} \parallel V_{\text{geo}}] \in \mathbb{R}^{138}$.
- Pipeline: `StandardScaler()` $\rightarrow$ `GridSearchCV` $\rightarrow$ `SVC(kernel="rbf", probability=True)`.

---

## 💡 Hướng Dẫn Phát Triển Tiếp Theo

1. **Bổ sung tính năng Liveness Detection (Chống giả mạo)**:
   - Sử dụng tỷ lệ mở mắt (Eye Aspect Ratio - EAR) để phát hiện chớp mắt tự nhiên, ngăn chặn việc dùng ảnh tĩnh/màn hình điện thoại để đánh lừa camera.
2. **Triển khai Web API / Mobile App**:
   - Tách phần suy luận thành API sử dụng **FastAPI** (`uvicorn`) để các ứng dụng web hoặc ứng dụng di động có thể gửi ảnh lên nhận diện dễ dàng.
3. **Thêm cơ chế từ chối người lạ (Unknown Rejection)**:
   - Dựa trên ngưỡng khoảng cách cực đại tới tâm cụm (Centroid Distance) hoặc ngưỡng xác suất không chắc chắn trong `config.py` (`CONFIDENCE_THRESHOLD = 0.70`).


Cài đặt toàn bộ thư viện cần thiết:
```powershell
pip install -r requirements.txt
```
