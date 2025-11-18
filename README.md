# Face App Attendance 📘

Ứng dụng điểm danh bằng nhận diện khuôn mặt, kết hợp kiểm tra chống giả mạo và nhận diện cảm xúc.

---

## 1. Yêu cầu hệ thống (Prerequisites)

Trước khi bắt đầu, đảm bảo máy tính đã cài đặt:

- **Python**: 3.8 – 3.10 (Khuyên dùng 3.10 cho TensorFlow)
- **Git**: Để clone mã nguồn
- **Git LFS**: Để tải các file model nặng (rất quan trọng)

---

## 2. Cài đặt chi tiết (Installation)

### Bước 1: Clone dự án

```bash
git clone https://github.com/KhoiBui16/Face_App_Attendance.git
cd Face_App_Attendance
```

### Bước 2: Tạo môi trường ảo (Virtual Environment)

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Bước 3: Cài đặt thư viện

```bash
pip install -r requirements.txt
```

> Lưu ý: Kiểm tra `requirements.txt` để xóa dòng thừa nếu có copy/paste lỗi.

### Bước 4: Chuẩn bị Models

Tạo thư mục `models/` ở thư mục gốc, sau đó thêm các file:

- `ResNet50_feature_extractor.keras` – Model trích xuất đặc trưng khuôn mặt
- `anti_spoof_model.h5` – Model chống giả mạo
- `emotion_model.h5` – Model nhận diện cảm xúc

> Nếu vừa clone từ Git và dùng Git LFS, chạy:

```bash
git lfs pull
```

để tải file về nếu chưa đầy đủ.

### Bước 5: Tạo cấu hình đăng nhập (Tùy chọn)

```bash
python generate_keys.py
```

> Lệnh này tạo file `config.yaml` chứa thông tin user Admin.

---

## 3. Chạy ứng dụng (Running the App)

```bash
streamlit run app.py
```

Trình duyệt sẽ tự động mở: [http://localhost:8501](http://localhost:8501)

---

## 4. Cấu trúc dự án (Project Structure)

```
Face_App_Attendance/
├── app.py                  # [MAIN] Giao diện chính
├── face_processing.py      # [CORE] Xử lý AI: load model, detect mặt, embedding
├── db.py                   # [DATABASE] Lưu/Xóa user, log CSV
├── generate_keys.py        # [UTIL] Mã hóa mật khẩu & tạo config.yaml
├── requirements.txt        # Thư viện cần thiết
├── models/                 # [DATA] File .keras, .h5
│   ├── ResNet50_feature_extractor.keras
│   ├── anti_spoof_model.h5
│   └── emotion_model.h5
├── face_db/                # [DATA] File .pkl chứa embedding người dùng
└── attendance_log.csv      # [LOG] Lưu lịch sử điểm danh
```

**Luồng hoạt động:**

- **Đăng ký:** app.py chụp ảnh → face_processing.py kiểm tra Spoof → tạo Embedding → db.py lưu vào `face_db/`
- **Điểm danh:** app.py chụp ảnh → face_processing.py tạo Embedding mới → so sánh Cosine Similarity → trả kết quả + cảm xúc → db.py ghi vào `attendance_log.csv`

---

## 5. Triển khai lên Web (Deploy)

### Bước 1: Chuẩn bị GitHub

- Đảm bảo code đã push lên GitHub với **Git LFS**.
- Chỉnh sửa `requirements.txt`:

```
streamlit
tensorflow-cpu
numpy
opencv-python-headless
mtcnn
scikit-learn
pandas
pytz
pyyaml
```

### Bước 2: Tạo `packages.txt` cho OpenCV

- Tạo file `packages.txt` ở thư mục gốc, thêm:

```
libgl1
```

### Bước 3: Deploy trên Streamlit Community Cloud

1. Truy cập [share.streamlit.io](https://share.streamlit.io)
2. Đăng nhập bằng GitHub
3. Chọn **New app** → chọn repo `Face_App_Attendance` → branch `main` → main file `app.py` → Deploy

**Lưu ý:**

- Nếu OOM (Out of Memory) do TensorFlow/ResNet50 → cân nhắc dùng model nhẹ hơn như MobileNetV2 hoặc deploy trên Hugging Face Spaces/Render
- Lần đầu deploy với Git LFS có thể tải chậm, kiên nhẫn chờ

---

## 6. Lưu ý thêm

- Luôn track **file lớn bằng LFS trước commit**
- Nếu commit cũ chứa file >100MB, cần **rewrite history** để push thành công
- Clone lại repo nếu dùng force-push history cũ
