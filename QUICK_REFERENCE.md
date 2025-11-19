# Quick Reference: Tính Năng Mới & Cách Sử Dụng

## 🎯 Session State Management

### Camera State

```python
# Camera tự động lưu trữ qua reruns
st.session_state.camera  # VideoCapture object hoặc None
st.session_state.captured_frame  # Frame cuối cùng được chụp
st.session_state.consecutive_match_count  # Đếm số frame khớp
```

### Embeddings Cache

```python
# Cache embeddings trong RAM thay vì đọc disk
st.session_state.embeddings_cache  # Dict[name -> embedding]
st.session_state.embedding_matrix  # numpy array cho cosine_similarity
st.session_state.embedding_names  # List tên users
```

**Cách làm mới cache:**

1. Click nút **"🔄 Làm mới Cache"** trong sidebar
2. Hoặc restart app

---

## ⚙️ Configuration (config.py)

### Environment Variables

```bash
# .env file (tạo file .env trong root folder)
MODEL_PATH=models/my_custom_model.keras
COSINE_THRESHOLD=0.65
CAMERA_INDEX=1
FRAME_SKIP=5
DEBUG=True
```

### Python Code

```python
from config import COSINE_THRESHOLD, PROCESS_EVERY_N_FRAMES

# Sử dụng trực tiếp
if similarity > COSINE_THRESHOLD:
    print("Match!")
```

---

## 🔒 Input Validation

### Đăng Ký User Mới

```python
# Validation tự động:
# - Tên: 2-50 ký tự, chỉ chữ cái và khoảng trắng (bao gồm tiếng Việt)
# - MSSV: 1-20 ký tự, chỉ chữ và số
# - Lớp: Tối đa 50 ký tự

# Ví dụ hợp lệ:
r_name = "Nguyễn Văn A"  # ✅
r_mssv = "SV001234"      # ✅
r_class = "CNTT K65"     # ✅

# Ví dụ KHÔNG hợp lệ:
r_name = "A"             # ❌ Quá ngắn (< 2 ký tự)
r_mssv = "SV@123"        # ❌ Có ký tự đặc biệt
r_name = "../../../etc"  # ❌ Blocked path traversal
```

---

## 📊 Performance Tips

### 1. Frame Skipping

```python
# Mặc định: Xử lý mỗi 3 frames
# Thay đổi trong config.py:
PROCESS_EVERY_N_FRAMES = 5  # Tăng lên nếu CPU yếu
```

### 2. Cache Management

```python
# Khi nào cần làm mới cache:
# ✅ Sau khi đăng ký user mới (tự động)
# ✅ Sau khi xóa user
# ✅ Khi số user hiển thị sai
# ✅ Khi recognition không chính xác

# Cách làm: Click "🔄 Làm mới Cache"
```

### 3. Database Optimization

```python
# LRU Cache tự động cho 128 users gần nhất
# Nếu có > 128 users, tăng trong db.py:
@lru_cache(maxsize=256)  # Hoặc 512, 1024
def get_user_info(name):
    ...
```

---

## 🐛 Debugging

### Log File Location

```
face_recognition_app/
├── face_recognition.log  # ← Errors được log tại đây
├── attendance_log.csv.lock  # File lock (tự động)
└── ...
```

### Xem Logs

```bash
# Windows PowerShell
Get-Content face_recognition.log -Tail 50

# Hoặc mở bằng text editor
```

### Common Issues

#### Camera không mở

```python
# Kiểm tra CAMERA_INDEX
# Mặc định: 0 (camera mặc định)
# Nếu có nhiều camera, thử 1, 2, 3...

# config.py
CAMERA_INDEX = 1  # Thử các giá trị khác
```

#### Recognition không chính xác

```python
# Giảm threshold trong config.py
COSINE_THRESHOLD = 0.55  # Mặc định: 0.6
# Giá trị thấp hơn = dễ match hơn (nhưng dễ false positive)
```

#### Out of Memory

```python
# Giảm cache size
LRU_CACHE_SIZE = 64  # Mặc định: 128

# Tăng frame skip
PROCESS_EVERY_N_FRAMES = 5  # Mặc định: 3
```

---

## 🔧 API Changes

### Không có breaking changes!

Tất cả functions giữ nguyên signature:

```python
# face_processing.py
register_face(name, mssv, class_name, image_bytes)  # ✅ Như cũ
verify_face(image_bytes, input_class_name, ...)     # ✅ Như cũ
detect_and_align(image_bytes, image_cv2)            # ✅ Như cũ

# db.py
save_user_data(name, mssv, class_name, embedding)   # ✅ Như cũ
load_embeddings()                                    # ✅ Như cũ
log_attendance(...)                                  # ✅ Như cũ
```

**Thay đổi nội bộ:**

- ✅ Model loading: Singleton pattern
- ✅ CSV writes: FileLock
- ✅ DataFrame: parse_dates + inplace ops
- ✅ User info: LRU cached

---

## 📈 Monitoring

### Sidebar Info

```
👥 15 người đã đăng ký  # ← Real-time count
```

### System Status

```python
# Kiểm tra cache
if st.session_state.embeddings_cache:
    print(f"Cache loaded: {len(st.session_state.embeddings_cache)} users")
else:
    print("Cache empty - will load on first use")
```

---

## 🎓 Best Practices

### 1. Đăng Ký Users

- ✅ Đảm bảo ánh sáng tốt
- ✅ Mặt nhìn thẳng vào camera
- ✅ Không đeo khẩu trang/kính đen
- ✅ Đăng ký nhiều góc độ nếu cần

### 2. Check-in/Check-out

- ✅ Giữ yên 2-3 giây để auto-capture
- ✅ Một người một lần
- ✅ Đợi "DONE!" xuất hiện

### 3. Maintenance

- ✅ Xóa cache 1 tuần/lần
- ✅ Backup `face_db/` folder thường xuyên
- ✅ Export attendance logs hàng tháng

---

## 💡 Advanced Features

### Custom Threshold Per User

```python
# Trong face_processing.py, thêm:
USER_THRESHOLDS = {
    "Nguyễn Văn A": 0.7,  # Stricter
    "Trần Thị B": 0.5,    # More lenient
}

# Trong recognize_from_crop():
threshold = USER_THRESHOLDS.get(name, COSINE_THRESHOLD)
```

### Backup Automation

```bash
# Windows Task Scheduler hoặc cron job
# Backup script (PowerShell)
Copy-Item face_db\ -Destination "backup_$(Get-Date -Format 'yyyyMMdd')" -Recurse
```

---

## 📞 Support Checklist

Trước khi báo lỗi, kiểm tra:

- [ ] Đã click "🔄 Làm mới Cache"?
- [ ] Đã xem `face_recognition.log`?
- [ ] Camera hoạt động bình thường?
- [ ] Đủ ánh sáng?
- [ ] Khuôn mặt rõ ràng, không bị che?
- [ ] Dependencies đã cài đầy đủ? (`pip install -r requirements.txt`)
- [ ] Python version: 3.8+?
- [ ] TensorFlow version: 2.x?

---

**Version:** 2.0 (19/11/2025)  
**Maintained by:** AI Optimization Team
