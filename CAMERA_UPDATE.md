# Camera Visualization Update

## Vấn đề đã fix

### 1. Camera Real-time Preview trong Streamlit

**Vấn đề ban đầu:**

- Camera sử dụng `cv2.imshow()` mở cửa sổ OpenCV riêng biệt
- Người dùng không nhìn thấy bounding box, score, ID trong app Streamlit
- Không thể xem real-time để điều chỉnh góc độ khuôn mặt

**Giải pháp:**

- ✅ Thay `cv2.imshow()` bằng `st.empty().image()` cho live preview
- ✅ Thêm nút "Bật Camera" và "Dừng Camera" với session state control
- ✅ Hiển thị bounding box, tên, score trực tiếp trong Streamlit app
- ✅ Thêm status placeholder để hiển thị trạng thái nhận diện
- ✅ Countdown timer hiển thị trên frame khi giữ yên mặt

**Cơ chế hoạt động:**

```python
# Tạo placeholder cho live preview
FRAME_WINDOW = st.empty()
status_placeholder = st.empty()

# Loop camera với điều khiển start/stop
while cap.isOpened() and not st.session_state.stop_camera:
    ret, frame = cap.read()

    # Xử lý nhận diện + vẽ bounding box
    debug_frame = frame.copy()
    cv2.rectangle(debug_frame, (x, y), (x+w, y+h), color, 3)
    cv2.putText(debug_frame, label, (x, y-5), ...)

    # Hiển thị trong Streamlit (không phải OpenCV window)
    display_frame = cv2.cvtColor(debug_frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(display_frame, channels="RGB", use_container_width=True)

    # Status text
    status_placeholder.info(f"🎯 Đang nhận diện: **{name}** (Còn {remain}s)")
```

**Features mới:**

- 🔴 **Start/Stop buttons**: Kiểm soát camera từ UI
- 📹 **Live preview**: Xem real-time trong Streamlit app
- 🎯 **Bounding boxes**: Khung màu xanh (nhận diện được) / đỏ (Unknown)
- 📊 **Score display**: Hiển thị điểm tương đồng cosine
- ⏱️ **Countdown timer**: Đếm ngược khi giữ yên mặt
- ⚠️ **Status messages**: Thông báo điều chỉnh góc độ khi cần

### 2. Tích hợp config.py toàn bộ codebase

**Files đã cập nhật:**

#### `face_processing.py`

```python
import config

# Đã thay thế:
COSINE_THRESHOLD = 0.6          → config.COSINE_THRESHOLD
IMG_SIZE = (224, 224)           → config.IMG_SIZE
MODEL_PATH = "models/..."       → config.MODEL_PATH
EMBEDDING_LAYER_NAME = "..."    → config.EMBEDDING_LAYER_NAME
SPOOF_IMG_SIZE = (224, 224)     → config.SPOOF_IMG_SIZE
EMOTION_IMG_SIZE = (224, 224)   → config.EMOTION_IMG_SIZE
```

#### `app.py`

```python
import config

# Đã sử dụng:
config.CAMERA_INDEX                 # VideoCapture index
config.PROCESS_EVERY_N_FRAMES       # Frame skipping
config.CONSECUTIVE_MATCH_THRESHOLD  # Auto-capture threshold
config.DETECTION_RESIZE_WIDTH       # Detection scale
config.FACE_MARGIN                  # Face crop margin
```

**Lợi ích:**

- ✅ Dễ thay đổi cấu hình qua environment variables
- ✅ Không cần sửa code khi chỉnh threshold
- ✅ Centralized configuration management
- ✅ Hỗ trợ deployment với .env file

## Testing

### Test Camera Preview

1. Chạy app: `streamlit run app.py`
2. Chọn "Live Camera (Real-time)"
3. Click "🔴 Bật Camera Real-time"
4. Kiểm tra:
   - ✅ Live preview hiển thị trong Streamlit (không có cửa sổ OpenCV)
   - ✅ Bounding box màu xanh khi nhận diện được
   - ✅ Label hiển thị tên + score
   - ✅ Countdown timer khi giữ yên
   - ✅ Nút "⏹️ Dừng Camera" hoạt động
   - ✅ Auto-capture sau 3 giây

### Test Config Integration

```python
# Kiểm tra config được load đúng
import config
print(config.COSINE_THRESHOLD)      # 0.6
print(config.PROCESS_EVERY_N_FRAMES) # 3
print(config.CAMERA_INDEX)          # 0
```

## Technical Details

### Session States Added

```python
st.session_state.camera_running = False   # Camera loop control
st.session_state.stop_camera = False      # Stop flag
```

### Camera Loop Changes

**Before:**

```python
while cap.isOpened():
    ...
    cv2.imshow("Smart Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

**After:**

```python
FRAME_WINDOW = st.empty()
while cap.isOpened() and not st.session_state.stop_camera:
    ...
    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                       channels="RGB",
                       use_container_width=True)
    time.sleep(0.03)  # Non-blocking delay
```

### Cleanup

```python
finally:
    if st.session_state.camera is not None:
        st.session_state.camera.release()
        st.session_state.camera = None
    st.session_state.camera_running = False
    status_placeholder.empty()
```

## Performance Impact

- **Frame rate**: ~30 FPS với `time.sleep(0.03)`
- **Frame skipping**: Process mỗi 3 frames (configurable via `config.PROCESS_EVERY_N_FRAMES`)
- **Memory**: Streamlit caching giữ nguyên, không tăng thêm
- **UI responsiveness**: Nút Stop hoạt động ngay lập tức

## Migration Notes

Nếu đã có code cũ sử dụng hardcoded values:

1. Import config: `import config`
2. Replace:
   - `0.6` → `config.COSINE_THRESHOLD`
   - `(224, 224)` → `config.IMG_SIZE`
   - `640` (detection width) → `config.DETECTION_RESIZE_WIDTH`
   - `0` (camera index) → `config.CAMERA_INDEX`
   - `3` (frame skip) → `config.PROCESS_EVERY_N_FRAMES`

## Environment Variables

Có thể override config qua `.env` file:

```bash
COSINE_THRESHOLD=0.65
CAMERA_INDEX=1
FRAME_SKIP=5
MATCH_THRESHOLD=5
DEBUG=True
```
