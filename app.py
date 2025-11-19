import streamlit as st
import pandas as pd
import face_processing
import db  # Import database functions
import altair as alt
import time
import cv2
import io
import numpy as np
from datetime import datetime, timedelta
from PIL import Image
import config  # Import configuration

# --- INITIALIZE SESSION STATE ---
if "camera" not in st.session_state:
    st.session_state.camera = None
if "captured_frame" not in st.session_state:
    st.session_state.captured_frame = None
if "consecutive_match_count" not in st.session_state:
    st.session_state.consecutive_match_count = 0
if "target_name_prev" not in st.session_state:
    st.session_state.target_name_prev = None
if "selected_user" not in st.session_state:
    st.session_state.selected_user = "-- Chọn --"
if "embeddings_cache" not in st.session_state:
    st.session_state.embeddings_cache = None
if "embedding_matrix" not in st.session_state:
    st.session_state.embedding_matrix = None
if "embedding_names" not in st.session_state:
    st.session_state.embedding_names = None
if "camera_running" not in st.session_state:
    st.session_state.camera_running = False
if "stop_camera" not in st.session_state:
    st.session_state.stop_camera = False

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    layout="wide", page_title="Hệ thống Quản trị Nhân sự", page_icon="🏢"
)

# --- CSS NÂNG CAO ---
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
    body { font-family: 'Roboto', sans-serif; background-color: #f8f9fa; }
    
    [data-testid="stSidebar"] { background-color: #2c3e50; }
    [data-testid="stSidebar"] * { color: #ecf0f1 !important; }

    /* Card Style */
    .user-list-card {
        background-color: white; border-radius: 10px; padding: 15px; margin-bottom: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05); border: 1px solid #e9ecef; text-align: center;
    }
    .user-list-card:hover {
        transform: translateY(-3px); box-shadow: 0 5px 15px rgba(0,0,0,0.1); border-color: #3498db;
    }
    .ul-avatar {
        font-size: 30px; margin-bottom: 10px; background: #f1f3f5; width: 50px; height: 50px;
        border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 10px auto;
    }
    .ul-name { font-weight: bold; color: #2c3e50; font-size: 16px; margin-bottom: 2px; }
    .ul-mssv { font-size: 13px; color: #7f8c8d; font-weight: 500; }
    .ul-class { 
        font-size: 12px; color: white; background-color: #3498db; 
        padding: 2px 8px; border-radius: 10px; display: inline-block; margin-top: 5px;
    }

    /* Profile Header */
    .profile-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px; border-radius: 15px; color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin-bottom: 20px; display: flex; align-items: center;
    }
    .avatar-circle {
        width: 80px; height: 80px; background-color: rgba(255,255,255,0.2);
        border-radius: 50%; display: flex; align-items: center; justify-content: center;
        font-size: 40px; margin-right: 20px; border: 2px solid rgba(255,255,255,0.5);
    }
    .profile-info h2 { margin: 0; font-size: 28px; font-weight: 700; }
    .profile-info p { margin: 5px 0 0 0; opacity: 0.9; font-size: 16px; }

    /* Metric Box Enhanced */
    .metric-container {
        background-color: white; padding: 20px; border-radius: 12px;
        border-left: 5px solid #3498db;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    .metric-label { font-size: 13px; color: #7f8c8d; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 26px; font-weight: 700; color: #2c3e50; margin-top: 5px; }
    .metric-delta { font-size: 12px; color: #27ae60; font-weight: 600; }
    
    /* Metric Box Simple (Profile) */
    .metric-box {
        background: white; padding: 15px; border-radius: 10px;
        border: 1px solid #e9ecef; text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
    .metric-value-small { font-size: 24px; font-weight: bold; color: #2c3e50; }
    .metric-label-small { font-size: 14px; color: #7f8c8d; }

    /* Button Style */
    .stButton > button { border-radius: 8px; font-weight: 600; border: none; transition: all 0.2s; }
    .stButton > button:hover { transform: scale(1.02); }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Hệ thống Quản trị Nhân sự AI 🚀")

with st.spinner("Đang khởi động hệ thống..."):
    try:
        face_processing.load_models()
    except:
        st.stop()

app_mode = st.sidebar.radio(
    "MENU ĐIỀU KHIỂN",
    ["🏠 Điểm danh (Camera)", "📊 Dashboard Tổng quan", "👥 Quản lý & Hồ sơ Nhân viên"],
)

# Cache control in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Hệ thống")
if st.sidebar.button("🔄 Làm mới Cache"):
    st.session_state.embeddings_cache = None
    st.session_state.embedding_matrix = None
    st.session_state.embedding_names = None
    st.cache_data.clear()
    st.cache_resource.clear()
    st.sidebar.success("✅ Đã xóa cache!")
    st.rerun()

# Display system status - Count from actual database, not cache
embeddings_count = db.count_registered_users()
st.sidebar.info(f"👥 **{embeddings_count}** người đã đăng ký")

# # ================= 1. ĐIỂM DANH =================
# if app_mode == "🏠 Điểm danh (Camera)":
#     st.header("Camera Điểm danh 📸")

#     # [MỚI] Thêm lựa chọn chế độ
#     attendance_mode = st.radio(
#         "Chọn chế độ:", ["Check-in (Vào ca)", "Check-out (Tan ca)"], horizontal=True
#     )

#     c1, c2 = st.columns([1.5, 1])
#     with c1:
#         st.info("Hệ thống hỗ trợ nhận diện nhiều người cùng lúc.")
#         current_class = st.text_input(
#             "📚 Nhập Lớp/Môn học hiện tại:",
#             placeholder="Ví dụ: Lập trình Python - Sáng T2",
#         )
#         img_buffer = st.camera_input("Live Feed")
#     with c2:
#         st.subheader("Kết quả Real-time")
#         if img_buffer:
#             with st.spinner("Đang phân tích..."):
#                 # Xác định action dựa trên radio button
#                 action_type = (
#                     "Check-in" if "Check-in" in attendance_mode else "Check-out"
#                 )

#                 # Truyền thêm action_type vào hàm verify
#                 names_str, img_out, emotion, score, action_str, should_reload = (
#                     face_processing.verify_face(img_buffer, current_class, action_type)
#                 )
#                 st.image(img_out, channels="RGB", width="stretch")

#                 if "Giả mạo" in names_str:
#                     st.error("⚠️ CẢNH BÁO: PHÁT HIỆN GIẢ MẠO!")
#                 elif names_str == "NGUOI LA":
#                     st.warning("🚫 Người lạ. Không tìm thấy dữ liệu.")
#                 elif names_str != "Không tìm thấy":
#                     st.success(f"✅ **Đã nhận diện:** {names_str}")

#                     # Hiển thị thông báo trạng thái rõ ràng
#                     if "thành công" in action_str:
#                         st.success(f"🎉 {action_str}")
#                     else:
#                         st.info(f"ℹ️ {action_str}")

#                     if current_class:
#                         st.caption(f"📌 Ghi nhận tại lớp: {current_class}")
#                     if emotion != "Multiple" and "Wrong" not in emotion:
#                         st.markdown(f"**Cảm xúc:** {emotion}")


# ================= 1. ĐIỂM DANH =================
if app_mode == "🏠 Điểm danh (Camera)":
    st.header("Camera Điểm danh 📸")

    col_cam_mode, col_action = st.columns(2)
    with col_cam_mode:
        camera_mode = st.radio(
            "Chọn loại Camera:",
            ["📷 Camera Cơ bản (Web/Mobile)", "🎥 Camera Real-time (OpenCV Window)"],
            horizontal=True,
        )
    with col_action:
        attendance_action = st.radio(
            "Hành động:", ["Check-in", "Check-out"], horizontal=True
        )

    c1, c2 = st.columns([1.5, 1])
    with c1:
        current_class = st.text_input(
            "📚 Nhập Lớp/Môn học:", placeholder="Ví dụ: Lập trình Python"
        )

    # ---------------------------------------------------------
    # CHẾ ĐỘ 1: CAMERA CƠ BẢN (Dùng st.camera_input)
    # ---------------------------------------------------------
    if "Cơ bản" in camera_mode:
        with c1:
            # Validate class name before allowing camera
            if not current_class or current_class.strip() == "":
                st.warning("⚠️ Vui lòng nhập Lớp/Môn học trước khi chụp ảnh!")
                img_buffer = None
            else:
                img_buffer = st.camera_input("Chụp ảnh để điểm danh")

        if img_buffer:
            with c2:
                st.subheader("Kết quả")
                with st.spinner("Đang xử lý..."):
                    names_str, img_out, emotion, score, action_str, _ = (
                        face_processing.verify_face(
                            img_buffer,
                            current_class,
                            attendance_action,
                            enable_logging=True,
                        )
                    )

                    st.image(img_out, channels="RGB", width="stretch")

                    if "Giả mạo" in names_str:
                        st.error("⚠️ CẢNH BÁO GIẢ MẠO!")
                    elif names_str == "NGUOI LA":
                        st.warning("🚫 Không tìm thấy dữ liệu.")
                    else:
                        st.success(f"✅ **{names_str}**")
                        if "thành công" in action_str:
                            st.balloons()
                            st.success(f"🎉 {action_str}")
                            # Note: Basic camera mode auto-clears when user takes new photo
                        else:
                            st.info(f"ℹ️ {action_str}")
                        st.caption(
                            f"Độ chính xác: {score*100:.1f}% | Cảm xúc: {emotion}"
                        )

    # ---------------------------------------------------------
    # CHẾ ĐỘ 2: STREAMLIT REAL-TIME (LIVE PREVIEW WITH BOUNDING BOX)
    # ---------------------------------------------------------
    else:
        with c1:
            st.info(
                "💡 Camera sẽ hiển thị live preview. Giữ yên khuôn mặt trong 2-3 giây để tự động điểm danh."
            )

            # Live preview placeholder - ĐẶT Ở TRÊN ĐẦU
            FRAME_WINDOW = st.empty()
            status_placeholder = st.empty()

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                # Validate class name before allowing camera start
                can_start = bool(current_class and current_class.strip())
                if not can_start and not st.session_state.camera_running:
                    st.warning("⚠️ Vui lòng nhập Lớp/Môn học trước!")

                start_cam = st.button(
                    "🔴 Bật Camera Real-time",
                    type="primary",
                    disabled=st.session_state.camera_running or not can_start,
                )
            with col_btn2:
                stop_cam = st.button(
                    "⏹️ Dừng Camera",
                    type="secondary",
                    disabled=not st.session_state.camera_running,
                )

            if stop_cam:
                st.session_state.stop_camera = True
                st.session_state.camera_running = False

        if start_cam or st.session_state.camera_running:
            st.session_state.camera_running = True
            st.session_state.stop_camera = False

            try:
                # 1. Khởi tạo camera từ session state
                if st.session_state.camera is None:
                    st.session_state.camera = cv2.VideoCapture(config.CAMERA_INDEX)
                    if not st.session_state.camera.isOpened():
                        st.error("❌ Camera không thể mở")
                        st.session_state.camera = None
                        st.session_state.camera_running = False
                        st.stop()

                    # Set camera resolution for better quality
                    st.session_state.camera.set(
                        cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH
                    )
                    st.session_state.camera.set(
                        cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT
                    )
                    st.session_state.camera.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)

                cap = st.session_state.camera
                detector, embed_model, _, _ = face_processing.load_models()

                # Load embeddings into cache if not already loaded
                if st.session_state.embeddings_cache is None:
                    st.session_state.embeddings_cache = (
                        face_processing.db.load_embeddings()
                    )
                    if st.session_state.embeddings_cache:
                        st.session_state.embedding_names = list(
                            st.session_state.embeddings_cache.keys()
                        )
                        st.session_state.embedding_matrix = np.array(
                            list(st.session_state.embeddings_cache.values())
                        )
                    else:
                        st.warning("Chưa có dữ liệu nhân viên!")
                        st.session_state.embedding_names = []
                        st.session_state.embedding_matrix = None

                known_names = st.session_state.embedding_names
                known_emb_matrix = st.session_state.embedding_matrix

                frame_count = 0

                # Loop với điều kiện stop
                while cap.isOpened() and not st.session_state.stop_camera:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame = cv2.flip(frame, 1)
                    frame_count += 1

                    # Skip processing for performance
                    process_this_frame = (
                        frame_count % config.PROCESS_EVERY_N_FRAMES == 0
                    )

                    debug_frame = frame.copy()
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w = rgb_frame.shape[:2]

                    if process_this_frame and detector:
                        # Resize nhận diện
                        scale = config.DETECTION_RESIZE_WIDTH / float(w)
                        small_frame = cv2.resize(rgb_frame, (0, 0), fx=scale, fy=scale)

                        detections = detector.detect_faces(small_frame)

                        if detections:
                            detection = max(
                                detections, key=lambda d: d["box"][2] * d["box"][3]
                            )
                            x, y, w_box, h_box = detection["box"]

                            # Scale về ảnh gốc
                            x = int(x / scale)
                            y = int(y / scale)
                            w_box = int(w_box / scale)
                            h_box = int(h_box / scale)

                            # Logic Margin (Cắt mặt)
                            margin = config.FACE_MARGIN
                            x_new = max(0, int(x - w_box * margin))
                            y_new = max(0, int(y - h_box * margin))
                            w_new = min(w - x_new, int(w_box * (1 + 2 * margin)))
                            h_new = min(h - y_new, int(h_box * (1 + 2 * margin)))

                            face_img = rgb_frame[
                                y_new : y_new + h_new, x_new : x_new + w_new
                            ]

                            name_disp = "Unknown"
                            score_disp = 0.0
                            color = (255, 0, 0)  # Đỏ (RGB format for display)

                            # Nhận diện nhanh
                            if face_img.shape[0] > 20 and known_emb_matrix is not None:
                                try:
                                    name_disp, score_disp = (
                                        face_processing.recognize_from_crop(
                                            face_img, known_emb_matrix, known_names
                                        )
                                    )

                                    if name_disp != "Unknown":
                                        color = (0, 255, 0)  # Xanh lá
                                except Exception:
                                    pass

                            # Vẽ bounding box (RGB format)
                            cv2.rectangle(
                                debug_frame, (x, y), (x + w_box, y + h_box), color, 3
                            )

                            # Vẽ label với background
                            label = f"{name_disp} ({score_disp:.2f})"
                            (label_w, label_h), baseline = cv2.getTextSize(
                                label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                            )
                            cv2.rectangle(
                                debug_frame,
                                (x, y - label_h - 10),
                                (x + label_w, y),
                                color,
                                -1,
                            )
                            cv2.putText(
                                debug_frame,
                                label,
                                (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (255, 255, 255),  # Trắng
                                2,
                            )

                            # --- LOGIC TỰ ĐỘNG CHỤP ---
                            if name_disp != "Unknown":
                                if name_disp == st.session_state.target_name_prev:
                                    st.session_state.consecutive_match_count += 1
                                else:
                                    st.session_state.consecutive_match_count = 0
                                    st.session_state.target_name_prev = name_disp

                                # Hiển thị đếm ngược
                                remain = (
                                    config.CONSECUTIVE_MATCH_THRESHOLD
                                    - st.session_state.consecutive_match_count
                                )
                                countdown_text = f"Giu nguyen {name_disp}... {remain}"
                                cv2.putText(
                                    debug_frame,
                                    countdown_text,
                                    (10, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1.2,
                                    (0, 255, 255),  # Vàng
                                    3,
                                )

                                status_placeholder.info(
                                    f"🎯 Đang nhận diện: **{name_disp}** (Còn {remain}s)"
                                )

                                # Chụp ảnh khi đủ số frame
                                if (
                                    st.session_state.consecutive_match_count
                                    >= config.CONSECUTIVE_MATCH_THRESHOLD
                                ):
                                    st.session_state.captured_frame = frame.copy()

                                    # Hiển thị thông báo DONE
                                    cv2.rectangle(
                                        debug_frame, (0, 0), (w, h), (0, 255, 0), 15
                                    )
                                    cv2.putText(
                                        debug_frame,
                                        "DONE! PROCESSING...",
                                        (w // 2 - 200, h // 2),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1.5,
                                        (0, 255, 0),
                                        4,
                                    )

                                    # Hiển thị frame DONE trong 1 giây
                                    done_frame = cv2.cvtColor(
                                        debug_frame, cv2.COLOR_BGR2RGB
                                    )
                                    display_height = config.DISPLAY_HEIGHT
                                    display_width = int(w * display_height / h)
                                    done_resized = cv2.resize(
                                        done_frame, (display_width, display_height)
                                    )

                                    FRAME_WINDOW.image(
                                        done_resized,
                                        channels="RGB",
                                        width=display_width,
                                    )
                                    time.sleep(1)

                                    # Dừng camera và break
                                    st.session_state.stop_camera = True
                                    st.session_state.camera_running = False
                                    break
                            else:
                                st.session_state.consecutive_match_count = 0
                                status_placeholder.warning(
                                    "⚠️ Không nhận diện được - Điều chỉnh góc độ"
                                )

                    # Hiển thị frame trong Streamlit với kích thước cố định
                    display_frame = cv2.cvtColor(debug_frame, cv2.COLOR_BGR2RGB)

                    # Resize về kích thước chuẩn cho display (tránh phóng to thu nhỏ)
                    display_height = config.DISPLAY_HEIGHT
                    display_width = int(w * display_height / h)
                    display_resized = cv2.resize(
                        display_frame, (display_width, display_height)
                    )

                    FRAME_WINDOW.image(
                        display_resized,
                        channels="RGB",
                        width=display_width,  # Fixed width thay vì use_container_width
                    )

                    # Giảm delay để mượt hơn
                    time.sleep(0.01)

            finally:
                # Cleanup resources
                if st.session_state.camera is not None:
                    st.session_state.camera.release()
                    st.session_state.camera = None
                st.session_state.camera_running = False
                status_placeholder.empty()
                # Reset counters
                st.session_state.consecutive_match_count = 0
                st.session_state.target_name_prev = None

            # --- XỬ LÝ KẾT QUẢ (Giữ nguyên) ---
            if st.session_state.captured_frame is not None:
                with c2:
                    st.subheader("✅ Đã tự động bắt được ảnh")
                    with st.spinner(f"Đang thực hiện {attendance_action}..."):
                        # [FIX] Truyền thẳng frame gốc vào verify_face, bỏ qua bước nén JPEG
                        names_str, img_out, emotion, score, action_str, _ = (
                            face_processing.verify_face(
                                image_bytes=None,
                                input_class_name=current_class,
                                action_type=attendance_action,
                                enable_logging=True,
                                image_cv2=st.session_state.captured_frame,  # Use session state
                            )
                        )

                        st.image(
                            img_out,
                            caption=f"Ảnh bằng chứng ({names_str})",
                            channels="RGB",
                        )

                        if "Giả mạo" in names_str:
                            st.error("⚠️ CẢNH BÁO GIẢ MẠO")
                        elif names_str == "NGUOI LA":
                            st.warning("🚫 Không nhận diện được người này.")
                        else:
                            st.success(f"👤 **{names_str}**")
                            if "thành công" in action_str:
                                st.balloons()
                                st.success(f"🎉 {action_str}")
                                # Reset captured frame after successful attendance
                                st.session_state.captured_frame = None
                                st.session_state.consecutive_match_count = 0
                                st.session_state.target_name_prev = None
                            else:
                                st.info(f"ℹ️ {action_str}")
                            st.caption(f"Score: {score:.2f} | Emotion: {emotion}")

                            if st.button("🔄 Tiếp tục người tiếp theo"):
                                st.session_state.captured_frame = None
                                st.session_state.consecutive_match_count = 0
                                st.session_state.target_name_prev = None
                                st.rerun()

# ================= 2. DASHBOARD TỔNG QUAN (FIX LỖI HIỂN THỊ) =================
elif app_mode == "📊 Dashboard Tổng quan":
    st.header("Báo cáo Hoạt động Toàn công ty 📊")

    # Nút làm mới thủ công để chắc chắn load dữ liệu mới nhất
    if st.button("🔄 Tải lại dữ liệu mới nhất"):
        st.cache_data.clear()
        st.rerun()

    df = face_processing.db.get_logs()
    embeddings = face_processing.db.load_embeddings()
    total_registered = len(embeddings)

    if df is None or df.empty:
        st.warning("📭 Chưa có dữ liệu log nào trong hệ thống.")
    else:
        try:
            # 1. Chuẩn hóa dữ liệu - Timestamps already parsed in get_logs()
            # No need to copy - operate directly for better performance

            # Validate timestamp column exists and is datetime
            if df["timestamp"].dtype != "datetime64[ns]":
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

            # Loại bỏ các dòng bị lỗi timestamp (NaT) - use inplace for efficiency
            df.dropna(subset=["timestamp"], inplace=True)

            df["Date"] = df["timestamp"].dt.date
            df["Hour"] = df["timestamp"].dt.hour
            df["DayOfWeek"] = df["timestamp"].dt.day_name()

            # 2. BỘ LỌC (FILTER)
            with st.expander("📅 Bộ lọc Dữ liệu", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    # Mặc định chọn "Toàn bộ" để đảm bảo thấy dữ liệu
                    filter_option = st.selectbox(
                        "Khoảng thời gian:",
                        ["Toàn bộ", "Hôm nay", "7 ngày qua", "Tháng này"],
                        index=0,
                    )
                with c2:
                    # Lấy danh sách lớp an toàn
                    raw_classes = list(df["class_name"].astype(str).unique())
                    clean_classes = [
                        x for x in raw_classes if x.lower() not in ["nan", "none", ""]
                    ]
                    all_classes = ["Tất cả"] + sorted(clean_classes)
                    selected_class = st.selectbox(
                        "Lọc theo Lớp/Phòng ban:", all_classes
                    )

            # 3. XỬ LÝ LỌC
            today = datetime.now().date()
            if filter_option == "Hôm nay":
                filtered_df = df[df["Date"] == today]
            elif filter_option == "7 ngày qua":
                start_date = today - timedelta(days=7)
                filtered_df = df[df["Date"] >= start_date]
            elif filter_option == "Tháng này":
                start_date = today.replace(day=1)
                filtered_df = df[df["Date"] >= start_date]
            else:
                filtered_df = df  # Toàn bộ

            if selected_class != "Tất cả":
                filtered_df = filtered_df[filtered_df["class_name"] == selected_class]

            # Danh sách người thực (Active Users)
            real_users_filtered = filtered_df[
                ~filtered_df["name_detected"].isin(["Người lạ", "N/A"])
            ]

            # --- METRICS ---
            total_scans = len(filtered_df)
            unique_active = real_users_filtered["name_detected"].nunique()
            checkin_count = len(filtered_df[filtered_df["action"] == "Check-in"])
            checkout_count = len(filtered_df[filtered_df["action"] == "Check-out"])

            # Số vắng (Tính dựa trên tổng đăng ký)
            absent_count = max(0, total_registered - unique_active)

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(
                    f"""<div class="metric-container" style="border-left: 5px solid #3498db;"><div class="metric-label">Lượt Quét (Filter)</div><div class="metric-value">{total_scans}</div></div>""",
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f"""<div class="metric-container" style="border-left: 5px solid #2ecc71;"><div class="metric-label">Nhân sự Active</div><div class="metric-value">{unique_active} <span style="font-size:14px; color:gray">/ {total_registered}</span></div></div>""",
                    unsafe_allow_html=True,
                )
            with m3:
                st.markdown(
                    f"""<div class="metric-container" style="border-left: 5px solid #f1c40f;"><div class="metric-label">Check-in</div><div class="metric-value">{checkin_count}</div></div>""",
                    unsafe_allow_html=True,
                )
            with m4:
                st.markdown(
                    f"""<div class="metric-container" style="border-left: 5px solid #9b59b6;"><div class="metric-label">Check-out</div><div class="metric-value">{checkout_count}</div></div>""",
                    unsafe_allow_html=True,
                )

            st.write("")

            # --- BIỂU ĐỒ (CHARTS) ---
            if not filtered_df.empty:
                col_chart1, col_chart2 = st.columns([2, 1])

                with col_chart1:
                    st.subheader("📈 Xu hướng Hoạt động")
                    # Chọn trục X phù hợp
                    if filter_option == "Hôm nay":
                        x_encode = alt.X("Hour:O", title="Giờ")
                        chart_data = (
                            filtered_df.groupby("Hour")
                            .size()
                            .reset_index(name="Counts")
                        )
                    else:
                        x_encode = alt.X(
                            "Date:T", axis=alt.Axis(format="%d/%m"), title="Ngày"
                        )
                        chart_data = (
                            filtered_df.groupby("Date")
                            .size()
                            .reset_index(name="Counts")
                        )

                    area_chart = (
                        alt.Chart(chart_data)
                        .mark_area(
                            line={"color": "#3498db"},
                            color=alt.Gradient(
                                gradient="linear",
                                stops=[
                                    alt.GradientStop(color="#3498db", offset=0),
                                    alt.GradientStop(
                                        color="rgba(255,255,255,0)", offset=1
                                    ),
                                ],
                                x1=1,
                                x2=1,
                                y1=1,
                                y2=0,
                            ),
                        )
                        .encode(
                            x=x_encode,
                            y=alt.Y("Counts:Q", title="Số lượt"),
                            tooltip=["Counts"],
                        )
                        .properties(height=300)
                    )
                    st.altair_chart(area_chart)

                with col_chart2:
                    st.subheader("🏆 Top Chăm chỉ")
                    if not real_users_filtered.empty:
                        top = (
                            real_users_filtered["name_detected"]
                            .value_counts()
                            .head(5)
                            .reset_index()
                        )
                        top.columns = ["Name", "Count"]
                        bar = (
                            alt.Chart(top)
                            .mark_bar()
                            .encode(
                                x=alt.X("Count:Q", title="Số lượt"),
                                y=alt.Y("Name:N", sort="-x", title="Tên"),
                                color=alt.Color("Count:Q", legend=None),
                                tooltip=["Name", "Count"],
                            )
                            .properties(height=300)
                        )
                        st.altair_chart(bar)
                    else:
                        st.info("Chưa có dữ liệu nhân viên.")
            else:
                st.info("Không có dữ liệu trong khoảng thời gian này.")

            # --- BẢNG CHI TIẾT ---
            st.subheader("📜 Nhật ký chi tiết")
            st.dataframe(
                filtered_df[
                    [
                        "timestamp",
                        "name_detected",
                        "mssv",
                        "class_name",
                        "action",
                        "emotion",
                    ]
                ].sort_values("timestamp", ascending=False),
                width="stretch",
            )

        except Exception as e:
            st.error(f"Lỗi hiển thị Dashboard: {e}")
            st.write("Vui lòng kiểm tra file 'attendance_log.csv'.")

# ================= 3. QUẢN LÝ & HỒ SƠ =================
elif app_mode == "👥 Quản lý & Hồ sơ Nhân viên":
    tab1, tab2 = st.tabs(["📂 Hồ sơ & Thống kê Cá nhân", "➕ Đăng ký Mới"])

    # --- TAB 1 ---
    with tab1:
        embeddings = face_processing.db.load_embeddings()
        if not embeddings:
            st.info("Danh sách trống.")
        else:
            users = list(embeddings.keys())
            col_list, col_detail = st.columns([2, 3])

            with col_list:
                st.markdown("### 📋 Danh sách nhân sự")
                search_txt = st.text_input(
                    "🔍 Tìm kiếm:", placeholder="Nhập tên hoặc MSSV..."
                )
                selected_user = st.selectbox("Chọn nhanh:", ["-- Chọn --"] + users)
                st.markdown("---")

                display_users = [u for u in users if search_txt.lower() in u.lower()]
                if not display_users:
                    st.warning("Không tìm thấy.")
                else:
                    grid_cols = st.columns(2)
                    for i, u_name in enumerate(display_users):
                        mssv, u_class = face_processing.db.get_user_info(u_name)
                        with grid_cols[i % 2]:
                            st.markdown(
                                f"""
                            <div class="user-list-card">
                                <div class="ul-avatar">👤</div>
                                <div class="ul-name">{u_name}</div>
                                <div class="ul-mssv">{mssv}</div>
                                <div class="ul-class">{u_class}</div>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

            with col_detail:
                if selected_user != "-- Chọn --":
                    mssv, class_name = face_processing.db.get_user_info(selected_user)
                    user_data_full = face_processing.db.get_full_user_data(
                        selected_user
                    )

                    st.markdown(
                        f"""
                    <div class="profile-card">
                        <div class="avatar-circle">👤</div>
                        <div class="profile-info">
                            <h2>{selected_user}</h2>
                            <p>🆔 MSSV: <b>{mssv}</b> &nbsp;|&nbsp; 🏫 Lớp: <b>{class_name}</b></p>
                        </div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    all_logs = face_processing.db.get_logs()
                    user_logs = pd.DataFrame()
                    if not all_logs.empty:
                        user_logs = all_logs[
                            all_logs["name_detected"] == selected_user
                        ].copy()

                    if not user_logs.empty:
                        user_logs["timestamp"] = pd.to_datetime(user_logs["timestamp"])

                        m1, m2, m3 = st.columns(3)
                        m1.markdown(
                            f"<div class='metric-box'><div class='metric-value-small'>{len(user_logs)}</div><div class='metric-label-small'>Tổng lượt quét</div></div>",
                            unsafe_allow_html=True,
                        )
                        last_seen = user_logs["timestamp"].max().strftime("%H:%M %d/%m")
                        m2.markdown(
                            f"<div class='metric-box'><div class='metric-value-small'>{last_seen}</div><div class='metric-label-small'>Lần cuối xuất hiện</div></div>",
                            unsafe_allow_html=True,
                        )

                        # --- FIX LỖI KEY ERROR Ở ĐÂY ---
                        fav_emo = "N/A"
                        if "emotion" in user_logs.columns:
                            mode_res = user_logs["emotion"].mode()
                            if not mode_res.empty:
                                fav_emo = mode_res[0]

                        m3.markdown(
                            f"<div class='metric-box'><div class='metric-value-small'>{fav_emo.split(' ')[0]}</div><div class='metric-label-small'>Tâm trạng chính</div></div>",
                            unsafe_allow_html=True,
                        )

                        st.write("")
                        st.markdown("#### 📅 Nhật ký hoạt động")
                        timeline = (
                            alt.Chart(user_logs)
                            .mark_circle(size=100)
                            .encode(
                                x=alt.X("timestamp:T", title="Thời gian"),
                                y=alt.Y("action:N", title="Hành động"),
                                color=alt.Color("action:N"),
                                tooltip=["timestamp", "action"],
                            )
                            .properties(height=150)
                            .interactive()
                        )
                        st.altair_chart(timeline)

                        c1, c2 = st.columns(2)
                        with c1:
                            st.write("##### 😊 Diễn biến Cảm xúc")
                            if "emotion" in user_logs.columns:
                                emo_timeline = (
                                    alt.Chart(user_logs)
                                    .mark_point(filled=True, size=80)
                                    .encode(
                                        x=alt.X(
                                            "timestamp:T",
                                            axis=alt.Axis(format="%H:%M"),
                                            title="Giờ",
                                        ),
                                        y=alt.Y("emotion:N", title="Cảm xúc"),
                                        color=alt.Color("emotion:N", legend=None),
                                        tooltip=["timestamp", "emotion"],
                                    )
                                    .properties(height=250)
                                )
                                st.altair_chart(emo_timeline)

                        with c2:
                            st.write("##### 📊 Tần suất theo Ngày")
                            user_logs["Date"] = user_logs["timestamp"].dt.date
                            daily_freq = (
                                user_logs.groupby("Date")
                                .size()
                                .reset_index(name="Counts")
                            )
                            freq_chart = (
                                alt.Chart(daily_freq)
                                .mark_bar()
                                .encode(
                                    x=alt.X("Date:T", title="Ngày"),
                                    y=alt.Y("Counts:Q", title="Số lượt"),
                                    tooltip=["Date", "Counts"],
                                )
                                .properties(height=250)
                            )
                            st.altair_chart(freq_chart)
                    else:
                        st.warning("⚠️ Nhân viên này chưa có dữ liệu điểm danh nào.")

                    st.markdown("---")
                    st.subheader("⚙️ Quản trị Hồ sơ")

                    tab_edit, tab_re_face, tab_history, tab_delete = st.tabs(
                        [
                            "✏️ Sửa Thông tin",
                            "📸 Cập nhật Khuôn mặt",
                            "📜 Lịch sử",
                            "🗑️ Xóa",
                        ]
                    )

                    with tab_edit:
                        if user_data_full is None:
                            st.error(
                                "❌ Không thể tải dữ liệu người dùng. Vui lòng thử lại."
                            )
                        else:
                            with st.form("edit_form"):
                                # Input fields
                                new_mssv = st.text_input("Mã số SV (Mới):", value=mssv)
                                new_class = st.text_input(
                                    "Lớp mặc định (Mới):", value=class_name
                                )
                                submit = st.form_submit_button("💾 Cập nhật thông tin")

                                if submit:
                                    # --- VALIDATION KHÔNG DÙNG REGEX ---

                                    # 1. Kiểm tra rỗng
                                    if not new_mssv.strip():
                                        st.error("MSSV không được để trống.")

                                    # 2. Kiểm tra độ dài MSSV (ví dụ: tối đa 20 ký tự)
                                    elif len(new_mssv) > 20:
                                        st.error("MSSV quá dài (tối đa 20 ký tự).")

                                    # 3. Kiểm tra ký tự đặc biệt trong MSSV (chỉ cho phép chữ và số)
                                    elif not new_mssv.isalnum():
                                        st.error("MSSV chỉ được chứa chữ cái và số.")

                                    # 4. Kiểm tra tên lớp (Rỗng hoặc quá dài)
                                    elif not new_class.strip():
                                        st.error("Tên lớp không được để trống.")
                                    elif len(new_class) > 50:
                                        st.error("Tên lớp quá dài (tối đa 50 ký tự).")
                                    else:
                                        face_processing.db.save_user_data(
                                            selected_user,
                                            new_mssv,
                                            new_class,
                                            user_data_full.get("embedding"),
                                        )
                                        st.success("Cập nhật thành công!")
                                        time.sleep(1)
                                        st.rerun()

                    with tab_re_face:
                        st.info("Chụp lại ảnh để thay thế dữ liệu khuôn mặt cũ.")
                        c_re_1, c_re_2 = st.columns([1, 1])
                        with c_re_1:
                            re_img = st.camera_input("Chụp ảnh mới", key="re_cam")
                        with c_re_2:
                            if re_img:
                                st.image(re_img, caption="Ảnh mới", width=200)
                                if st.button("Lưu khuôn mặt mới", type="primary"):
                                    with st.spinner("Đang cập nhật..."):
                                        res = face_processing.register_face(
                                            selected_user, mssv, class_name, re_img
                                        )
                                        if "thành công" in res:
                                            st.success(
                                                f"Đã cập nhật khuôn mặt cho {selected_user}!"
                                            )
                                            time.sleep(1)
                                        else:
                                            st.error(res)

                    with tab_history:
                        if not user_logs.empty:
                            st.dataframe(
                                user_logs[
                                    [
                                        "timestamp",
                                        "class_name",
                                        "action",
                                        "emotion",
                                        "similarity_score",
                                    ]
                                ],
                                width="stretch",
                            )
                        else:
                            st.info("Chưa có dữ liệu.")

                    with tab_delete:
                        st.warning("Hành động này sẽ xóa vĩnh viễn dữ liệu.")
                        if st.button(f"Xác nhận xóa {selected_user}", type="primary"):
                            if face_processing.db.delete_embedding(selected_user):
                                st.success("Đã xóa thành công!")
                                time.sleep(1)
                                st.rerun()
                else:
                    st.info(
                        "👈 Vui lòng chọn một nhân viên từ danh sách bên trái để xem hồ sơ."
                    )

    # --- TAB 2: ĐĂNG KÝ MỚI ---
    with tab2:
        st.header("Đăng ký Nhân viên Mới")
        c_form, c_cam = st.columns([1, 1])
        with c_cam:
            reg_img = st.camera_input("Chụp ảnh khuôn mặt")
        with c_form:
            st.subheader("1. Nhập thông tin")
            r_name = st.text_input("Họ tên (*):")
            r_mssv = st.text_input("Mã số (*):")
            r_class = st.text_input("Lớp/Phòng ban:")
            st.subheader("2. Kiểm tra ảnh")
            if reg_img:
                st.image(reg_img, caption="Ảnh vừa chụp", width=250)
                st.success("Ảnh đã sẵn sàng!")
            else:
                st.info("Chưa có ảnh.")
            st.markdown("---")
            r_btn = st.button("Lưu Dữ liệu 💾", type="primary", width="stretch")

        if r_btn:
            import re

            if not r_name or not r_mssv:
                st.error("Vui lòng nhập Tên và Mã số.")
            elif not reg_img:
                st.error("Vui lòng chụp ảnh.")
            elif not re.match(r"^[a-zA-Z\sÀ-ỹ]{2,50}$", r_name):
                st.error("Tên không hợp lệ (2-50 ký tự, chỉ chữ cái và khoảng trắng)")
            elif not re.match(r"^[a-zA-Z0-9]{1,20}$", r_mssv):
                st.error("MSSV không hợp lệ (1-20 ký tự, chỉ chữ và số)")
            elif r_class and len(r_class) > 50:
                st.error("Tên lớp quá dài (tối đa 50 ký tự)")
            else:
                with st.spinner("Đang xử lý..."):
                    # Clear embeddings cache to force reload after registration
                    st.session_state.embeddings_cache = None
                    st.session_state.embedding_matrix = None
                    st.session_state.embedding_names = None

                    res = face_processing.register_face(
                        r_name, r_mssv, r_class, reg_img
                    )
                    if "thành công" in res:
                        st.balloons()
                        st.success(res)
                    else:
                        st.error(res)
