import streamlit as st
import face_processing

# --- CẤU HÌNH TRANG & CSS LÀM ĐẸP GIAO DIỆN ---
st.set_page_config(layout="wide", page_title="Hệ thống Nhận diện", page_icon="🧑‍💻")

# --- CSS TÙY CHỈNH (ĐÃ BỎ `aria_label` NHƯNG GIỮ LẠI STYLE CHUNG) ---
st.markdown(
    """
<style>
    /* Đổi font chữ toàn bộ app */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
    body, .stApp {
        font-family: 'Roboto', sans-serif;
        background-color: #F0F2F6; /* Màu nền app nhạt */
    }

    /* Làm đẹp header */
    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0);
    }

    /* Làm đẹp sidebar */
    [data-testid="stSidebar"] {
        background-color: #0F172A; /* Màu nền sidebar (Xanh đậm) */
        border-right: 2px solid #334155;
    }
    [data-testid="stSidebar"] .stSelectbox label {
        color: #F1F5F9;
        font-weight: 700;
        font-size: 1.1rem;
    }

    /* Làm đẹp nút bấm */
    .stButton > button {
        background-color: #3B82F6;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 500;
        box-shadow: 0 4px 14px 0 rgba(0, 118, 255, 0.39);
        transition: all 0.2s ease-in-out;
    }
    .stButton > button:hover {
        background-color: #2563EB;
        box-shadow: 0 6px 20px 0 rgba(0, 118, 255, 0.23);
        transform: translateY(-2px);
    }
    
    /* Nút Xóa, Xác nhận, Hủy bỏ giờ sẽ CÙNG MÀU XANH DƯƠNG */
    /* CSS cho [aria-label] đã bị xóa để tránh lỗi */


    /* Làm đẹp khung camera */
    [data-testid="stCameraInput"] video {
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.1);
    }

    /* --- CSS MỚI CHO DASHBOARD --- */
    .block-container {
        padding-top: 2rem;
    }
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stHorizontalBlock"] {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    [data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0;
        border-radius: 8px;
    }
    [data-testid="stDataFrame"] .col-header {
        background-color: #F8FAFC;
        font-size: 1rem;
        font-weight: 600;
        color: #334155;
    }

</style>
""",
    unsafe_allow_html=True,
)


st.title("Hệ thống Điểm danh & Nhận diện Khuôn mặt 🧑‍💻")
st.write("**Đồ án COS30082** - Tích hợp Nhận diện khuôn mặt, Cảm xúc & Chống giả mạo")

# --- Tải models (giữ nguyên) ---
with st.spinner("Đang tải model, vui lòng đợi..."):
    try:
        face_processing.load_models()
        st.sidebar.success("Models đã tải xong!")
    except Exception as e:
        st.sidebar.error(f"Lỗi tải model: {e}")
        st.stop()

# --- Giao diện Sidebar (giữ nguyên) ---
st.sidebar.header("Chức năng")
app_mode = st.sidebar.selectbox(
    "Chọn chức năng:",
    [
        "🏠 Điểm danh / Xác thực",
        "👤 Đăng ký (Khuôn mặt)",
        "📊 Logs Điểm danh",
        "🛠️ Quản lý Người dùng",
    ],
    label_visibility="hidden",
)

# --- Logic các trang ---

if app_mode == "👤 Đăng ký (Khuôn mặt)":
    st.header("Đăng ký Người dùng mới 👤")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.info(
            """
            **Hướng dẫn Đăng ký:**
            1.  Nhập tên của bạn (ví dụ: "Anh Khoi").
            2.  Nhấn "Mở webcam".
            3.  Nhìn thẳng camera và nhấn "Chụp ảnh".
            4.  Nhấn "Thực hiện Đăng ký".
            """
        )
        name = st.text_input("1. Nhập tên của bạn:", placeholder="Ví dụ: Anh Khoi")
        img_buffer = st.camera_input("2. Mở webcam để chụp ảnh đăng ký")

        if st.button("4. Thực hiện Đăng ký", width='stretch'):
            if not name:
                st.error("Vui lòng nhập tên trước khi đăng ký.")
            elif img_buffer is None:
                st.error("Vui lòng chụp ảnh trước khi đăng ký (Bước 3).")
            else:
                with st.spinner("Đang xử lý đăng ký..."):
                    status = face_processing.register_face(name, img_buffer)
                    if "thành công" in status:
                        st.success(status)
                    else:
                        st.error(status)
    with col2:
        st.subheader("3. Ảnh chụp của bạn:")
        if img_buffer is not None:
            # --- (ĐÃ SỬA) ---
            st.image(img_buffer, caption="Ảnh vừa chụp", width='stretch')
        else:
            st.info("Hình ảnh chụp từ webcam sẽ xuất hiện ở đây.")


elif app_mode == "🏠 Điểm danh / Xác thực":
    st.header("Kiểm tra Điểm danh 📸")
    col1, col2 = st.columns([1, 1])

    with col1:
        img_buffer = st.camera_input("Mở webcam để điểm danh")

    with col2:
        if img_buffer is not None:
            with st.spinner("Đang phân tích..."):
                status, annotated_image, emotion, similarity_score, action_taken = (
                    face_processing.verify_face(img_buffer)
                )

            if annotated_image is not None:
                # --- (ĐÃ SỬA) ---
                st.image(
                    annotated_image,
                    caption="Kết quả phát hiện",
                    width='stretch',
                )

            if status == "Giả mạo (Spoof)":
                st.error("PHÁT HIỆN GIẢ MẠO! 🛑 Yêu cầu điểm danh bị từ chối.")
            elif status == "Không tìm thấy":
                st.warning(
                    "Không tìm thấy khuôn mặt. Vui lòng giữ yên và nhìn thẳng camera."
                )
            elif status == "NGƯỜI LẠ (Stranger)":
                st.error(
                    f"NGƯỜI LẠ (Stranger). (Score: {similarity_score:.2f})", icon="🚫"
                )
                st.info(
                    "Nếu bạn là người dùng mới, vui lòng qua tab 'Đăng ký (Khuôn mặt)'."
                )
            elif status == "Không có CSDL":
                st.warning("Hệ thống chưa có ai đăng ký. Vui lòng đăng ký trước.")
            else:
                if action_taken == "Check-in":
                    st.success(f"Chào {status}! Bạn đã **Check-in** thành công. ✅")
                elif action_taken == "Check-out":
                    st.info(f"Tạm biệt {status}! Bạn đã **Check-out** thành công. 🚪")

                m_col1, m_col2 = st.columns(2)
                m_col1.metric(
                    label="Độ tương đồng (Score)", value=f"{similarity_score:.2f}"
                )
                m_col2.metric(label="Cảm xúc", value=emotion)
        else:
            st.info("Kết quả sẽ xuất hiện ở đây sau khi bạn chụp ảnh.")


elif app_mode == "📊 Logs Điểm danh":
    st.header("Dashboard - Logs Điểm danh 📊")
    st.write("Toàn bộ lịch sử check-in, check-out và các sự kiện khác được ghi lại.")

    log_data = face_processing.db.get_logs()
    if log_data is None or log_data.empty:
        st.warning("Chưa có log nào được ghi lại.")
    else:
        st.dataframe(log_data, width='stretch')
        st.download_button(
            label="Tải log về (CSV)",
            data=log_data.to_csv(index=False).encode("utf-8"),
            file_name="attendance_log.csv",
            mime="text/csv",
        )
        st.info(
            "Lưu ý: Để xóa hoặc chỉnh sửa log, vui lòng tải file CSV về và xử lý bằng Excel."
        )


elif app_mode == "🛠️ Quản lý Người dùng":
    st.header("Quản lý Người dùng đã Đăng ký 🛠️")
    st.write("Xem và xóa các người dùng đã đăng ký trong hệ thống.")

    known_embeddings = face_processing.db.load_embeddings()

    if not known_embeddings:
        st.info("Chưa có ai đăng ký khuôn mặt trong hệ thống.")
    else:
        users = list(known_embeddings.keys())
        st.write(f"Đã tìm thấy **{len(users)}** người dùng đã đăng ký:")

        col1, col2 = st.columns([2, 1])
        with col1:
            user_to_delete = st.selectbox("Chọn user để xóa:", users)

        with col2:
            st.write("")
            st.write("")
            if st.button(
                "Xóa",
                key="delete_user",
                width='stretch',
                help=f"Xóa vĩnh viễn user {user_to_delete}",
            ):
                if "confirm_delete" not in st.session_state:
                    st.session_state.confirm_delete = False
                st.session_state.user_to_delete = user_to_delete
                st.session_state.confirm_delete = True

        if "confirm_delete" in st.session_state and st.session_state.confirm_delete:
            st.warning(
                f"Bạn có chắc chắn muốn xóa vĩnh viễn **{st.session_state.user_to_delete}** không?",
                icon="⚠️",
            )
            col1_confirm, col2_confirm, _ = st.columns([1, 1, 4])
            with col1_confirm:
                # --- (ĐÃ SỬA) ---
                if st.button(
                    "XÁC NHẬN XÓA", width='stretch'
                ):  # Đã xóa aria_label
                    success = face_processing.db.delete_embedding(
                        st.session_state.user_to_delete
                    )
                    if success:
                        st.success(
                            f"Đã xóa thành công user {st.session_state.user_to_delete}."
                        )
                    else:
                        st.error("Có lỗi xảy ra khi xóa user.")
                    st.session_state.confirm_delete = False
                    st.rerun()
            with col2_confirm:
                # --- (ĐÃ SỬA) ---
                if st.button("Hủy bỏ", width='stretch'):  # Đã xóa aria_label
                    st.session_state.confirm_delete = False
                    st.rerun()
