import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
from sklearn.metrics.pairwise import cosine_similarity
import db
import config
from tensorflow.keras.applications.efficientnet import (
    preprocess_input as efficientnet_preprocess,
)
import logging

# Configure logging
logging.basicConfig(
    filename="face_recognition.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# --- CẤU HÌNH ---
# Import emotion labels from config
EMOTION_LABELS = config.EMOTION_LABELS
EMOTION_ICONS = config.EMOTION_ICONS

# --- Module-level cache for models ---
_CACHED_MODELS = None


# --- Tải model ---
@st.cache_resource
def load_models():
    global _CACHED_MODELS
    if _CACHED_MODELS is not None:
        return _CACHED_MODELS

    print("⚡ Đang tải models tối ưu...")
    detector = MTCNN(min_face_size=60)
    try:
        full_model = tf.keras.models.load_model(
            config.MODEL_PATH,
            custom_objects={
                "preprocess_input": tf.keras.applications.efficientnet.preprocess_input
            },
            compile=False,
        )
        embed_model = tf.keras.Model(
            inputs=full_model.input,
            outputs=full_model.get_layer(config.EMBEDDING_LAYER_NAME).output,
        )
        print("✅ Đã tải Face Recognition Model (B4)")
    except Exception as e:
        logging.exception("Model loading failed")
        st.error(f"Lỗi tải model Face: {e}")
        return None, None, None, None

    emotion_model, spoof_model = None, None
    if config.EMOTION_MODEL_PATH:
        try:
            emotion_model = tf.keras.models.load_model(
                config.EMOTION_MODEL_PATH,
                custom_objects={
                    "preprocess_input": tf.keras.applications.resnet.preprocess_input
                },
                compile=False,
            )
            print("✅ Đã tải Emotion Model (ResNet50)")
        except Exception as e:
            logging.exception("Emotion model loading failed")
            print(f"⚠️ Lỗi Emotion: {e}")

    if config.SPOOF_MODEL_PATH:
        try:
            spoof_model = tf.keras.models.load_model(
                config.SPOOF_MODEL_PATH,
                custom_objects={
                    "preprocess_input": tf.keras.applications.resnet.preprocess_input
                },
                compile=False,
            )
            print("✅ Đã tải Anti-Spoof Model (ResNet50)")
        except Exception as e:
            logging.exception("Spoof detection model loading failed")
            print(f"⚠️ Lỗi Spoof: {e}")

    _CACHED_MODELS = (detector, embed_model, spoof_model, emotion_model)
    return _CACHED_MODELS


# --- Helper Functions ---
def get_embedding(face_img_rgb):
    """Tạo vector đặc trưng từ ảnh khuôn mặt.

    Args:
        face_img_rgb: Ảnh khuôn mặt ĐÃ RESIZE về (224, 224) từ detect_and_align()

    Returns:
        Normalized embedding vector
    """
    _, embed_model, _, _ = load_models()

    if embed_model is None:
        logging.error("Embedding model is None")
        return None

    try:
        # Ảnh đã được resize từ detect_and_align() rồi, không cần resize lại
        face_tensor = np.expand_dims(face_img_rgb.astype("float32"), axis=0)
        face_tensor = tf.keras.applications.efficientnet.preprocess_input(face_tensor)

        # Predict
        embedding = embed_model(face_tensor, training=False)
        embedding = embedding.numpy()[0]

        return embedding / np.linalg.norm(embedding)
    except Exception as e:
        logging.exception(f"Error in get_embedding: {e}")
        return None


def recognize_from_crop(face_img_rgb, known_emb_matrix, known_names):
    """Hàm nhận diện nhanh dùng cho vòng lặp Real-time của OpenCV.

    Args:
        face_img_rgb: Ảnh mặt cắt (CHƯA resize) từ camera
        known_emb_matrix: Ma trận embeddings đã biết
        known_names: Danh sách tên tương ứng

    Returns:
        (name, similarity_score)
    """
    if face_img_rgb.shape[0] < 20 or face_img_rgb.shape[1] < 20:
        return "Unknown", 0.0

    # Resize về 224x224 trước khi tạo embedding
    try:
        face_resized = cv2.resize(face_img_rgb, config.IMG_SIZE)
    except:
        return "Unknown", 0.0

    curr_emb = get_embedding(face_resized)
    if curr_emb is None:
        return "Unknown", 0.0

    # Tính toán so khớp
    sims = cosine_similarity(curr_emb.reshape(1, -1), known_emb_matrix)[0]
    idx_max = np.argmax(sims)
    max_sim = sims[idx_max]

    if max_sim > config.COSINE_THRESHOLD:
        return known_names[idx_max], max_sim
    return "Unknown", max_sim


def detect_emotion(face_img_rgb):
    _, _, _, emotion_model = load_models()
    if emotion_model is None:
        return "N/A"
    try:
        face_resized = cv2.resize(face_img_rgb, config.EMOTION_IMG_SIZE)
        input_tensor = np.expand_dims(face_resized, axis=0).astype("float32")
        # Use ResNet preprocess for ResNet50 emotion model
        input_tensor = tf.keras.applications.resnet.preprocess_input(input_tensor)
        predictions = emotion_model(input_tensor, training=False).numpy()[0]
        if len(predictions) > 10:
            return "N/A"
        idx = np.argmax(predictions)
        return (
            f"{EMOTION_LABELS[idx]} {EMOTION_ICONS.get(EMOTION_LABELS[idx], '')}"
            if idx < len(EMOTION_LABELS)
            else "Unknown"
        )
    except:
        return "N/A"


# --- Pipeline Chính ---
def detect_and_align(image_bytes=None, image_cv2=None):
    """
    Hàm phát hiện khuôn mặt, hỗ trợ cả input là Bytes (Webcam Streamlit)
    và Numpy Array (OpenCV Real-time).
    """
    detector, _, _, _ = load_models()
    if detector is None:
        return None, None, None

    # Ưu tiên xử lý CV2 (Numpy) trước nếu có
    if image_cv2 is not None:
        img_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    elif image_bytes:
        image_bytes.seek(0)
        img = cv2.imdecode(
            np.frombuffer(image_bytes.read(), np.uint8), cv2.IMREAD_COLOR
        )
        if img is None:
            return None, None, None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        return None, None, None

    h_img, w_img = img_rgb.shape[:2]
    target_w = 640
    scale = target_w / float(w_img) if w_img > target_w else 1.0
    img_small = (
        cv2.resize(img_rgb, (0, 0), fx=scale, fy=scale) if scale < 1.0 else img_rgb
    )

    detections = detector.detect_faces(img_small)
    if not detections:
        return None, None, None

    # Lấy mặt to nhất
    detection = max(detections, key=lambda d: d["box"][2] * d["box"][3])
    x, y, w, h = detection["box"]

    # Scale tọa độ về ảnh gốc
    if scale < 1.0:
        x, y, w, h = int(x / scale), int(y / scale), int(w / scale), int(h / scale)

    # [QUAN TRỌNG] Thêm Margin (Lề) 20% để lấy trọn khuôn mặt
    margin = 0.2
    x_new = max(0, int(x - w * margin))
    y_new = max(0, int(y - h * margin))
    w_new = min(w_img - x_new, int(w * (1 + 2 * margin)))
    h_new = min(h_img - y_new, int(h * (1 + 2 * margin)))

    face_img = img_rgb[y_new : y_new + h_new, x_new : x_new + w_new]

    try:
        face_resized = cv2.resize(face_img, config.IMG_SIZE)
    except:
        return None, None, None

    # Trả về: Ảnh mặt đã resize (224x224), Ảnh gốc, Tọa độ mặt gốc (để vẽ khung)
    return face_resized, img_rgb, (x, y, w, h)


# --- UI Functions ---
def register_face(name, mssv, class_name, image_bytes):
    """Đăng ký khuôn mặt mới."""
    if not name or not mssv:
        return "Vui lòng nhập tên và MSSV."

    # Gọi hàm detect
    face_img, _, _ = detect_and_align(image_bytes)

    if face_img is None:
        return "Không tìm thấy khuôn mặt."
    try:
        # Hàm này giờ đã an toàn, tự resize về 224x224
        embedding = get_embedding(face_img)
        if embedding is None:
            return "Lỗi xử lý ảnh (Embedding)."

        db.save_user_data(name, mssv, class_name, embedding)
        return f"Đăng ký thành công: {name}"
    except Exception as e:
        return f"Lỗi: {e}"


def verify_face(
    image_bytes=None,
    input_class_name="",
    action_type="Check-in",
    enable_logging=True,
    image_cv2=None,
):
    """
    Hàm nhận diện và điểm danh.
    Hỗ trợ nhận input từ cả Streamlit (bytes) và OpenCV (numpy array).
    """

    detector, _, spoof_model, _ = load_models()
    if detector is None:
        return "Lỗi Model", None, "N/A", 0.0, "N/A", False

    # 1. Detect & Align (Tự động chọn nguồn ảnh phù hợp)
    face_img, img_rgb_full, coords = detect_and_align(
        image_bytes=image_bytes, image_cv2=image_cv2
    )

    # 2. Chuẩn bị ảnh vẽ kết quả (img_draw)
    img_draw = None
    if img_rgb_full is not None:
        img_draw = img_rgb_full.copy()  # Đã là RGB
    else:
        # Fallback: Cố gắng load ảnh gốc để trả về cho UI dù không detect được mặt
        if image_cv2 is not None:
            img_draw = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
        elif image_bytes:
            image_bytes.seek(0)
            img_bgr = cv2.imdecode(
                np.frombuffer(image_bytes.read(), np.uint8), cv2.IMREAD_COLOR
            )
            if img_bgr is not None:
                img_draw = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Kiểm tra kết quả detect
    if face_img is None or coords is None:
        return "Không tìm thấy", img_draw, "N/A", 0.0, "N/A", False

    x, y, w, h = coords

    # Lọc ảnh quá nhỏ
    if face_img.shape[0] < 40 or face_img.shape[1] < 40:
        return "Không tìm thấy", img_draw, "N/A", 0.0, "N/A", False

    # Ensure img_draw is valid before cv2 operations
    if img_draw is None:
        return "Lỗi ảnh", None, "N/A", 0.0, "N/A", False

    # --- SPOOF CHECK ---
    is_real_face = True
    spoof_score = 0.0
    if spoof_model:
        try:
            spoof_input = cv2.resize(face_img, config.SPOOF_IMG_SIZE).astype("float32")
            spoof_input = np.expand_dims(spoof_input, axis=0)
            # Use ResNet preprocess for ResNet50 anti-spoof model
            spoof_input = tf.keras.applications.resnet.preprocess_input(spoof_input)
            spoof_pred = spoof_model(spoof_input, training=False).numpy()

            # Get real face score (binary classification: [fake, real])
            # Score > threshold = REAL, Score < threshold = FAKE
            spoof_score = (
                spoof_pred[0][1] if spoof_pred.shape[-1] == 2 else spoof_pred[0][0]
            )
            is_real_face = spoof_score > config.SPOOF_THRESHOLD

            # Log spoof result but DON'T early return - continue to recognition
            if is_real_face:
                print(
                    f"✅ [SPOOF] Real face: score={spoof_score:.4f} (threshold={config.SPOOF_THRESHOLD})"
                )
            else:
                print(
                    f"⚠️ [SPOOF] Detected FAKE face: score={spoof_score:.4f} (threshold={config.SPOOF_THRESHOLD})"
                )
        except Exception as e:
            print(f"⚠️ [SPOOF] Error: {e}")
            spoof_score = 0.0
            is_real_face = False
            pass

    # --- RECOGNITION ---
    live_emb = get_embedding(face_img)
    if live_emb is None:
        return "Lỗi ảnh", img_draw, "N/A", 0.0, "N/A", False

    known_embeddings = db.load_embeddings()
    best_name = "Unknown"
    max_sim = 0.0

    if known_embeddings:
        names = list(known_embeddings.keys())
        emb_matrix = np.array(list(known_embeddings.values()))
        sims = cosine_similarity(live_emb.reshape(1, -1), emb_matrix)[0]
        idx_max = np.argmax(sims)
        max_sim = sims[idx_max]
        if max_sim > config.COSINE_THRESHOLD:
            best_name = names[idx_max]

    print(f"👤 [REC] Name: {best_name} | Sim: {max_sim:.4f}")
    emotion = detect_emotion(face_img)

    # --- LOGIC GHI LOG ---
    # Kiểm tra cả cosine similarity VÀ anti-spoof score
    pass_cosine = max_sim > config.COSINE_THRESHOLD
    pass_spoof = is_real_face  # Already checked against SPOOF_THRESHOLD
    both_pass = pass_cosine and pass_spoof

    color = (0, 255, 0) if (best_name != "Unknown" and both_pass) else (255, 0, 0)
    action_log, has_new_checkin = "N/A", False

    if best_name != "Unknown":
        if not both_pass:
            # Nhận diện được nhưng không đạt threshold - hiển thị CẢ HAI điều kiện
            fail_reasons = []
            if not pass_cosine:
                fail_reasons.append(f"Cosine {max_sim:.3f} < {config.COSINE_THRESHOLD}")
                print(f"⚠️ [CHECK] {best_name}: Cosine failed ({max_sim:.3f})")
            if not pass_spoof:
                fail_reasons.append(
                    f"Spoof {spoof_score:.3f} < {config.SPOOF_THRESHOLD}"
                )
                print(f"⚠️ [CHECK] {best_name}: Spoof check failed ({spoof_score:.3f})")

            action_log = f"⚠️ Không đạt: {' & '.join(fail_reasons)}"
        else:
            # CẢ HAI ĐỀU PASS - Cho phép ghi log
            mssv, r_class = db.get_user_info(best_name)
            final_class = input_class_name if input_class_name else r_class
            last_action = db.get_last_action(best_name)

            if action_type == "Check-in":
                if last_action != "Check-in":
                    action_log = f"Sẵn sàng Check-in: {best_name}"
                else:
                    action_log = f"⚠️ {best_name} đã Check-in rồi."
            else:  # Check-out
                if last_action == "Check-in":
                    action_log = f"Sẵn sàng Check-out: {best_name}"
                else:
                    action_log = f"⚠️ {best_name} chưa thể Check-out."

            # Chỉ ghi log nếu enable_logging = True VÀ cả 2 threshold đều pass
            if enable_logging and both_pass:
                if action_type == "Check-in" and last_action != "Check-in":
                    db.log_attendance(
                        best_name,
                        mssv,
                        final_class,
                        "Check-in",
                        max_sim,
                        spoof_score,
                        emotion,
                    )
                    has_new_checkin = True
                    action_log = "✅ Check-in thành công!"
                    print(
                        f"✅ [LOG] Check-in: {best_name} (cos={max_sim:.3f}, spoof={spoof_score:.3f})"
                    )
                elif action_type == "Check-out" and last_action == "Check-in":
                    db.log_attendance(
                        best_name,
                        mssv,
                        final_class,
                        "Check-out",
                        max_sim,
                        spoof_score,
                        emotion,
                    )
                    has_new_checkin = True
                    action_log = "✅ Check-out thành công!"
                    print(
                        f"✅ [LOG] Check-out: {best_name} (cos={max_sim:.3f}, spoof={spoof_score:.3f})"
                    )

    # Vẽ Box với màu tùy theo kết quả
    cv2.rectangle(img_draw, (x, y), (x + w, y + h), color, 2)

    # Label chính: Tên + Cosine score
    label = f"{best_name} (cos:{max_sim:.2f})" if best_name != "Unknown" else "NGUOI LA"
    cv2.putText(img_draw, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Label phụ: Anti-spoof score (hiển thị luôn nếu có spoof model)
    if spoof_model and spoof_score > 0:
        spoof_label = f"Spoof: {spoof_score:.3f}"
        spoof_color = (0, 255, 0) if is_real_face else (255, 0, 0)
        cv2.putText(
            img_draw,
            spoof_label,
            (x, y - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            spoof_color,
            2,
        )
    if "Wrong" not in emotion and "N/A" not in emotion:
        cv2.putText(
            img_draw,
            emotion.split(" ")[0],
            (x, y + h + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

    return best_name, img_draw, emotion, max_sim, action_log, has_new_checkin
