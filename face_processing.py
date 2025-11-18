import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
from sklearn.metrics.pairwise import cosine_similarity
import db

# --- Hằng số ---
MODEL_PATH = "models/ResNet50_feature_extractor.keras"
EMBEDDING_LAYER_NAME = "cnn_embedding"
IMG_SIZE = (224, 224)
COSINE_THRESHOLD = 0.8

# --- Hằng số cho 2 model mới ---
# (!!!) CHÚ Ý: Đặt tên file model của bạn vào đây
SPOOF_MODEL_PATH = "models/anti_spoof_model.h5"
EMOTION_MODEL_PATH = "models/emotion_model.h5"

# (!!!) CHÚ Ý: Các thông số này PHẢI KHỚP với model bạn tải về
SPOOF_IMG_SIZE = (224, 224)  # Giả sử model spoof dùng 224x224
EMOTION_IMG_SIZE = (48, 48)  # Model emotion (FER2013) thường là 48x48
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
EMOTION_ICONS = {
    "Happy": "😊",
    "Sad": "😢",
    "Angry": "😠",
    "Surprise": "😮",
    "Neutral": "😐",
    "Fear": "😨",
    "Disgust": "🤢",
}


# --- Tải model (Cập nhật) ---
@st.cache_resource
def load_models():
    print("Đang tải models...")
    detector = MTCNN()

    # 1. Model Embedding (Của bạn)
    try:
        full_model = tf.keras.models.load_model(MODEL_PATH)
        embed_model = tf.keras.Model(
            inputs=full_model.input,
            outputs=full_model.get_layer(EMBEDDING_LAYER_NAME).output,
        )
    except Exception as e:
        st.error(f"Lỗi khi tải model embedding: {e}.")
        return None, None, None, None

    # 2. Model Anti-Spoof
    try:
        spoof_model = tf.keras.models.load_model(SPOOF_MODEL_PATH)
        print(f"Tải model Anti-Spoof '{SPOOF_MODEL_PATH}' thành công.")
    except Exception as e:
        print(f"Không tìm thấy model anti-spoof tại '{SPOOF_MODEL_PATH}'. Bỏ qua...")
        spoof_model = None

    # 3. Model Emotion Detection
    try:
        emotion_model = tf.keras.models.load_model(EMOTION_MODEL_PATH)
        print(f"Tải model Emotion '{EMOTION_MODEL_PATH}' thành công.")
    except Exception as e:
        print(f"Không tìm thấy model emotion tại '{EMOTION_MODEL_PATH}'. Bỏ qua...")
        emotion_model = None

    print("Tải models thành công.")
    return detector, embed_model, spoof_model, emotion_model


# --- Các hàm Pipeline ---
def detect_and_align(image_bytes):
    """
    Phát hiện khuôn mặt, trả về ảnh đã cắt VÀ tọa độ.
    """
    detector, _, _, _ = load_models()
    if detector is None:
        return None, None, None

    # --- (ĐÃ THÊM) ---
    # Tua lại file stream về đầu trước khi đọc
    image_bytes.seek(0)

    img = cv2.imdecode(np.frombuffer(image_bytes.read(), np.uint8), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(img_rgb)

    if not detections:
        return None, None, None  # Không tìm thấy mặt

    detection = detections[0]
    x, y, w, h = detection["box"]
    face_coords = (x, y, w, h)  # Tọa độ khuôn mặt

    face_img = img_rgb[y : y + h, x : x + w]
    face_img_resized = cv2.resize(face_img, IMG_SIZE)

    return face_img_resized, img_rgb, face_coords


def check_anti_spoof(face_img_rgb):
    _, _, spoof_model, _ = load_models()
    if spoof_model is None:
        print("Bỏ qua anti-spoofing (chưa có model).")
        return False
    face_resized_spoof = cv2.resize(face_img_rgb, SPOOF_IMG_SIZE)
    input_tensor = np.expand_dims(face_resized_spoof, axis=0)
    input_tensor = input_tensor / 255.0
    prediction = spoof_model.predict(input_tensor)[0][0]
    SPOOF_THRESHOLD = 0.8
    if prediction > SPOOF_THRESHOLD:
        print(f"Phát hiện Spoof! Score: {prediction:.2f}")
        return True
    else:
        print(f"Ảnh thật. Score: {prediction:.2f}")
        return False


def get_embedding(face_img_rgb):
    _, embed_model, _, _ = load_models()
    face_tensor = np.expand_dims(face_img_rgb, axis=0)
    face_tensor_preprocessed = tf.keras.applications.resnet.preprocess_input(
        face_tensor
    )
    embedding = embed_model.predict(face_tensor_preprocessed)[0]
    embedding = embedding / np.linalg.norm(embedding)
    return embedding


def detect_emotion(face_img_rgb):
    """(Cập nhật) Phát hiện cảm xúc."""
    _, _, _, emotion_model = load_models()
    if emotion_model is None:
        return "N/A"

    # --- Tiền xử lý cho model Emotion ---
    # 1. Chuyển về ảnh xám (Grayscale)
    face_gray = cv2.cvtColor(face_img_rgb, cv2.COLOR_RGB2GRAY)

    # 2. Resize về 48x48
    face_resized_emotion = cv2.resize(face_gray, EMOTION_IMG_SIZE)

    # 3. Chuẩn hóa [0, 1]
    input_tensor = face_resized_emotion / 255.0

    # 4. Thêm chiều batch (1) và chiều kênh (1)
    input_tensor = np.expand_dims(input_tensor, axis=-1)  # (48, 48) -> (48, 48, 1)
    input_tensor = np.expand_dims(input_tensor, axis=0)  # (48, 48, 1) -> (1, 48, 48, 1)

    # 5. Dự đoán
    predictions = emotion_model.predict(input_tensor)[0]
    emotion_index = np.argmax(predictions)
    emotion_text = EMOTION_LABELS[emotion_index]

    print(f"Phát hiện cảm xúc: {emotion_text} ({predictions[emotion_index]:.2f})")
    return f"{emotion_text} {EMOTION_ICONS.get(emotion_text, '')}"


# --- Các hàm chính cho UI (Cập nhật) ---
def register_face(name, image_bytes):
    if not name:
        return "Vui lòng nhập tên."

    # face_img_224x224 là ảnh đã resize (224, 224, 3)
    # img_rgb_original là ảnh gốc (từ webcam)
    face_img_224x224, _, img_rgb_original = detect_and_align(image_bytes)

    if face_img_224x224 is None:
        return "Không phát hiện thấy khuôn mặt."

    # Chạy anti-spoof trên ảnh gốc (chất lượng cao hơn)
    if check_anti_spoof(img_rgb_original):
        return "Phát hiện giả mạo (spoof)! Đăng ký thất bại."

    embedding = get_embedding(face_img_224x224)
    try:
        db.save_embedding(name, embedding)
        return f"Đăng ký thành công cho {name}!"
    except Exception as e:
        return f"Lỗi khi lưu embedding: {e}"


def verify_face(image_bytes):
    """
    Thực hiện pipeline, VÀ VẼ Bounding Box (Đỏ/Xanh)
    """
    # 1. Detect và lấy tọa độ
    face_img_224x224, img_to_draw_on, face_coords = detect_and_align(image_bytes)

    action_taken = "N/A"

    if face_img_224x224 is None:
        db.log_attendance("N/A", "Không tìm thấy khuôn mặt", 0.0, "N/A")

        # --- (ĐÃ SỬA LỖI CRASH) ---
        # Tua lại file stream về đầu
        image_bytes.seek(0)

        # Đọc lại ảnh gốc để hiển thị
        img_rgb_original = cv2.imdecode(
            np.frombuffer(image_bytes.read(), np.uint8), cv2.IMREAD_COLOR
        )
        annotated_img = cv2.cvtColor(img_rgb_original, cv2.COLOR_BGR2RGB)
        return "Không tìm thấy", annotated_img, "N/A", 0.0, action_taken

    # 2. Check Anti-spoof
    if check_anti_spoof(img_to_draw_on):
        db.log_attendance("N/A", "Giả mạo (Spoof)", 0.0, "N/A")
        action_taken = "Giả mạo (Spoof)"

        x, y, w, h = face_coords
        cv2.rectangle(img_to_draw_on, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(
            img_to_draw_on,
            "SPOOF DETECTED",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2,
        )
        return "Giả mạo (Spoof)", img_to_draw_on, "N/A", 0.0, action_taken

    # 3. Lấy embedding và so sánh
    live_embedding = get_embedding(face_img_224x224)
    known_embeddings = db.load_embeddings()

    if not known_embeddings:
        x, y, w, h = face_coords
        cv2.rectangle(img_to_draw_on, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(
            img_to_draw_on,
            "NGUOI LA (Stranger)",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2,
        )
        return "Không có CSDL", img_to_draw_on, "N/A", 0.0, action_taken

    # ... (Phần còn lại của hàm verify_face giữ nguyên) ...
    best_match_name = "Không nhận diện được"
    max_sim = 0.0
    for name, saved_embedding in known_embeddings.items():
        sim = cosine_similarity(
            live_embedding.reshape(1, -1), saved_embedding.reshape(1, -1)
        )[0][0]
        if sim > max_sim:
            max_sim = sim
            if sim > COSINE_THRESHOLD:
                best_match_name = name

    emotion = "N/A"
    display_status = ""
    color = (255, 0, 0)  # Mặc định là đỏ

    if best_match_name != "Không nhận diện được":
        display_status = best_match_name
        color = (0, 255, 0)  # Xanh
        emotion = detect_emotion(face_img_224x224)
        last_action = db.get_last_action(best_match_name)
        if last_action is None or last_action == "Check-out":
            action_taken = "Check-in"
        elif last_action == "Check-in":
            action_taken = "Check-out"
        db.log_attendance(best_match_name, action_taken, max_sim, emotion)
    else:
        display_status = "NGƯỜI LẠ (Stranger)"
        color = (255, 0, 0)  # Đỏ
        action_taken = "Nhận diện thất bại"
        db.log_attendance("Người lạ", action_taken, max_sim, emotion)

    x, y, w, h = face_coords
    cv2.rectangle(img_to_draw_on, (x, y), (x + w, y + h), color, 2)
    cv2.putText(
        img_to_draw_on,
        display_status,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
    )
    return display_status, img_to_draw_on, emotion, max_sim, action_taken
