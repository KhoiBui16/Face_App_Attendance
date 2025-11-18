import os
import pickle
import numpy as np
import csv
from datetime import datetime
import pytz
import pandas as pd

DB_DIR = "face_db"
LOG_FILE = "attendance_log.csv"
# --- (ĐÃ THAY ĐỔI) ---
# Bỏ cột "status", thay bằng cột "action" (Check-in, Check-out, v.v.)
LOG_HEADER = ["timestamp", "name_detected", "action", "similarity_score", "emotion"]

os.makedirs(DB_DIR, exist_ok=True)

# ... (Hàm save_embedding, load_embeddings, delete_embedding giữ nguyên) ...
# (Dưới đây là các hàm đã được sửa)


def initialize_log_file():
    """Tạo file log với header nếu chưa tồn tại."""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(LOG_HEADER)


# --- (ĐÃ THAY ĐỔI) ---
# Hàm log_attendance giờ nhận "action" (Check-in/Out) thay vì "status"
def log_attendance(name, action, score, emotion):
    """Ghi lại một bản ghi điểm danh vào file CSV."""
    initialize_log_file()

    vn_tz = pytz.timezone("Asia/Ho_Chi_Minh")
    timestamp = datetime.now(vn_tz).strftime("%Y-%m-%d %H:%M:%S")

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, name, action, f"{score:.2f}", emotion])
    print(f"Đã ghi log: {name}, Action: {action}, Score: {score:.2f}")


# --- (HÀM MỚI) ---
def get_last_action(name):
    """
    Tìm hành động cuối cùng (Check-in/Check-out) của user TRONG NGÀY HÔM NAY.
    Returns: "Check-in", "Check-out", hoặc None
    """
    if not os.path.exists(LOG_FILE):
        return None

    try:
        df = pd.read_csv(LOG_FILE)
        if df.empty:
            return None

        # Lấy ngày hôm nay (YYYY-MM-DD)
        vn_tz = pytz.timezone("Asia/Ho_Chi_Minh")
        today_str = datetime.now(vn_tz).strftime("%Y-%m-%d")

        # Lọc log của user đó VÀ trong ngày hôm nay
        # Giả sử timestamp có dạng "2025-11-18 10:00:00"
        user_logs_today = df[
            (df["name_detected"] == name)
            & (df["timestamp"].str.startswith(today_str))
            & (df["action"].isin(["Check-in", "Check-out"]))
        ]

        if user_logs_today.empty:
            return None  # Chưa có hành động nào hôm nay

        # Sắp xếp và lấy hành động cuối cùng
        last_action = user_logs_today.sort_values(by="timestamp", ascending=False).iloc[
            0
        ]
        return last_action["action"]

    except Exception as e:
        print(f"Lỗi khi get_last_action: {e}")
        return None


def get_logs():
    """Đọc toàn bộ file log bằng Pandas."""
    if not os.path.exists(LOG_FILE):
        initialize_log_file()  # Tạo file nếu chưa có
        return pd.DataFrame(columns=LOG_HEADER)
    try:
        df = pd.read_csv(LOG_FILE)
        if df.empty:
            return pd.DataFrame(columns=LOG_HEADER)
        return df.sort_values(by="timestamp", ascending=False)
    except pd.errors.EmptyDataError:
        # File tồn tại nhưng rỗng
        return pd.DataFrame(columns=LOG_HEADER)
    except Exception as e:
        print(f"Lỗi đọc log: {e}")
        return None


# (Các hàm save, load, delete embedding khác giữ nguyên)
def save_embedding(name, embedding):
    filepath = os.path.join(DB_DIR, f"{name}.pkl")
    embedding = embedding / np.linalg.norm(embedding)
    with open(filepath, "wb") as f:
        pickle.dump(embedding, f)
    print(f"Đã lưu embedding cho {name} tại {filepath}")


def load_embeddings():
    embeddings = {}
    for filename in os.listdir(DB_DIR):
        if filename.endswith(".pkl"):
            filepath = os.path.join(DB_DIR, filename)
            name = os.path.splitext(filename)[0]
            with open(filepath, "rb") as f:
                embedding = pickle.load(f)
                embeddings[name] = embedding
    return embeddings


def delete_embedding(name):
    filepath = os.path.join(DB_DIR, f"{name}.pkl")
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"Đã xóa user {name} tại {filepath}")
            return True
        else:
            print(f"Không tìm thấy user {name} để xóa.")
            return False
    except Exception as e:
        print(f"Lỗi khi xóa user {name}: {e}")
        return False
