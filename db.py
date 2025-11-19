import os
import pickle
import numpy as np
import csv
from datetime import datetime
import pytz
import pandas as pd
from filelock import FileLock
from functools import lru_cache

# Thư mục lưu dữ liệu khuôn mặt
DB_DIR = "face_db"
# Tên file log điểm danh
LOG_FILE = "attendance_log.csv"

LOG_HEADER = [
    "timestamp",
    "name_detected",
    "mssv",
    "class_name",
    "action",
    "similarity_score",
    "spoof_score",
    "emotion",
]

# Tạo thư mục nếu chưa có
os.makedirs(DB_DIR, exist_ok=True)


def initialize_log_file():
    """Tạo file log nếu chưa tồn tại."""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(LOG_HEADER)


def log_attendance(name, mssv, class_name, action, score, spoof_score, emotion):
    """Ghi log điểm danh vào CSV với file locking."""
    initialize_log_file()
    vn_tz = pytz.timezone("Asia/Ho_Chi_Minh")
    timestamp = datetime.now(vn_tz).strftime("%Y-%m-%d %H:%M:%S")

    # Use file lock to prevent concurrent write corruption
    lock = FileLock(LOG_FILE + ".lock", timeout=10)
    try:
        with lock:
            # Mở file mode 'a' (append) để ghi nối tiếp
            with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        timestamp,
                        name,
                        mssv,
                        class_name,
                        action,
                        f"{score:.2f}",
                        f"{spoof_score:.3f}",
                        emotion,
                    ]
                )
                # Đảm bảo dữ liệu được ghi ngay lập tức xuống ổ cứng
                f.flush()
                os.fsync(f.fileno())
    except Exception as e:
        print(f"❌ Lỗi ghi log: {e}")
        return

    print(
        f"✅ Logged: {name} ({mssv}) - {action} - Cos: {score:.2f} - Spoof: {spoof_score:.3f} - Emotion: {emotion}"
    )


def get_logs():
    """Đọc toàn bộ log lên DataFrame."""
    if not os.path.exists(LOG_FILE):
        initialize_log_file()
        return pd.DataFrame(columns=LOG_HEADER)
    try:
        # Đọc file CSV, bỏ qua các dòng lỗi, parse timestamp ngay khi đọc
        df = pd.read_csv(
            LOG_FILE,
            on_bad_lines="skip",
            parse_dates=[
                "timestamp"
            ],  # Parse timestamp during read for better performance
            date_format="%Y-%m-%d %H:%M:%S",
        )
        if df.empty:
            return pd.DataFrame(columns=LOG_HEADER)

        # Sắp xếp giảm dần theo thời gian (Mới nhất lên đầu) để hiển thị đẹp
        df.sort_values(by="timestamp", ascending=False, inplace=True)
        return df
    except Exception as e:
        print(f"❌ Lỗi đọc logs: {e}")
        return pd.DataFrame(columns=LOG_HEADER)


def get_last_action(name):
    """Lấy trạng thái cuối cùng (Check-in/Check-out) của user trong ngày."""
    if not os.path.exists(LOG_FILE):
        return None

    try:
        df = pd.read_csv(LOG_FILE, on_bad_lines="skip")
    except:
        return None

    if df.empty or "action" not in df.columns:
        return None

    vn_tz = pytz.timezone("Asia/Ho_Chi_Minh")
    today_str = datetime.now(vn_tz).strftime("%Y-%m-%d")

    # Lọc log của user trong ngày hôm nay
    user_logs = df[
        (df["name_detected"] == name)
        & (df["timestamp"].astype(str).str.startswith(today_str))
    ]

    if user_logs.empty:
        return None

    # [QUAN TRỌNG] Lấy dòng cuối cùng (mới nhất) thay vì dòng đầu tiên
    return user_logs.iloc[-1]["action"]


def save_user_data(name, mssv, class_name, embedding):
    """Lưu dữ liệu người dùng (Embedding + Info)."""
    filepath = os.path.join(DB_DIR, f"{name}.pkl")

    # Chuẩn hóa vector embedding trước khi lưu
    if embedding is not None:
        embedding = embedding / np.linalg.norm(embedding)

    user_data = {"embedding": embedding, "mssv": mssv, "class_name": class_name}

    with open(filepath, "wb") as f:
        pickle.dump(user_data, f)
    print(f"💾 Saved data for: {name}")


def load_embeddings():
    """Load tất cả embedding lên RAM để nhận diện."""
    embeddings = {}
    if not os.path.exists(DB_DIR):
        return embeddings

    for filename in os.listdir(DB_DIR):
        if filename.endswith(".pkl"):
            name = os.path.splitext(filename)[0]
            filepath = os.path.join(DB_DIR, filename)
            try:
                with open(filepath, "rb") as f:
                    data = pickle.load(f)
                    # Tương thích ngược: Nếu data là dict thì lấy key 'embedding', ngược lại lấy chính nó
                    if isinstance(data, dict):
                        embeddings[name] = data["embedding"]
                    else:
                        embeddings[name] = data
            except:
                pass
    return embeddings


@lru_cache(maxsize=128)
def get_user_info(name):
    """Lấy thông tin MSSV, Lớp. Cached for performance."""
    filepath = os.path.join(DB_DIR, f"{name}.pkl")
    if os.path.exists(filepath):
        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    return data.get("mssv", "N/A"), data.get("class_name", "N/A")
        except:
            pass
    return "N/A", "N/A"


def get_full_user_data(name):
    """Lấy full data (dùng cho tab chỉnh sửa hồ sơ)."""
    filepath = os.path.join(DB_DIR, f"{name}.pkl")
    if os.path.exists(filepath):
        try:
            with open(filepath, "rb") as f:
                return pickle.load(f)
        except:
            pass
    return None


def delete_embedding(name):
    """Xóa dữ liệu người dùng."""
    filepath = os.path.join(DB_DIR, f"{name}.pkl")
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
    except:
        pass
    return False


def count_registered_users():
    """Đếm số người đã đăng ký trong database."""
    if not os.path.exists(DB_DIR):
        return 0

    count = 0
    for filename in os.listdir(DB_DIR):
        if filename.endswith(".pkl"):
            count += 1
    return count


def get_all_user_names():
    """Lấy danh sách tất cả tên người dùng đã đăng ký."""
    if not os.path.exists(DB_DIR):
        return []

    users = []
    for filename in os.listdir(DB_DIR):
        if filename.endswith(".pkl"):
            name = os.path.splitext(filename)[0]
            users.append(name)
    return sorted(users)
