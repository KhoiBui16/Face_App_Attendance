"""
Configuration file for Face Recognition Attendance System
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DB_DIR = BASE_DIR / "face_db"

# Model configuration
MODEL_PATH = os.getenv(
    "MODEL_PATH", str(MODELS_DIR / "EfficientNetB4_feature_extractor_ver_2.keras")
)
EMBEDDING_LAYER_NAME = "cnn_embedding"
IMG_SIZE = (224, 224)

# Recognition thresholds
COSINE_THRESHOLD = float(os.getenv("COSINE_THRESHOLD", "0.6"))

# Anti-spoofing configuration
SPOOF_MODEL_PATH = os.getenv(
    "SPOOF_MODEL_PATH", str(MODELS_DIR / "ResNet50_antispoof_finetune.keras")
)
SPOOF_IMG_SIZE = (224, 224)
SPOOF_THRESHOLD = float(
    os.getenv("SPOOF_THRESHOLD", "0.35")
)  # Real face if score > 0.35 (tuned from real-world data)

# Emotion detection configuration
EMOTION_MODEL_PATH = os.getenv(
    "EMOTION_MODEL_PATH", str(MODELS_DIR / "ResNet50_emotion_detect.keras")
)
EMOTION_IMG_SIZE = (224, 224)

# Emotion labels mapping (index 0-7 from model output)
EMOTION_LABELS = [
    "Anger",  # 0
    "Disgust",  # 1
    "Fear",  # 2
    "Happy",  # 3
    "Sadness",  # 4
    "Surprise",  # 5
    "Neutral",  # 6
    "Contempt",  # 7
]

EMOTION_ICONS = {
    "Anger": "\U0001f620",  # 😠
    "Disgust": "\U0001f922",  # 🤢
    "Fear": "\U0001f628",  # 😨
    "Happy": "\U0001f60a",  # 😊
    "Sadness": "\U0001f622",  # 😢
    "Surprise": "\U0001f62e",  # 😮
    "Neutral": "\U0001f610",  # 😐
    "Contempt": "\U0001f612",  # 😒
}

# Database configuration
DB_DIR_PATH = str(DB_DIR)
LOG_FILE = str(BASE_DIR / "attendance_log.csv")
LOG_HEADER = [
    "timestamp",
    "name_detected",
    "mssv",
    "class_name",
    "action",
    "similarity_score",
    "emotion",
]

# Camera configuration
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "1280"))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "720"))
CAMERA_FPS = int(os.getenv("CAMERA_FPS", "30"))
PROCESS_EVERY_N_FRAMES = int(os.getenv("FRAME_SKIP", "3"))
CONSECUTIVE_MATCH_THRESHOLD = int(os.getenv("MATCH_THRESHOLD", "3"))

# Display configuration
DISPLAY_HEIGHT = int(os.getenv("DISPLAY_HEIGHT", "480"))  # Fixed display height

# Face detection configuration
MIN_FACE_SIZE = int(os.getenv("MIN_FACE_SIZE", "60"))
FACE_MARGIN = float(os.getenv("FACE_MARGIN", "0.2"))
DETECTION_RESIZE_WIDTH = int(os.getenv("DETECTION_WIDTH", "640"))

# Cache configuration
LRU_CACHE_SIZE = int(os.getenv("CACHE_SIZE", "128"))

# Logging
ENABLE_DEBUG_LOGGING = os.getenv("DEBUG", "False").lower() == "true"
