import pickle
from pathlib import Path
import streamlit_authenticator as stauth
import yaml


# ---------------------------------------------
# USERNAME & PASSWORD (ĐÃ CHỈNH SỬA)
# ---------------------------------------------
usernames = ["Admin"]  # <-- Tên đăng nhập bạn muốn (viết hoa)
passwords = ["admin123"]  # <-- Mật khẩu

print("Đang mã hóa mật khẩu...")


# ---------------------------------------------
# HASH PASSWORD (API MỚI)
# ---------------------------------------------
hasher = stauth.Hasher()
hashed_passwords = [hasher.hash(password) for password in passwords]


print("Mã hóa thành công!")


# ---------------------------------------------
# TẠO CONFIG (ĐÃ SỬA)
# ---------------------------------------------
config_data = {
    "credentials": {"usernames": {}},
    "cookie": {
        "expiry_days": 30,
        "key": "some_random_secret_key_123",
        "name": "face_app_cookie",
    },
}


# --- Gán user + hashed pass (ĐÃ CHỈNH SỬA) ---
for username, hashed_password in zip(usernames, hashed_passwords):
    config_data["credentials"]["usernames"][username] = {
        "email": f"{username.lower()}@example.com",  # Email vẫn viết thường
        "name": username,  # Tên hiển thị bây giờ cũng là "Admin"
        "password": hashed_password,
    }


# ---------------------------------------------
# LƯU FILE CONFIG.YAML
# ---------------------------------------------
config_path = Path(__file__).parent / "config.yaml"

with config_path.open("w") as file:
    yaml.dump(config_data, file, default_flow_style=False)

print("Đã tạo file 'config.yaml' thành công!")
print("Bạn có thể chạy: streamlit run app.py")
print("---")
print("Nếu quên mật khẩu, chỉ cần chạy lại file này.")
