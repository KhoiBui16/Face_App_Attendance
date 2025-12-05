# Face App Attendance 📘
Attendance application using face recognition, combined with anti-spoofing detection and emotion recognition.
---
## 1. System Requirements (Prerequisites)
Before starting, ensure your computer has:
- **Python**: 3.8 – 3.10 (3.10 recommended for TensorFlow)
- **Git**: To clone the source code
- **Git LFS**: To download large model files (very important)
---
## 2. Detailed Installation (Installation)
### Step 1: Clone the project
```bash
git clone https://github.com/KhoiBui16/Face_App_Attendance.git
cd Face_App_Attendance
```
### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate
# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```
### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```
> Note: Check `requirements.txt` to remove any redundant lines if there are copy/paste errors.
### Step 4: Prepare Models
Create a `models/` folder in the root directory, then add these files:
- `ResNet50_feature_extractor.keras` – Face feature extraction model
- `anti_spoof_model.h5` – Anti-spoofing model
- `emotion_model.h5` – Emotion recognition model
> If you just cloned from Git and are using Git LFS, run:
```bash
git lfs pull
```
to download files if they're not complete.
### Step 5: Create login configuration (Optional)
```bash
python generate_keys.py
```
> This command creates a `config.yaml` file containing Admin user information.
---
## 3. Running the Application (Running the App)
```bash
streamlit run app.py
```
The browser will automatically open: [http://localhost:8501](http://localhost:8501)
---
## 4. Project Structure (Project Structure)
```
Face_App_Attendance/
├── app.py                  # [MAIN] Main interface
├── face_processing.py      # [CORE] AI processing: load model, detect face, embedding
├── db.py                   # [DATABASE] Save/Delete user, log CSV
├── generate_keys.py        # [UTIL] Password encryption & create config.yaml
├── requirements.txt        # Required libraries
├── models/                 # [DATA] .keras, .h5 files
│   ├── ResNet50_feature_extractor.keras
│   ├── anti_spoof_model.h5
│   └── emotion_model.h5
├── face_db/                # [DATA] .pkl files containing user embeddings
└── attendance_log.csv      # [LOG] Stores attendance history
```
**Workflow:**
- **Registration:** app.py captures photo → face_processing.py checks Spoof → creates Embedding → db.py saves to `face_db/`
- **Attendance:** app.py captures photo → face_processing.py creates new Embedding → compares Cosine Similarity → returns result + emotion → db.py writes to `attendance_log.csv`
---
## 5. Web Deployment (Deploy)
### Step 1: Prepare GitHub
- Ensure code is pushed to GitHub with **Git LFS**.
- Edit `requirements.txt`:
```
streamlit
tensorflow-cpu
numpy
opencv-python-headless
mtcnn
scikit-learn
pandas
pytz
pyyaml
```
### Step 2: Create `packages.txt` for OpenCV
- Create a `packages.txt` file in the root directory, add:
```
libgl1
```
### Step 3: Deploy on Streamlit Community Cloud
1. Visit [share.streamlit.io](https://share.streamlit.io)
2. Login with GitHub
3. Select **New app** → choose `Face_App_Attendance` repo → `main` branch → main file `app.py` → Deploy
**Notes:**
- If OOM (Out of Memory) occurs due to TensorFlow/ResNet50 → consider using lighter models like MobileNetV2 or deploy on Hugging Face Spaces/Render
- First deployment with Git LFS may be slow to load, please be patient
---
## 6. Additional Notes
- Always track **large files with LFS before commit**
- If old commits contain files >100MB, you need to **rewrite history** to push successfully
- Clone the repo again if using force-push on old history
