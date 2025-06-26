# 🧠 Deepfake Detection System

A deep learning-powered system to detect manipulated or fake (deepfake) images and videos using a CNN model (ResNet18). This project extracts frames from video/image input, performs classification, visualizes results using Grad-CAM, and generates detailed reports in PDF format.

## 🔧 Features

- Detects real vs fake faces in videos and images
- Frame-by-frame prediction using ResNet18 (PyTorch)
- Grad-CAM visual heatmaps for model explainability
- Automatically generates a downloadable PDF report
- Clean frontend built in ReactJS
- Flask backend API for inference and processing

---

## 🚀 Tech Stack

| Layer         | Technology                          |
|---------------|--------------------------------------|
| **Frontend**  | ReactJS, TailwindCSS, Axios          |
| **Backend**   | Python, Flask, OpenCV, ReportLab     |
| **ML Model**  | PyTorch, ResNet18, Grad-CAM          |

---

## 📁 Folder Structure

Deepfake/
├── deepfake frontend/ # React-based frontend for UI and file upload
├── model/ # Trained model weights
├── server.py # Flask backend API
├── utils.py # Grad-CAM, PDF generation, frame processing
├── requirements.txt # Backend Python dependencies



---

## 🧪 How to Run

### 1. Backend (Flask)

```bash
cd Deepfake
pip install -r requirements.txt
python server.py
2. Frontend (ReactJS)
bash
Copy
Edit
cd "deepfake frontend"
npm install
npm start
Then open http://localhost:3000 to access the app.

🎓 Dataset Used
FaceForensics++

DeepFake Detection Challenge (Kaggle)

