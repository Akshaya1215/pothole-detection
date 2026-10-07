# Road Pothole Detection System 🚧

A machine learning-powered system to detect road potholes using **YOLOv8** (You Only Look Once v8) and Python. This repository contains both a **Django web application** (with image upload, video detection, and real-time webcam streaming) and a **Gradio AI dashboard**, plus standalone detection and training scripts.

---

## 🛠️ Features

*   **Django Web Application**:
    *   **Image Detection**: Upload images and view bounding box annotations identifying potholes.
    *   **Video Detection**: Upload video files to automatically identify and mark potholes frame-by-frame.
    *   **Webcam Detection**: Stream video directly from your webcam to detect potholes in real time.
*   **Gradio Dashboard**: A lightweight, standalone web interface (`app.py`) for quick image testing.
*   **Ready-to-Use Weights**: Includes `best.pt` in the root folder, so you can run inference out of the box.
*   **Training & Prediction Scripts**: Simple scripts (`train.py`, `video_detect.py`, `webcam_detection.py`) to customize and retrain the model.

---

## 📂 Project Structure

```
├── pothole_webapp/       # Primary Django Web Application
│   ├── detection/        # Django App handling YOLO inference, templates, and views
│   ├── pothole_webapp/   # Django Project configuration settings and routing
│   └── manage.py         # Django CLI entrypoint
├── pothole_project/      # Secondary Django template / skeleton folder
├── app.py                # Gradio-based Image Inference UI
├── best.pt               # Trained YOLOv8 pothole detection model weights (19.1 MB)
├── data.yaml             # Dataset configuration file pointing to train/val/test splits
├── train.py              # Script to retrain the model on custom images
├── video_detect.py       # Script for running detection on local mp4 files
└── webcam_detection.py   # Script for running live webcam detection via OpenCV
```

---

## 🚀 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/Praveenraju10/Pothole-detection.git
cd Pothole-detection
```

### 2. Install dependencies
Ensure you have Python 3.8+ installed, then install the required packages:
```bash
pip install ultralytics django opencv-python gradio python-multipart
```

### 3. Running the Django Web App 🌐
Navigate to the `pothole_webapp` folder, apply database migrations, and start the development server:
```bash
cd pothole_webapp
python manage.py migrate
python manage.py runserver
```
Open your browser and navigate to `http://127.0.0.1:8000/`. You will find:
*   **Image Detection**: `http://127.0.0.1:8000/`
*   **Video Detection**: `http://127.0.0.1:8000/video/`
*   **Webcam Stream**: `http://127.0.0.1:8000/webcam/`

### 4. Running the Gradio Interface 📊
Run the Gradio dashboard from the project root:
```bash
python app.py
```
This will open a local web page (typically at `http://127.0.0.1:7860/`) where you can upload photos of roads and view detection results.

---

## 📊 Dataset Information

The model was trained on the **Potholes Detection Dataset** from Roboflow.
*   **Dataset URL**: [Roboflow Universe - Potholes Detection](https://universe.roboflow.com/project-ssayl/potholes-detection-d4rma)
*   **Preprocessing**: Resized to 640x640 pixels
*   **Augmentations**: Horizontal flips (with a 50% probability)

To download or train the model again:
1.  Visit the Roboflow link.
2.  Download the dataset in **YOLOv8** format.
3.  Extract the `train`, `valid`, and `test` folders into the root directory of this repository.
4.  Run `python train.py`.

---

## 🛡️ License

This project utilizes a dataset under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.
