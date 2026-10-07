from ultralytics import YOLO

# Load model only once when Django starts
import os

# Find best.pt relative to this file
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, "best.pt")

model = YOLO(model_path)
