from ultralytics import YOLO

# Load YOLO model (use your best.pt path)
import os

# Find best.pt relative to this file to handle run location independence
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_path = os.path.join(base_dir, "best.pt")

model = YOLO(model_path)

