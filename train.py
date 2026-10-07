from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=7,
    imgsz=640,
    batch=16,
    workers=4,
    name="pothole_yolo"
)

print("Training completed successfully!")






