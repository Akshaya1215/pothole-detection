from ultralytics import YOLO

model = YOLO("best.pt")

video_path = "test_video.mp4"  # change to your video file path

results = model.predict(
    source=video_path,
    conf=0.4,
    save=True,
    project="video_output",
    name="pothole_detection",
    exist_ok=True
)

print("Video detection completed!")
print("Output saved in: video_output/pothole_detection")



