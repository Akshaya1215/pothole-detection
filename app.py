import gradio as gr
from ultralytics import YOLO
import cv2

# Load trained YOLO model
model = YOLO("best.pt")  # load from current directory

def detect_pothole(image):
    results = model.predict(image, conf=0.4)
    annotated = results[0].plot()    # Draw bounding boxes
    return annotated                 # Return annotated image to Gradio

# Create Gradio Interface
demo = gr.Interface(
    fn=detect_pothole,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Image(type="numpy"),
    title="Road Pothole Detection System",
    description="Upload a road image and the model detects potholes"
)

demo.launch()
