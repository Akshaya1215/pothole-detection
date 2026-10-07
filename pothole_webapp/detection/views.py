import cv2
import numpy as np
import base64
from django.http import StreamingHttpResponse
from django.shortcuts import render
from .forms import UploadImageForm, UploadVideoForm
from .yolo_model import model

def detect_pothole(request):
    original_image = None
    result_image = None
    pothole_count = None

    if request.method == "POST":
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            upload = request.FILES['image']

            # Convert uploaded file to numpy array
            file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # Save original image preview
            _, orig_buffer = cv2.imencode(".jpg", image)
            original_image = base64.b64encode(orig_buffer).decode()

            # YOLO detection
            results = model.predict(image, conf=0.4)
            annotated = results[0].plot()

            pothole_count = len(results[0].boxes)

            _, buffer = cv2.imencode(".jpg", annotated)
            result_image = base64.b64encode(buffer).decode()
    else:
        form = UploadImageForm()

    return render(request, "detect.html", {
        "form": form,
        "original_image": original_image,
        "result_image": result_image,
        "pothole_count": pothole_count
    })

def video_detection(request):
    processed_video = None

    if request.method == "POST":
        form = UploadVideoForm(request.POST, request.FILES)
        if form.is_valid():
            video = request.FILES["video"]

            # Save uploaded video temporarily
            input_path = "uploaded_video.mp4"
            with open(input_path, "wb+") as f:
                for chunk in video.chunks():
                    f.write(chunk)

            # Run YOLO video detection
            results = model.predict(
                source=input_path,
                conf=0.4,
                save=True,
                project="video_results",
                name="output",
                exist_ok=True
            )

            # Output video path
            processed_video = "video_results/output/uploaded_video.mp4"

    else:
        form = UploadVideoForm()

    return render(request, "video_detect.html", {
        "form": form,
        "processed_video": processed_video,
    })

from django.http import StreamingHttpResponse
import cv2
from .yolo_model import model

def webcam_stream():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO prediction
        results = model.predict(frame, conf=0.4)
        annotated = results[0].plot()

        # Convert to jpeg
        ret, jpeg = cv2.imencode('.jpg', annotated)
        frame_bytes = jpeg.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

def webcam_detection(request):
    return StreamingHttpResponse(
        webcam_stream(),
        content_type="multipart/x-mixed-replace;boundary=frame"
    )
