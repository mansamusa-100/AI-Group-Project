import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO

st.set_page_config(page_title="AI Vehicle Detection System", layout="wide")
st.title("🚗 AI-Based Vehicle Detection & Traffic Monitoring")

model = YOLO("yolov8n.pt")

uploaded_video = st.file_uploader("Upload Traffic Video", type=["mp4", "avi"])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile.name)

    count_line_y = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.6)
    counted_ids = set()
    vehicle_count = 0

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            classes=[2, 3, 5, 7],
            conf=0.4
        )

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            ids = results[0].boxes.id.cpu()

            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)
                center_y = int((y1 + y2) / 2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if center_y > count_line_y and int(track_id) not in counted_ids:
                    counted_ids.add(int(track_id))
                    vehicle_count += 1

        if vehicle_count < 10:
            traffic_status = "LOW"
        elif vehicle_count < 25:
            traffic_status = "MEDIUM"
        else:
            traffic_status = "HIGH"

        cv2.putText(frame, f"Vehicles: {vehicle_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.putText(frame, f"Traffic: {traffic_status}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        stframe.image(frame, channels="BGR")

    cap.release()
