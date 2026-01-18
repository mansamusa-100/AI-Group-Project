import cv2
import time
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

video_path = "Cars Moving On Road.mp4"
cap = cv2.VideoCapture(video_path)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    "output_Cars_Moving_On_Road_tracking.avi",
    cv2.VideoWriter_fourcc(*"XVID"),
    fps,
    (w, h)
)

count_line_y = int(h * 0.6)
counted_ids = set()
vehicle_count = 0

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        classes=[2, 3, 5, 7],  # car, motorbike, bus, truck
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

    curr_time = time.time()
    fps_display = int(1 / (curr_time - prev_time))
    prev_time = curr_time

    if vehicle_count < 10:
        traffic_status = "LOW"
    elif vehicle_count < 25:
        traffic_status = "MEDIUM"
    else:
        traffic_status = "HIGH"

    cv2.line(frame, (0, count_line_y), (w, count_line_y), (0, 0, 255), 2)
    cv2.putText(frame, f"Vehicles: {vehicle_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, f"Traffic: {traffic_status}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"FPS: {fps_display}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    out.write(frame)

cap.release()
out.release()

print("Processing complete.")
print("Total vehicles counted:", vehicle_count)
