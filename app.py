import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

# ---------------- LOAD MODEL ----------------
model = YOLO("yolov8n.pt")

# ---------------- GLOBALS ----------------
heatmap = None
history = []

# ---------------- FUNCTIONS ----------------

def detect_people(frame):
    results = model(frame)
    count = 0
    boxes = []

    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0:  # person class
                count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    return frame, count, boxes


def calculate_density(count, area=50):
    return count / area


def risk_level(density):
    if density < 2:
        return "SAFE"
    elif density < 4:
        return "WARNING"
    else:
        return "DANGER"


def predict_density(current):
    global history
    history.append(current)

    if len(history) > 10:
        history.pop(0)

    avg = sum(history[-3:]) / len(history[-3:])

    if avg > current:
        return "INCREASING RISK"
    return "STABLE"


def update_heatmap(frame, boxes):
    global heatmap

    h, w, _ = frame.shape
    if heatmap is None:
        heatmap = np.zeros((h, w), dtype=np.float32)

    for (x1, y1, x2, y2) in boxes:
        heatmap[y1:y2, x1:x2] += 1

    heatmap_blur = cv2.GaussianBlur(heatmap, (25,25), 0)
    heatmap_norm = cv2.normalize(heatmap_blur, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_color = cv2.applyColorMap(heatmap_norm.astype('uint8'), cv2.COLORMAP_JET)

    return cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)


# ---------------- STREAMLIT UI ----------------

st.title("🚨 Crowd Density & Stampede Prevention AI")

video_source = st.sidebar.selectbox(
    "Choose Input Source",
    ["Video File", "Webcam"]
)

if video_source == "Video File":
    video_path = st.sidebar.text_input("Enter video path", "videos/crowd.mp4")
    cap = cv2.VideoCapture(video_path)
else:
    cap = cv2.VideoCapture(0)

frame_window = st.image([])
info_box = st.empty()

# ---------------- MAIN LOOP ----------------

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.warning("Video ended or cannot read")
        break

    # Detection
    frame, count, boxes = detect_people(frame)

    # Density
    density = calculate_density(count)
    risk = risk_level(density)
    prediction = predict_density(density)

    # Heatmap
    frame = update_heatmap(frame, boxes)

    # Overlay Text
    cv2.putText(frame, f"People: {count}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.putText(frame, f"Density: {density:.2f}", (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.putText(frame, f"Risk: {risk}", (20,120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.putText(frame, f"Prediction: {prediction}", (20,160),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    # Alerts
    if risk == "DANGER":
        alert = "🚨 CRITICAL ALERT: HIGH CROWD DENSITY!"
    elif risk == "WARNING":
        alert = "⚠️ WARNING: Monitor crowd"
    else:
        alert = "✅ SAFE"

    # Show frame
    frame_window.image(frame, channels="BGR")

    # Show metrics
    info_box.markdown(f"""
    ### 📊 Live Stats
    - 👥 People Count: **{count}**
    - 📏 Density: **{density:.2f}**
    - ⚠️ Risk Level: **{risk}**
    - 🔮 Prediction: **{prediction}**
    - 🚨 Alert: **{alert}**
    """)

cap.release()
