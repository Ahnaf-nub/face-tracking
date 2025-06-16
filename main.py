import os
import cv2
import numpy as np
import onnxruntime as ort
import urllib.request
from ultralytics import YOLO

# Download MiDaS ONNX model if missing
midas_model_path = "midas_small.onnx"
midas_model_url = "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx"
if not os.path.exists(midas_model_path):
    print(f"{midas_model_path} not found. Downloading from {midas_model_url}...")
    urllib.request.urlretrieve(midas_model_url, midas_model_path)
    print("MiDaS model download complete.")

# Load MiDaS ONNX model
midas_sess = ort.InferenceSession(midas_model_path, providers=["CPUExecutionProvider"])

# Load YOLOv5n model via ultralytics package (auto-downloads yolov5n.pt)
yolo_model = YOLO("yolov5n.pt")

def prep_midas(frame, size=256):
    img = cv2.resize(frame, (size, size)).astype(np.float32)
    img /= 255.0
    img -= np.array([0.485, 0.456, 0.406], dtype=np.float32)[None, None, :]
    img /= np.array([0.229, 0.224, 0.225], dtype=np.float32)[None, None, :]
    img = img.transpose(2, 0, 1)[None, ...].astype(np.float32)
    return img

def decide_steering(obstacles, frame_w):
    L, C, R = 0, 0, 0  # left, center, right counts

    for ob in obstacles:
        x1, x2, d = ob['x1'], ob['x2'], ob['depth']
        zone = (x1 + x2) / 2

        if d < 3.0:  # threshold in meters
            if zone < frame_w / 3:
                L += 1
            elif zone < 2 * frame_w / 3:
                C += 1
            else:
                R += 1

    print(f"Obstacles - L:{L} C:{C} R:{R}")

    if C > 0:
        if L <= R:
            return "TURN LEFT"
        else:
            return "TURN RIGHT"
    elif L > R:
        return "TURN RIGHT"
    elif R > L:
        return "TURN LEFT"
    else:
        return "GO FORWARD"

cap = cv2.VideoCapture(0)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_h, frame_w = frame.shape[:2]

    # YOLO detection with Ultralytics (BGR input)
    results = yolo_model(frame)

    # MiDaS depth every 3 frames
    if frame_count % 3 == 0:
        midas_inp = prep_midas(frame)
        depth_map_raw = midas_sess.run(None, {midas_sess.get_inputs()[0].name: midas_inp})[0][0]
        depth_map_raw = cv2.resize(depth_map_raw, (frame_w, frame_h))

        # Normalize depth to 0-1
        depth_min = depth_map_raw.min()
        depth_max = depth_map_raw.max()
        depth_norm = (depth_map_raw - depth_min) / (depth_max - depth_min + 1e-6)

        # Invert so closer = smaller distance
        depth_map = 1.0 - depth_norm

        # Scale depth to meters (adjust if needed)
        scale_factor = 10.0
        depth_map *= scale_factor

    obstacles = []

    # Parse YOLO detections
    detections = results[0].boxes
    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())

        if conf < 0.25:
            continue

        roi_depth = depth_map[y1:y2, x1:x2]
        est_depth = np.median(roi_depth) if roi_depth.size else 10.0

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"D={est_depth:.2f}m", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        obstacles.append({'x1': x1, 'x2': x2, 'depth': est_depth})

    move = decide_steering(obstacles, frame_w)
    print(f"DECISION: {move}")
    cv2.putText(frame, f"ACTION: {move}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    depth_vis = cv2.applyColorMap(cv2.convertScaleAbs(depth_map / np.max(depth_map), 255), cv2.COLORMAP_TURBO)
    overlay = cv2.addWeighted(frame, 0.6, depth_vis, 0.4, 0)

    cv2.imshow("AutoNav Pi5", overlay)
    if cv2.waitKey(1) == 27:
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

