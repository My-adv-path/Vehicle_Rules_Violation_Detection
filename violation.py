import cv2 as cv
from ultralytics import YOLO
import pandas as pd
from tracker import Tracker
import time
import os
import easyocr
import re
import numpy as np

# ─── Directory Setup ───────────────────────────────────────────────────────────
os.makedirs("violation", exist_ok=True)
os.makedirs("plates", exist_ok=True)

# ─── Model Loading ─────────────────────────────────────────────────────────────
vehicle_model = YOLO('yolov8n.pt')
plate_model = YOLO("best_new.pt")

# ─── EasyOCR Reader ────────────────────────────────────────────────────────────
reader = easyocr.Reader(['en'], gpu=False)

# ─── Config ────────────────────────────────────────────────────────────────────
class_list = ["bus", "car", "motorcycle", "truck", "bicycle"]
tracker = Tracker()

video = cv.VideoCapture(r"C:\Users\Asus\Downloads\Vehicle Detection\test1_updated.mp4")
frame_count = 0

signal_state = "RED"
red_time = 10
green_time = 10
last_switch_time = time.time()
pad_x = 20
pad_y = 60

if not video.isOpened():
    print("Video cannot be accessed")
    exit()


# ─── Plate Preprocessing ───────────────────────────────────────────────────────
def preprocess_plate(plate_crop):
    """
    Returns a list of preprocessed grayscale images to try OCR on.
    Applies sharpening + CLAHE to handle blurry CCTV input.
    """
    # Aggressively upscale for blurry input (3x instead of 2x)
    plate_crop = cv.resize(plate_crop, None, fx=3, fy=3, interpolation=cv.INTER_CUBIC)

    gray = cv.cvtColor(plate_crop, cv.COLOR_BGR2GRAY)

    # 1. Sharpening kernel — critical fix for blurred video
    sharpen_kernel = np.array([[-1, -1, -1],
                                [-1,  9, -1],
                                [-1, -1, -1]])
    sharpened = cv.filter2D(gray, -1, sharpen_kernel)

    # 2. CLAHE — better than equalizeHist for license plates
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(sharpened)

    # 3. Otsu on sharpened
    _, img_otsu = cv.threshold(sharpened, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # 4. CLAHE + Otsu
    _, img_clahe_otsu = cv.threshold(clahe_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # 5. Adaptive threshold on CLAHE image
    img_adaptive = cv.adaptiveThreshold(
        clahe_img, 255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY, 13, 3
    )

    # 6. Denoised + Otsu
    denoised = cv.fastNlMeansDenoising(sharpened, h=20)
    _, img_denoised = cv.threshold(denoised, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    return [sharpened, img_otsu, img_clahe_otsu, img_adaptive, img_denoised]

# ─── OCR on Plate ──────────────────────────────────────────────────────────────
def run_ocr_on_plate(plate_crop):
    """
    Runs EasyOCR across multiple preprocessed versions of the plate image.
    Stitches fragmented text detections and matches Indian plate pattern.
    Returns best matched plate text or best guess fallback.
    """
    images = preprocess_plate(plate_crop)
    pattern = r'^[A-Z]{2}\s?[0-9]{2}\s?[A-Z]{1,2}\s?[0-9]{3,4}$'

    plate_text = None
    best_conf = 0
    fallback_text = None
    fallback_conf = 0

    for idx, image in enumerate(images):
        ocr_results = reader.readtext(
            image,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            detail=1,
            paragraph=False,
            contrast_ths=0.3,      # Lower = picks up low-contrast blurry plates
            adjust_contrast=0.7,   # Auto contrast boost
            text_threshold=0.6,    # Lower threshold for blurry text
            low_text=0.3,
            width_ths=0.8
        )

        if not ocr_results:
            continue

        # Stitch all detected fragments together (handles split detections)
        full_text = "".join([res[1].replace(" ", "").upper() for res in ocr_results])
        avg_conf = float(np.mean([res[2] for res in ocr_results]))

        print(f"[Image {idx+1}] Stitched OCR: '{full_text}' | Confidence: {avg_conf:.2f}")

        if avg_conf > best_conf:
            if re.search(pattern, full_text):
                plate_text = full_text
                best_conf = avg_conf

        # Keep best guess even if pattern doesn't match (fallback)
        if avg_conf > fallback_conf:
            fallback_text = full_text
            fallback_conf = avg_conf

    # Return matched plate or fallback best guess
    if plate_text:
        return plate_text
    elif fallback_text:
        print(f"[WARN] OCR: '{fallback_text}'")
        return fallback_text
    return None


# ─── Main Loop ─────────────────────────────────────────────────────────────────
while True:
    ret, frame = video.read()

    if not ret:
        print("No more frames exist...exiting")
        break

    frame = cv.resize(frame, (1000, 600))
    frame_count += 1

    # Process every 3rd frame only
    if frame_count % 3 != 0:
        continue

    stop_line_y = 450
    vehicle_ids = [1, 2, 3, 5, 7]

    # ── Vehicle Detection ──────────────────────────────────────────────────────
    result = vehicle_model.predict(frame, classes=vehicle_ids)
    a = result[0].boxes.data.detach().cpu().numpy()

    boxes = []
    conf_scores = []
    filtered_boxes = []

    for (x1, y1, x2, y2, conf, classid) in a:
        boxes.append([int(x1), int(y1), int(x2), int(y2)])
        conf_scores.append(float(conf))

    if boxes:
        indices = cv.dnn.NMSBoxes(
            boxes, conf_scores,
            score_threshold=0.3,
            nms_threshold=0.5
        )
        for i in indices:
            i = i[0] if isinstance(i, (list, tuple)) else i
            x1, y1, x2, y2 = boxes[i]
            filtered_boxes.append([x1, y1, x2, y2])

    bbox_id = tracker.update(filtered_boxes)

    # ── Draw Stop Line ─────────────────────────────────────────────────────────
    cv.line(frame, (240, stop_line_y), (1150, stop_line_y), (0, 255, 255), 2)

    # ── Signal Timer Logic ─────────────────────────────────────────────────────
    current_time = time.time()
    if signal_state == "RED":
        cv.circle(frame, (15, 15), 15, (0, 0, 255), -1)
        if current_time - last_switch_time >= red_time:
            signal_state = "GREEN"
            last_switch_time = current_time
    elif signal_state == "GREEN":
        cv.circle(frame, (15, 15), 15, (0, 255, 0), -1)
        if current_time - last_switch_time >= green_time:
            signal_state = "RED"
            last_switch_time = current_time

    print(f"Signal: {signal_state}")

    # ── Per-Vehicle Processing ─────────────────────────────────────────────────
    for bbox in bbox_id:
        id = bbox[-1]
        cx, cy = tracker.objects[id]
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

        # Padded crop
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(frame.shape[1], x2 + pad_x)
        y2 = min(frame.shape[0], y2 + pad_y)

        prev_cx, prev_cy = tracker.prev_positions[id]

        cv.circle(frame, (cx, cy), 2, (0, 0, 255), -1)
        cv.putText(frame, str(id), (cx, cy), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # ── Violation Detection ────────────────────────────────────────────────
        if 330 < cx < 1190:
            if prev_cy < stop_line_y and cy > stop_line_y:
                if signal_state == "RED":
                    vehicle_crop = frame[y1:y2, x1:x2]

                    if vehicle_crop.size == 0:
                        print(f"[WARN] Empty vehicle crop for ID {id}")
                        continue

                    # Use bottom 60% of vehicle for plate region
                    h, w = vehicle_crop.shape[:2]
                    vehicle_crop = vehicle_crop[int(h * 0.4):h, :]

                    timestamp = int(time.time())
                    cv.imwrite(f"violation/vehicle_{timestamp}.jpg", vehicle_crop)
                    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                    # ── Plate Detection ───────────────────────────────────────
                    plate_results = plate_model.predict(vehicle_crop, conf=0.15)
                    plate_crop = None

                    for p in plate_results[0].boxes.data:
                        px1, py1, px2, py2, conf, cls = p
                        px1, py1, px2, py2 = map(int, [px1, py1, px2, py2])
                        plate_crop = vehicle_crop[py1:py2, px1:px2]

                    if plate_crop is None or plate_crop.size == 0:
                        print(f"[WARN] No plate detected for vehicle ID {id}")
                        continue

                    # Save raw plate before processing
                    cv.imwrite(f"plates/plate_{timestamp}.jpg", plate_crop)

                    # ── OCR ───────────────────────────────────────────────────
                    plate_text = run_ocr_on_plate(plate_crop)

                    if plate_text:
                        print(f"[VIOLATION] Vehicle ID {id} | Plate: {plate_text}")
                        cv.putText(
                            frame, plate_text,
                            (x1, y1 - 10),
                            cv.FONT_HERSHEY_SIMPLEX,
                            0.4, (0, 0, 255), 1
                        )
                    else:
                        print(f"[VIOLATION] Vehicle ID {id} | Plate: UNREADABLE")

    # ── Display ────────────────────────────────────────────────────────────────
    cv.imshow("frame", frame)
    if cv.waitKey(30) == ord("s"):
        break

video.release()
cv.destroyAllWindows()


        
