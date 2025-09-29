# --- Step 1: Import Necessary Libraries ---
import cv2
from ultralytics import YOLO
import time
import numpy as np
from collections import defaultdict
import random
from src.vision.tracker import Sort

# --- Global list to store calibration click points ---
calibration_points = []

# --- Mouse callback function for calibration ---
def mouse_callback(event, x, y, flags, param):
    """
    This function is called on every mouse event.
    It now stores the click points and provides visual feedback.
    """
    global calibration_points
    if event == cv2.EVENT_LBUTTONDOWN:
        calibration_points.append((x, y))
        print(f"Point {len(calibration_points)} captured at - X: {x}, Y: {y}")

# --- RF Sensor Simulation ---
def get_simulated_rf_metrics(current_person_count):
    if current_person_count == 0:
        return random.randint(1, 5)
    else:
        base_count = current_person_count * random.uniform(2.0, 4.0)
        noise = random.randint(-2, 2)
        return max(0, int(base_count + noise))

# --- UPGRADED: Height Classification Function with Linear Interpolation ---
def classify_by_height(bbox):
    """
    Classifies a bounding box using a more accurate linear interpolation
    to account for perspective across the entire frame.
    """
    y2 = bbox[3] # Y-coordinate of the feet
    box_height = bbox[3] - bbox[1]

    # --- CALIBRATION PARAMETERS (Update these with your measured values) ---
    CLOSE_LINE_Y = 400 
    ADULT_HEIGHT_THRESHOLD_CLOSE = 250
    FAR_LINE_Y = 250
    ADULT_HEIGHT_THRESHOLD_FAR = 120
    # --------------------------------------------------------------------

    # Handle cases where the person is outside our calibrated range
    if y2 > CLOSE_LINE_Y:
        return "Adult" if box_height > ADULT_HEIGHT_THRESHOLD_CLOSE else "Child"
    if y2 < FAR_LINE_Y:
        return "Adult" if box_height > ADULT_HEIGHT_THRESHOLD_FAR else "Child"

    # --- Linear Interpolation ---
    # Calculate how far the person is between the far and close lines (0.0 to 1.0)
    # Add a small epsilon to avoid division by zero if lines are the same
    epsilon = 1e-6 
    position_ratio = (y2 - FAR_LINE_Y) / (CLOSE_LINE_Y - FAR_LINE_Y + epsilon)

    # Calculate the expected adult height at this exact position
    expected_adult_height = ADULT_HEIGHT_THRESHOLD_FAR + position_ratio * (ADULT_HEIGHT_THRESHOLD_CLOSE - ADULT_HEIGHT_THRESHOLD_FAR)

    # Classify based on the calculated threshold
    if box_height > expected_adult_height:
        return "Adult"
    else:
        return "Child"

# --- Main Application Function ---
def main():
    print("INFO: Starting Unified Pipeline Simulation...")

    # --- Initialize Models and Tracker ---
    try:
        model = YOLO("yolov8n.pt")
        print("INFO: YOLOv8 person detector loaded successfully.")
        tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
        print("INFO: SORT tracker initialized.")
    except Exception as e:
        print(f"ERROR: Failed to load base models. {e}")
        return

    # --- Dictionaries for Dwell Time ---
    dwell_start_times = defaultdict(lambda: 0)
    dwell_times = defaultdict(lambda: 0)

    # --- Video Source Setup ---
    video_source = "cv_pipeline/test_video.mp4"
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"ERROR: Could not open video source '{video_source}'.")
        return
    print(f"INFO: Video source '{video_source}' opened. Starting inference loop...")
    print("INFO: Press 'q' in the display window to quit.")

    # --- Reporting Timer Setup ---
    aggregation_interval_sec = 15
    last_aggregation_time = time.time()

    # --- THE MAIN LOOP ---
    while True:
        # --- CALIBRATION LOGIC ---
        if len(calibration_points) == 4:
            close_feet_y = calibration_points[0][1]
            close_head_y = calibration_points[1][1]
            far_feet_y = calibration_points[2][1]
            far_head_y = calibration_points[3][1]

            close_height = abs(close_feet_y - close_head_y)
            far_height = abs(far_feet_y - far_head_y)

            print("\n--- CALIBRATION COMPLETE ---")
            print("Please copy these values into the 'classify_by_height' function:")
            print(f"CLOSE_LINE_Y = {close_feet_y}")
            print(f"ADULT_HEIGHT_THRESHOLD_CLOSE = {close_height}")
            print(f"FAR_LINE_Y = {far_feet_y}")
            print(f"ADULT_HEIGHT_THRESHOLD_FAR = {far_height}")
            print("----------------------------\n")
            
            cv2.waitKey(0)
            break
        # --- END OF CALIBRATION LOGIC ---

        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        current_frame_time = time.time()
        
        results = model(frame, classes=[0], verbose=False)
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu.numpy()
            detections.append([x1, y1, x2, y2, conf])
        
        detections_np = np.array(detections)
        if len(detections_np) > 0:
            tracked_objects = tracker.update(detections_np)
        else:
            tracked_objects = tracker.update()
        
        annotated_frame = frame.copy()
        person_count = len(tracked_objects)
        
        current_track_ids = {int(t[4]) for t in tracked_objects}
        for track_id in current_track_ids:
            if dwell_start_times[track_id] == 0:
                dwell_start_times[track_id] = current_frame_time
        for track_id in current_track_ids:
            dwell_times[track_id] = current_frame_time - dwell_start_times[track_id]
        lost_track_ids = [tid for tid in dwell_start_times if tid not in current_track_ids]
        for track_id in lost_track_ids:
            del dwell_start_times[track_id]
            del dwell_times[track_id]

        demographic_counts = defaultdict(lambda: 0) 

        for point in calibration_points:
            cv2.circle(annotated_frame, point, 5, (0, 0, 255), -1)

        for track in tracked_objects:
            x1, y1, x2, y2, track_id = track.astype(int)
            
            bbox = [x1, y1, x2, y2]
            height_class = classify_by_height(bbox)
            demographic_counts[height_class] += 1
            label = f"ID: {track_id} | {height_class}" 

            dwell_time = dwell_times.get(track_id, 0.0)
            final_label = f"{label} ({dwell_time:.1f}s)"

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, final_label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if time.time() - last_aggregation_time >= aggregation_interval_sec:
            avg_dwell_time = sum(dwell_times.values()) / len(dwell_times) if dwell_times else 0.0
            rf_device_count = get_simulated_rf_metrics(person_count)
            output_data = {
                "metrics": {
                    "cv": { "people_count": person_count, "avg_dwell_time_sec": round(avg_dwell_time, 2) },
                    "rf": { "proximate_device_count": rf_device_count },
                    "demographics": dict(demographic_counts)
                }
            }
            print("\n--- UNIFIED AGGREGATED REPORT ---")
            print(output_data)
            print("---------------------------------\n")
            last_aggregation_time = time.time()

        cv2.namedWindow("DOOH CV Simulation")
        cv2.setMouseCallback("DOOH CV Simulation", mouse_callback)
        cv2.imshow("DOOH CV Simulation", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print("INFO: Shutting down pipeline...")
    cap.release()
    cv2.destroyAllWindows()
    print("INFO: Pipeline finished successfully.")

if __name__ == "__main__":
    main()