import os
import sys
import json
import time
import logging
import cv2
import mediapipe as mp
import numpy as np

from activity_detector import (
    calculate_ear, estimate_head_pose, detect_hands_joined,
    PhoneDetector, ActivityStateMachine
)
from media_player import MediaPlayer

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")


def load_config(config_path: str = "config.json") -> dict:
    """Load settings from config.json with fallback defaults."""
    default_config = {
        "activities": {
            "eyes_closed": "media/close_eye.mpeg",
            "hands_joined": "media/joinhand.mpeg",
            "phone_detected": "media/phone.mpeg",
            "distracted": None,
            "focused_on_book": None,
            "idle": None
        },
        "ear_threshold": 0.21,
        "eyes_closed_frame_threshold": 15,
        "state_debounce_seconds": 1.5,
        "yaw_distracted_threshold": 25.0,
        "pitch_focused_book_threshold": 15.0,
        "phone_confidence_threshold": 0.4,
        "use_yolo_phone": True
    }

    if not os.path.exists(config_path):
        logging.warning(f"Config file '{config_path}' not found. Creating default config.json.")
        with open(config_path, "w") as f:
            json.dump(default_config, f, indent=2)
        return default_config

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            logging.info(f"Loaded config from '{config_path}'.")
            return config
    except Exception as e:
        logging.error(f"Error reading '{config_path}': {e}. Using default config.")
        return default_config


def draw_debug_overlay(frame, active_state: str, ear_avg: float, yaw: float, pitch: float,
                       eyes_closed_count: int, eyes_closed_max: int, phone_detected: bool,
                       hands_joined: bool, phone_boxes: list, fps: float, show_details: bool = True):
    """Render color-coded activity status banner and real-time sensor overlay on frame."""
    h, w = frame.shape[:2]

    # Color palettes (BGR format)
    state_colors = {
        "eyes_closed": (0, 0, 220),        # Red
        "hands_joined": (211, 0, 148),     # Purple / Magenta (Namaste / Praying)
        "phone_detected": (0, 140, 255),   # Orange
        "distracted": (0, 215, 255),       # Yellow
        "focused_on_book": (255, 144, 30), # Deep Blue/Cyan
        "idle": (50, 205, 50)              # Green
    }

    banner_color = state_colors.get(active_state, (128, 128, 128))

    # Top Status Banner
    cv2.rectangle(frame, (0, 0), (w, 60), banner_color, -1)
    status_text = f"ACTIVITY: {active_state.upper()}"
    cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)

    # Draw Phone Bounding Boxes if detected
    for (bx1, by1, bx2, by2, conf) in phone_boxes:
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 165, 255), 3)
        cv2.putText(frame, f"PHONE {conf:.2f}", (bx1, max(20, by1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    if show_details:
        # Side Info Panel Background
        panel_w = 330
        panel_h = 220
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 70), (10 + panel_w, 70 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Print Sensor Values
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(frame, f"EAR (Eye Aspect Ratio): {ear_avg:.3f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Eyes Closed Progress Bar
        bar_x, bar_y, bar_w, bar_h = 20, 130, 200, 12
        progress = min(1.0, eyes_closed_count / max(1, eyes_closed_max))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), 1)
        fill_w = int(bar_w * progress)
        fill_color = (0, 0, 255) if progress >= 1.0 else (0, 255, 255)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + fill_h), fill_color, -1)
        cv2.putText(frame, f"{eyes_closed_count}/{eyes_closed_max} frames", (bar_x + bar_w + 10, bar_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        cv2.putText(frame, f"Head Yaw (Left/Right): {yaw:.1f} deg", (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Head Pitch (Up/Down):  {pitch:.1f} deg", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Hands Joined (Namaste): {'YES' if hands_joined else 'NO'}", (20, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (211, 0, 148) if hands_joined else (200, 200, 200), 2 if hands_joined else 1)
        cv2.putText(frame, f"Phone Detected: {'YES' if phone_detected else 'NO'}", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 165, 255) if phone_detected else (200, 200, 200), 2 if phone_detected else 1)
        
        cv2.putText(frame, "Keys: [q] Quit  [c] Toggle Stats Panel", (20, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)


class LandmarksWrapper:
    def __init__(self, landmark_list):
        self.landmark = landmark_list


def ensure_task_models():
    """Download MediaPipe task models if not already present."""
    models = {
        "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    }
    import urllib.request
    for filename, url in models.items():
        if not os.path.exists(filename):
            logging.info(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filename)
            logging.info(f"Downloaded {filename}.")


def main():
    config = load_config("config.json")

    # Initialize Modules
    activity_map = config.get("activities", {})
    player = MediaPlayer(activity_map)
    state_machine = ActivityStateMachine(config)
    phone_detector = PhoneDetector(
        use_yolo=config.get("use_yolo_phone", True),
        conf_threshold=config.get("phone_confidence_threshold", 0.4)
    )

    # Initialize MediaPipe Tasks (FaceLandmarker & HandLandmarker)
    ensure_task_models()
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    base_face = mp_python.BaseOptions(model_asset_path='face_landmarker.task')
    options_face = vision.FaceLandmarkerOptions(
        base_options=base_face,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(options_face)

    base_hand = mp_python.BaseOptions(model_asset_path='hand_landmarker.task')
    options_hand = vision.HandLandmarkerOptions(
        base_options=base_hand,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2
    )
    hand_landmarker = vision.HandLandmarker.create_from_options(options_hand)

    # Attempt to open camera across indices and backend APIs
    cap = None
    for cam_idx in [0, 1, 2, 3, -1]:
        for api in ([cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_MSMF] if sys.platform.startswith('win') else [cv2.CAP_ANY]):
            temp_cap = cv2.VideoCapture(cam_idx, api)
            if temp_cap.isOpened():
                ret, test_frame = temp_cap.read()
                if ret and test_frame is not None:
                    cap = temp_cap
                    logging.info(f"Opened webcam at index {cam_idx} (API: {api}).")
                    break
                temp_cap.release()
        if cap is not None:
            break

    if cap is None or not cap.isOpened():
        logging.error("No accessible webcam found! Please ensure your webcam is connected.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    show_details = True
    prev_time = time.time()
    fps = 0.0

    window_name = "Real-Time Activity-Based Media Trigger"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    logging.info("Starting Video Capture Loop... Press 'q' in OpenCV window to stop.")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to grab video frame. Retrying...")
                time.sleep(0.05)
                continue

            curr_time = time.time()
            dt = curr_time - prev_time
            if dt > 0:
                fps = 1.0 / dt
            prev_time = curr_time

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            face_results = face_landmarker.detect(mp_image)
            hand_results = hand_landmarker.detect(mp_image)

            ear_avg, yaw, pitch = 0.0, 0.0, 0.0
            face_detected = False
            face_box = None

            if face_results.face_landmarks:
                face_landmarks = LandmarksWrapper(face_results.face_landmarks[0])
                face_detected = True

                ear_avg, ear_l, ear_r = calculate_ear(face_landmarks, w, h)
                pitch, yaw, roll = estimate_head_pose(face_landmarks, w, h)

                xs = [lm.x * w for lm in face_landmarks.landmark]
                ys = [lm.y * h for lm in face_landmarks.landmark]
                face_box = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))

            hand_landmarks_list = [LandmarksWrapper(h) for h in hand_results.hand_landmarks] if hand_results.hand_landmarks else []

            # Detect joined hands (praying / Namaste gesture)
            hands_joined_flag = detect_hands_joined(hand_landmarks_list)

            # Run Phone Detection
            phone_detected, phone_boxes = phone_detector.detect(
                frame, face_box=face_box, hand_landmarks_list=hand_landmarks_list
            )

            # Update State Machine
            current_state = state_machine.update(
                ear_avg=ear_avg,
                yaw=yaw,
                pitch=pitch,
                phone_detected=phone_detected,
                face_detected=face_detected,
                hands_joined=hands_joined_flag
            )

            # Trigger Audio Playback on State Change
            player.play_for_state(current_state)

            # Render OpenCV Debug Overlay
            draw_debug_overlay(
                frame=frame,
                active_state=current_state,
                ear_avg=ear_avg,
                yaw=yaw,
                pitch=pitch,
                eyes_closed_count=state_machine.eyes_closed_counter,
                eyes_closed_max=state_machine.eyes_closed_frame_threshold,
                phone_detected=phone_detected,
                hands_joined=hands_joined_flag,
                phone_boxes=phone_boxes,
                fps=fps,
                show_details=show_details
            )

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logging.info("Quit key 'q' pressed. Shutting down...")
                break
            elif key == ord('c'):
                show_details = not show_details

    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received. Stopping...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        player.stop()
        face_landmarker.close()
        hand_landmarker.close()
        logging.info("Cleanup complete. Application exited.")


if __name__ == "__main__":
    main()
