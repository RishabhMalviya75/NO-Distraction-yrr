import time
import math
import logging
import numpy as np
import cv2

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

# MediaPipe FaceMesh Landmark Indices for Eye Aspect Ratio (EAR)
RIGHT_EYE_LANDMARKS = {
    'p1': 33,   # Outer corner
    'p4': 133,  # Inner corner
    'p2': 160,  # Top 1
    'p6': 144,  # Bottom 1
    'p3': 158,  # Top 2
    'p5': 153   # Bottom 2
}

LEFT_EYE_LANDMARKS = {
    'p1': 362,  # Inner corner
    'p4': 263,  # Outer corner
    'p2': 385,  # Top 1
    'p6': 380,  # Bottom 1
    'p3': 387,  # Top 2
    'p5': 373   # Bottom 2
}

# 3D Generic Model Points for Head Pose Estimation (solvePnP)
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),             # Nose tip (1)
    (0.0, -330.0, -65.0),        # Chin (152)
    (-225.0, 170.0, -135.0),     # Right Eye outer corner (33)
    (225.0, 170.0, -135.0),      # Left Eye outer corner (263)
    (-150.0, -150.0, -125.0),    # Right Mouth corner (61)
    (150.0, -150.0, -125.0)      # Left Mouth corner (291)
], dtype=np.float64)

HEAD_POSE_LANDMARK_IDS = [1, 152, 33, 263, 61, 291]


def calculate_single_ear(landmarks, eye_indices: dict[str, int], img_w: int, img_h: int) -> float:
    """
    Calculate Eye Aspect Ratio (EAR) for a single eye given facial landmarks.
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    """
    try:
        def get_pt(idx_name):
            lm = landmarks[eye_indices[idx_name]]
            return np.array([lm.x * img_w, lm.y * img_h])

        p1 = get_pt('p1')
        p2 = get_pt('p2')
        p3 = get_pt('p3')
        p4 = get_pt('p4')
        p5 = get_pt('p5')
        p6 = get_pt('p6')

        # Vertical distances
        v1 = np.linalg.norm(p2 - p6)
        v2 = np.linalg.norm(p3 - p5)
        # Horizontal distance
        h = np.linalg.norm(p1 - p4)

        if h < 1e-6:
            return 0.0

        ear = (v1 + v2) / (2.0 * h)
        return float(ear)
    except (IndexError, AttributeError):
        return 0.0


def calculate_ear(face_landmarks, img_w: int, img_h: int) -> tuple[float, float, float]:
    """
    Calculate average EAR along with left and right EAR.
    Returns: (ear_avg, ear_left, ear_right)
    """
    ear_left = calculate_single_ear(face_landmarks.landmark, LEFT_EYE_LANDMARKS, img_w, img_h)
    ear_right = calculate_single_ear(face_landmarks.landmark, RIGHT_EYE_LANDMARKS, img_w, img_h)
    ear_avg = (ear_left + ear_right) / 2.0
    return ear_avg, ear_left, ear_right


def estimate_head_pose(face_landmarks, img_w: int, img_h: int) -> tuple[float, float, float]:
    """
    Estimate Head Pose (Yaw, Pitch, Roll) using 6 facial landmarks via cv2.solvePnP.
    Returns: (pitch, yaw, roll) in degrees.
    """
    image_points = []
    try:
        for lm_id in HEAD_POSE_LANDMARK_IDS:
            lm = face_landmarks.landmark[lm_id]
            image_points.append([lm.x * img_w, lm.y * img_h])
        image_points = np.array(image_points, dtype=np.float64)

        focal_length = img_w
        center = (img_w / 2.0, img_h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return 0.0, 0.0, 0.0

        rmat, _ = cv2.Rodrigues(rvec)
        proj_matrix = np.hstack((rmat, tvec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

        pitch = float(euler_angles[0, 0])
        yaw = float(euler_angles[1, 0])
        roll = float(euler_angles[2, 0])

        return pitch, yaw, roll
    except Exception as e:
        logging.debug(f"Head pose estimation failed: {e}")
        return 0.0, 0.0, 0.0


def detect_hands_joined(hand_landmarks_list) -> bool:
    """
    Detect if user has joined their palms/hands together (praying / Namaste gesture).
    Checks 3D/2D distance between key landmarks of 2 hands.
    """
    if not hand_landmarks_list or len(hand_landmarks_list) < 2:
        return False

    hand1_lms = hand_landmarks_list[0].landmark
    hand2_lms = hand_landmarks_list[1].landmark

    # Key landmark indices: 0 (wrist), 9 (palm center), 8 (index tip), 12 (middle tip)
    check_pairs = [0, 9, 8, 12]
    total_dist = 0.0

    for idx in check_pairs:
        dx = hand1_lms[idx].x - hand2_lms[idx].x
        dy = hand1_lms[idx].y - hand2_lms[idx].y
        total_dist += math.sqrt(dx * dx + dy * dy)

    avg_dist = total_dist / len(check_pairs)
    # Hands are considered joined if average distance between key joints < 0.18
    return avg_dist < 0.18


class PhoneDetector:
    """
    Detects phone presence using YOLOv8 (ultralytics) with class ID 67 ('cell phone').
    Provides fallback heuristic if YOLO fails or is disabled.
    """
    def __init__(self, use_yolo: bool = True, conf_threshold: float = 0.4):
        self.use_yolo = use_yolo
        self.conf_threshold = conf_threshold
        self.model = None

        if self.use_yolo:
            try:
                from ultralytics import YOLO
                logging.info("Initializing YOLOv8n model for phone detection...")
                self.model = YOLO("yolov8n.pt")
                logging.info("YOLOv8n model loaded successfully.")
            except Exception as e:
                logging.warning(f"Could not initialize YOLO model ({e}). Falling back to heuristic phone detection.")
                self.use_yolo = False

    def detect(self, frame, face_box=None, hand_landmarks_list=None) -> tuple[bool, list]:
        """
        Runs object detection on frame.
        Returns: (phone_detected_flag, bounding_boxes_list)
        """
        boxes = []
        if self.use_yolo and self.model is not None:
            try:
                results = self.model(frame, verbose=False, conf=self.conf_threshold)[0]
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls_id == 67 and conf >= self.conf_threshold:
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        boxes.append((xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf))

                if len(boxes) > 0:
                    return True, boxes
            except Exception as e:
                logging.error(f"YOLO inference error: {e}")

        # Fallback Heuristic: Hand landmark near ear/face area
        if hand_landmarks_list and face_box:
            img_h, img_w = frame.shape[:2]
            fx1, fy1, fx2, fy2 = face_box
            margin = int((fx2 - fx1) * 0.4)
            expanded_face_box = (max(0, fx1 - margin), max(0, fy1 - margin), min(img_w, fx2 + margin), min(img_h, fy2 + margin))

            for hand_lms in hand_landmarks_list:
                for lm in hand_lms.landmark:
                    hx, hy = int(lm.x * img_w), int(lm.y * img_h)
                    if (expanded_face_box[0] <= hx <= expanded_face_box[2] and
                        expanded_face_box[1] <= hy <= expanded_face_box[3]):
                        return True, []

        return False, []


class ActivityStateMachine:
    """
    Priority-ordered state machine with candidate state debouncing.
    Priority order:
      1. eyes_closed
      2. hands_joined
      3. phone_detected
      4. distracted
      5. focused_on_book
      6. idle (default)
    """
    def __init__(self, config: dict):
        self.ear_threshold = config.get("ear_threshold", 0.21)
        self.eyes_closed_frame_threshold = config.get("eyes_closed_frame_threshold", 15)
        self.yaw_distracted_threshold = config.get("yaw_distracted_threshold", 25.0)
        self.pitch_focused_book_threshold = config.get("pitch_focused_book_threshold", 15.0)
        self.debounce_seconds = config.get("state_debounce_seconds", 1.5)

        self.eyes_closed_counter = 0
        self.active_state = "idle"
        self.candidate_state = "idle"
        self.candidate_start_time = time.time()
        self.initialized = False

    def update(self, ear_avg: float, yaw: float, pitch: float, phone_detected: bool,
               face_detected: bool, hands_joined: bool = False) -> str:
        """
        Updates consecutive eyes closed frame count and computes priority state.
        Applies candidate state debouncing.
        """
        if face_detected and ear_avg < self.ear_threshold:
            self.eyes_closed_counter += 1
        else:
            self.eyes_closed_counter = 0

        is_eyes_closed = (self.eyes_closed_counter >= self.eyes_closed_frame_threshold)

        # Priority Hierarchy
        if is_eyes_closed:
            raw_state = "eyes_closed"
        elif hands_joined:
            raw_state = "hands_joined"
        elif phone_detected:
            raw_state = "phone_detected"
        elif face_detected and abs(yaw) > self.yaw_distracted_threshold:
            raw_state = "distracted"
        elif face_detected and pitch > self.pitch_focused_book_threshold:
            raw_state = "focused_on_book"
        else:
            raw_state = "idle"

        # Candidate State Debouncing
        now = time.time()
        if not self.initialized:
            self.active_state = raw_state
            self.candidate_state = raw_state
            self.candidate_start_time = now
            self.initialized = True
            return self.active_state

        if raw_state == self.candidate_state:
            if (now - self.candidate_start_time) >= self.debounce_seconds:
                self.active_state = raw_state
        else:
            self.candidate_state = raw_state
            self.candidate_start_time = now

        return self.active_state
