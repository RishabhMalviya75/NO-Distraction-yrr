import time
import os
import unittest
import numpy as np

from activity_detector import (
    calculate_ear, estimate_head_pose, detect_hands_joined,
    ActivityStateMachine, PhoneDetector
)
from media_player import MediaPlayer

class MockLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z

class MockHand:
    def __init__(self, offset_x=0.0):
        self.landmark = [MockLandmark(0.5 + offset_x, 0.5) for _ in range(21)]

class MockFaceLandmarks:
    def __init__(self):
        self.landmark = [MockLandmark(0.5, 0.5) for _ in range(468)]
        # Right Eye
        self.landmark[33] = MockLandmark(0.3, 0.4)
        self.landmark[133] = MockLandmark(0.4, 0.4)
        self.landmark[160] = MockLandmark(0.35, 0.38)
        self.landmark[144] = MockLandmark(0.35, 0.42)
        self.landmark[158] = MockLandmark(0.37, 0.38)
        self.landmark[153] = MockLandmark(0.37, 0.42)

        # Left Eye
        self.landmark[362] = MockLandmark(0.6, 0.4)
        self.landmark[263] = MockLandmark(0.7, 0.4)
        self.landmark[385] = MockLandmark(0.65, 0.38)
        self.landmark[380] = MockLandmark(0.65, 0.42)
        self.landmark[387] = MockLandmark(0.67, 0.38)
        self.landmark[373] = MockLandmark(0.67, 0.42)

class TestActivityDetection(unittest.TestCase):

    def test_ear_calculation(self):
        landmarks = MockFaceLandmarks()
        ear_avg, ear_l, ear_r = calculate_ear(landmarks, 640, 480)
        self.assertGreater(ear_avg, 0.0)
        self.assertIsInstance(ear_avg, float)

    def test_head_pose_estimation(self):
        landmarks = MockFaceLandmarks()
        pitch, yaw, roll = estimate_head_pose(landmarks, 640, 480)
        self.assertIsInstance(pitch, float)
        self.assertIsInstance(yaw, float)
        self.assertIsInstance(roll, float)

    def test_hands_joined_detection(self):
        # Two hands right next to each other (joined palms)
        hand1 = MockHand(offset_x=0.0)
        hand2 = MockHand(offset_x=0.02)
        self.assertTrue(detect_hands_joined([hand1, hand2]))

        # Two hands far apart
        hand3 = MockHand(offset_x=0.4)
        self.assertFalse(detect_hands_joined([hand1, hand3]))

    def test_state_machine_priorities(self):
        config = {
            "ear_threshold": 0.21,
            "eyes_closed_frame_threshold": 3,
            "state_debounce_seconds": 0.1,
            "yaw_distracted_threshold": 25.0,
            "pitch_focused_book_threshold": 15.0
        }
        sm = ActivityStateMachine(config)

        # Frame 1: Normal idle state
        state = sm.update(ear_avg=0.30, yaw=0.0, pitch=0.0, phone_detected=False, face_detected=True, hands_joined=False)
        self.assertEqual(state, "idle")

        # Test Priority 1: Eyes Closed
        sm.update(ear_avg=0.15, yaw=0.0, pitch=0.0, phone_detected=True, face_detected=True, hands_joined=True)
        sm.update(ear_avg=0.15, yaw=0.0, pitch=0.0, phone_detected=True, face_detected=True, hands_joined=True)
        sm.update(ear_avg=0.15, yaw=0.0, pitch=0.0, phone_detected=True, face_detected=True, hands_joined=True)
        time.sleep(0.15)
        state = sm.update(ear_avg=0.15, yaw=0.0, pitch=0.0, phone_detected=True, face_detected=True, hands_joined=True)
        self.assertEqual(state, "eyes_closed")

        # Test Priority 2: Hands Joined
        sm = ActivityStateMachine(config)
        sm.update(ear_avg=0.30, yaw=0.0, pitch=0.0, phone_detected=False, face_detected=True, hands_joined=False)
        sm.update(ear_avg=0.30, yaw=0.0, pitch=0.0, phone_detected=True, face_detected=True, hands_joined=True)
        time.sleep(0.15)
        state = sm.update(ear_avg=0.30, yaw=0.0, pitch=0.0, phone_detected=True, face_detected=True, hands_joined=True)
        self.assertEqual(state, "hands_joined")

        # Test Priority 3: Phone Detected
        sm = ActivityStateMachine(config)
        sm.update(ear_avg=0.30, yaw=0.0, pitch=0.0, phone_detected=False, face_detected=True, hands_joined=False)
        sm.update(ear_avg=0.30, yaw=30.0, pitch=20.0, phone_detected=True, face_detected=True, hands_joined=False)
        time.sleep(0.15)
        state = sm.update(ear_avg=0.30, yaw=30.0, pitch=20.0, phone_detected=True, face_detected=True, hands_joined=False)
        self.assertEqual(state, "phone_detected")

    def test_media_player_with_all_media(self):
        activity_map = {
            "eyes_closed": "media/close_eye.mpeg",
            "hands_joined": "media/joinhand.mpeg",
            "phone_detected": "media/phone.mpeg",
            "distracted": None,
            "focused_on_book": None,
            "idle": None
        }
        player = MediaPlayer(activity_map)
        self.assertTrue(player.is_initialized)

        player.play_for_state("hands_joined")
        self.assertEqual(player.current_state, "hands_joined")
        time.sleep(0.15)

        player.play_for_state("idle")
        self.assertEqual(player.current_state, "idle")

        player.stop()

if __name__ == "__main__":
    unittest.main()
