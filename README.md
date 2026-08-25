# Real-Time Activity-Based Media Trigger (Computer Vision)

A Python computer vision application that detects user activities via webcam in real-time and dynamically plays corresponding audio (`.mpeg`, `.mp3`, `.wav`, etc.) clips based on the detected activity state.

---

## Features

- **Activity Detection**:
  1. **Eyes Closed (`eyes_closed`)**: Calculates Eye Aspect Ratio (EAR) using MediaPipe 468 3D face mesh landmarks. Flags state when eyes remain closed across configurable consecutive frames (avoids false alerts from blinking).
  2. **Phone Detected (`phone_detected`)**: Leverages YOLOv8 (`ultralytics`) COCO object detection (class `cell phone`) with proximity heuristics.
  3. **Distracted (`distracted`)**: Uses 3D head pose estimation (`cv2.solvePnP`) to compute head Yaw angle (detects turning left/right away from screen).
  4. **Focused on Book (`focused_on_book`)**: Computes head Pitch angle to detect steady downward gaze towards a book or desk area.
  5. **Idle (`idle`)**: Default active state when looking normally at the camera/screen.
- **Priority-Ordered State Machine**:
  `eyes_closed` > `phone_detected` > `distracted` > `focused_on_book` > `idle`
- **Audio State Engine**: Non-blocking audio playback using Pygame Mixer. On state changes, gracefully switches or stops audio without repeating tracks frame-by-frame.
- **Debounce & Cooldown**: Configurable state debounce timer (default `1.5s`) prevents stuttering from transient noise or momentary state flickers.
- **On-Screen OpenCV Overlay**: Color-coded status banner, real-time EAR bar, Yaw/Pitch gauges, phone bounding boxes, and FPS meter.

---

## Project Structure

```
cv-activity-media-trigger/
├── main.py                 # Video capture loop, debug UI rendering, state orchestration
├── activity_detector.py    # EAR calculation, 3D head pose (Yaw/Pitch), YOLO phone detector, state machine
├── media_player.py         # Pygame mixer audio player wrapper
├── config.json             # State-to-audio file mappings and detection thresholds
├── media/                  # User audio clips (.mpeg, .mp3, .wav)
├── requirements.txt        # Pinned Python package dependencies
└── README.md               # Setup and usage guide
```

---

## Prerequisites & Installation

### 1. Requirements
- **Python 3.10+**
- **Webcam** (USB or integrated camera)

### 2. Install Dependencies

Install required Python packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Running the Application

Launch the main computer vision loop:

```bash
python main.py
```

### Controls:
- **`q`**: Quit the application.
- **`c`**: Toggle the on-screen sensor diagnostic overlay panel.

---

## Configuration (`config.json`)

All threshold values, state debouncing rules, and media file mappings are stored in `config.json`:

```json
{
  "activities": {
    "eyes_closed": "media/close_eye.mpeg",
    "phone_detected": null,
    "distracted": null,
    "focused_on_book": null,
    "idle": null
  },
  "ear_threshold": 0.21,
  "eyes_closed_frame_threshold": 15,
  "state_debounce_seconds": 1.5,
  "yaw_distracted_threshold": 25.0,
  "pitch_focused_book_threshold": 15.0,
  "phone_confidence_threshold": 0.4,
  "use_yolo_phone": true
}
```

### Adding Your Own Media Files

1. Place your audio files (`.mpeg`, `.mp3`, `.wav`, etc.) into the `media/` folder.
2. Edit `config.json` to map the activity state to the relative file path. For example:
   ```json
   "activities": {
     "eyes_closed": "media/close_eye.mpeg",
     "phone_detected": "media/my_phone_alert.mp3",
     "distracted": "media/stay_focused.wav"
   }
   ```
3. Set any unneeded activity to `null` to disable audio for that state.

---

## Threshold Tuning Guide

1. **Eye Aspect Ratio (`ear_threshold`)**:
   - Standard open eye EAR: `~0.25 - 0.35`
   - Closed eye EAR: `< 0.20`
   - *Tuning*: If drowsy alerts trigger while eyes are open, lower `ear_threshold` (e.g. `0.19`). If closed eyes aren't detected, raise `ear_threshold` (e.g. `0.23`).

2. **Eyes Closed Consecutive Frames (`eyes_closed_frame_threshold`)**:
   - Default: `15` frames (at ~30 FPS, equals ~0.5s of continuous eye closure).
   - *Tuning*: Increase to `25-30` frames if normal eye blinking triggers false drowsiness alerts.

3. **Head Yaw Angle (`yaw_distracted_threshold`)**:
   - Measures left/right head rotation in degrees.
   - *Tuning*: Looking slightly off-center is normal; set to `25.0`–`30.0` degrees so only significant sideways head turns trigger the `distracted` state.

4. **Head Pitch Angle (`pitch_focused_book_threshold`)**:
   - Measures up/down head tilt in degrees (positive values indicate looking down).
   - *Tuning*: Adjust `15.0`–`20.0` degrees based on where your book or desk reading area is positioned relative to your camera.

5. **State Debounce Timer (`state_debounce_seconds`)**:
   - Controls how long a candidate state must persist before audio switches (default `1.5s`).
   - Prevents rapid audio restarts when switching positions.
