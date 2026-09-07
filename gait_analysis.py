"""
gait_analysis.py
-----------------
Extracts knee-joint gait features from a walking video using MediaPipe's
PoseLandmarker (Tasks API).

NOTE: Newer versions of the `mediapipe` package (0.10.31+) removed the old
`mp.solutions.pose` API. This file has been updated to use the current
`mp.tasks.vision.PoseLandmarker` API instead. It also auto-downloads the
small pose-landmarker model file on first run if it isn't present yet.

Features extracted (v1 — kept deliberately small and interpretable):
    1. knee_flexion_range_left / right : max - min knee angle over the video (deg)
    2. asymmetry_index                : |left_range - right_range| / avg(range) * 100 (%)
    3. cadence_proxy                  : estimated steps per second, from ankle
                                         vertical oscillation peaks

These are NOT clinical-grade gait metrics — they are a fast, explainable
proxy set suitable for a screening/triage tool, not a diagnostic device.
"""

import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from dataclasses import dataclass, asdict
from typing import List, Dict

# ----------------------------------------------------------------------
# Pose landmark indices (fixed order used by MediaPipe's 33-point model).
# The old `mp.solutions.pose.PoseLandmark` enum is gone in newer versions,
# so we hardcode the indices we need directly.
# ----------------------------------------------------------------------
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

# Where the pose landmarker model file will be cached locally.
_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
_MODEL_PATH = os.path.join(_MODEL_DIR, "pose_landmarker_lite.task")
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)


def _ensure_model_downloaded() -> str:
    """Download the pose landmarker model file once, if not already present."""
    if not os.path.exists(_MODEL_PATH):
        os.makedirs(_MODEL_DIR, exist_ok=True)
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    return _MODEL_PATH


@dataclass
class GaitFeatures:
    knee_flexion_range_left: float
    knee_flexion_range_right: float
    asymmetry_index: float
    cadence_proxy: float
    frames_analyzed: int
    frames_with_pose: int
    pose_detection_rate: float

    def as_dict(self) -> Dict:
        return asdict(self)


def _angle_3pt(a, b, c) -> float:
    """Angle at point b (in degrees), given 3 (x, y) points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom == 0:
        return np.nan
    cosine = np.dot(ba, bc) / denom
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _count_peaks(signal: np.ndarray, min_distance: int = 5) -> int:
    """Very lightweight peak counter (no scipy dependency)."""
    peaks = 0
    last_peak_idx = -min_distance
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i - 1] and signal[i] >= signal[i + 1]:
            if i - last_peak_idx >= min_distance:
                peaks += 1
                last_peak_idx = i
    return peaks


def extract_gait_features(video_path: str, sample_every_n: int = 1) -> GaitFeatures:
    """
    Run MediaPipe PoseLandmarker over the video and return GaitFeatures.

    Parameters
    ----------
    video_path : path to a short (5-20s) side-on walking video
    sample_every_n : process every Nth frame (speed vs accuracy tradeoff)
    """
    model_path = _ensure_model_downloaded()

    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    left_knee_angles: List[float] = []
    right_knee_angles: List[float] = []
    left_ankle_y: List[float] = []
    frames_total = 0
    frames_with_pose = 0

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % sample_every_n != 0:
                continue
            frames_total += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((frame_idx / fps) * 1000) if fps > 0 else frame_idx

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if not result.pose_landmarks:
                continue
            frames_with_pose += 1

            # First detected person's landmarks (list of 33 NormalizedLandmark)
            lm = result.pose_landmarks[0]

            def pt(idx):
                p = lm[idx]
                return (p.x, p.y)

            l_angle = _angle_3pt(pt(LEFT_HIP), pt(LEFT_KNEE), pt(LEFT_ANKLE))
            r_angle = _angle_3pt(pt(RIGHT_HIP), pt(RIGHT_KNEE), pt(RIGHT_ANKLE))

            if not np.isnan(l_angle):
                left_knee_angles.append(l_angle)
            if not np.isnan(r_angle):
                right_knee_angles.append(r_angle)

            left_ankle_y.append(lm[LEFT_ANKLE].y)

    cap.release()

    if len(left_knee_angles) < 5 or len(right_knee_angles) < 5:
        raise ValueError(
            "Not enough pose detections in this video. "
            "Use a clearer, well-lit, side-on full-body walking video."
        )

    left_range = float(np.max(left_knee_angles) - np.min(left_knee_angles))
    right_range = float(np.max(right_knee_angles) - np.min(right_knee_angles))
    avg_range = (left_range + right_range) / 2 if (left_range + right_range) > 0 else 1e-6
    asymmetry = abs(left_range - right_range) / avg_range * 100

    # crude cadence proxy: count vertical oscillation peaks in ankle trajectory
    ankle_arr = np.array(left_ankle_y)
    ankle_arr = -ankle_arr  # invert so "up" is a peak (image y grows downward)
    n_peaks = _count_peaks(ankle_arr, min_distance=max(3, int(fps / 6)))
    duration_sec = frames_total / fps if fps > 0 else frames_total / 30.0
    cadence_proxy = (n_peaks / duration_sec) if duration_sec > 0 else 0.0

    return GaitFeatures(
        knee_flexion_range_left=round(left_range, 1),
        knee_flexion_range_right=round(right_range, 1),
        asymmetry_index=round(asymmetry, 1),
        cadence_proxy=round(cadence_proxy, 2),
        frames_analyzed=frames_total,
        frames_with_pose=frames_with_pose,
        pose_detection_rate=round(
            (frames_with_pose / frames_total * 100) if frames_total else 0.0, 1
        ),
    )
