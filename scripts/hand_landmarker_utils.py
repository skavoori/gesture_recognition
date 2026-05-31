"""Shared MediaPipe Hand Landmarker setup (Tasks API; mp.solutions removed in 0.10+)."""

from __future__ import annotations

import urllib.request
from typing import Optional

import numpy as np
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.core import image as mp_image_module
from mediapipe.tasks.python.vision.core import (
    vision_task_running_mode as running_mode_module,
)

from paths import ROOT

HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_LANDMARKER_PATH = ROOT / "models" / "hand_landmarker.task"


def ensure_hand_landmarker_model() -> str:
    if HAND_LANDMARKER_PATH.is_file():
        return str(HAND_LANDMARKER_PATH)

    HAND_LANDMARKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Downloading hand landmarker model to {HAND_LANDMARKER_PATH}")
    urllib.request.urlretrieve(HAND_LANDMARKER_URL, HAND_LANDMARKER_PATH)
    return str(HAND_LANDMARKER_PATH)


def create_hand_landmarker(
    *,
    running_mode: running_mode_module.VisionTaskRunningMode = (
        running_mode_module.VisionTaskRunningMode.VIDEO
    ),
    num_hands: int = 1,
    min_hand_detection_confidence: float = 0.5,
    min_hand_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> HandLandmarker:
    options = HandLandmarkerOptions(
        base_options=base_options_module.BaseOptions(
            model_asset_path=ensure_hand_landmarker_model()
        ),
        running_mode=running_mode,
        num_hands=num_hands,
        min_hand_detection_confidence=min_hand_detection_confidence,
        min_hand_presence_confidence=min_hand_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return HandLandmarker.create_from_options(options)


def detect_hands_on_rgb(
    frame_rgb: np.ndarray,
    landmarker: HandLandmarker,
    timestamp_ms: Optional[int] = None,
):
    """Run hand detection on an RGB frame (uint8, HxWx3)."""
    mp_image = mp_image_module.Image(
        image_format=mp_image_module.ImageFormat.SRGB,
        data=np.ascontiguousarray(frame_rgb),
    )
    if timestamp_ms is not None:
        return landmarker.detect_for_video(mp_image, timestamp_ms)
    return landmarker.detect(mp_image)
