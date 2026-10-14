"""Helpers for running MediaPipe Tasks on images and video frames."""

from pathlib import Path
from types import SimpleNamespace

import cv2
import mediapipe as mp


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"


def create_hand_landmarker():
    """Create a video-mode hand detector using the downloaded Tasks model."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "MediaPipe hand model not found. Download hand_landmarker.task into "
            f"{MODEL_PATH.parent} (see the README setup instructions)."
        )

    options = mp.tasks.vision.HandLandmarkerOptions(
        # MediaPipe's Metal helper aborts on this Mac setup; CPU is supported here.
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH),
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def mediapipe_detection(image, model, timestamp_ms):
    """Detect hands in a frame and return it mirrored for display and drawing."""
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    task_result = model.detect_for_video(mp_image, timestamp_ms)

    left_hand = None
    right_hand = None
    for landmarks, handedness in zip(task_result.hand_landmarks, task_result.handedness):
        label = handedness[0].category_name.lower() if handedness else ""
        # Tasks handedness labels assume a mirrored (selfie) input image.
        hand = SimpleNamespace(landmark=landmarks)
        if label == "left":
            left_hand = hand
        elif label == "right":
            right_hand = hand

    # Keep the result shape expected by the sign and drawing code.
    results = SimpleNamespace(
        left_hand_landmarks=left_hand,
        right_hand_landmarks=right_hand,
    )
    return image, results
