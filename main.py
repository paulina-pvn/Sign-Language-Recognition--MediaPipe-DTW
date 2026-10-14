import time

import cv2

from sign_recorder import SignRecorder
from utils.dataset_utils import load_dataset, load_reference_signs
from utils.mediapipe_utils import create_hand_landmarker, mediapipe_detection
from webcam_manager import WebcamManager


if __name__ == "__main__":
    videos = load_dataset()
    reference_signs = load_reference_signs(videos)
    sign_recorder = SignRecorder(reference_signs)
    webcam_manager = WebcamManager()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open the webcam. Check macOS camera permission for your terminal."
        )

    try:
        with create_hand_landmarker() as hand_landmarker:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Could not read a frame from the webcam.")
                    break

                timestamp_ms = time.monotonic_ns() // 1_000_000
                image, results = mediapipe_detection(
                    frame, hand_landmarker, timestamp_ms
                )
                sign_detected, is_recording = sign_recorder.process_results(results)
                webcam_manager.update(image, results, sign_detected, is_recording)

                pressed_key = cv2.waitKey(1) & 0xFF
                if pressed_key == ord("r"):
                    sign_recorder.record()
                elif pressed_key == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
