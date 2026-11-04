import cv2
import os
import numpy as np
import pickle as pkl
from pathlib import Path
from utils.mediapipe_utils import create_hand_landmarker, mediapipe_detection


DATASET_FOLDER = os.path.join("data", "dataset_tasks")
VIDEO_EXTENSIONS = {".mp4", ".webm"}


def landmark_to_array(mp_landmark_list):
    """Return a np array of size (nb_keypoints x 3)"""
    if mp_landmark_list is None:
        return np.zeros((21, 3), dtype=float)
    keypoints = []
    for landmark in mp_landmark_list.landmark:
        keypoints.append([landmark.x, landmark.y, landmark.z])
    return np.nan_to_num(keypoints)


def extract_landmarks(results):
    """Extract the results of both hands and convert them to a np array of size
    if a hand doesn't appear, return an array of zeros

    :param results: mediapipe object that contains the 3D position of all keypoints
    :return: Two np arrays of size (1, 21 * 3) = (1, nb_keypoints * nb_coordinates) corresponding to both hands
    """
    # Pose landmarks were saved by the original project but are not used by its
    # hand-only sign embedding. Keep a zero-filled pose entry for cache compatibility.
    pose = np.zeros(99).tolist()

    left_hand = np.zeros(63).tolist()
    if results.left_hand_landmarks:
        left_hand = landmark_to_array(results.left_hand_landmarks).reshape(63).tolist()

    right_hand = np.zeros(63).tolist()
    if results.right_hand_landmarks:
        right_hand = (
            landmark_to_array(results.right_hand_landmarks).reshape(63).tolist()
        )
    return pose, left_hand, right_hand


def save_landmarks_from_video(video_name):
    landmark_list = {"pose": [], "left_hand": [], "right_hand": []}
    video_relative = Path(video_name)
    video_stem = video_relative.name
    video_folder = os.path.join("data", "videos", str(video_relative.parent))
    video_path = next(
        (
            os.path.join(video_folder, file_name)
            for file_name in os.listdir(video_folder)
            if Path(file_name).stem == video_stem
            and Path(file_name).suffix.lower() in VIDEO_EXTENSIONS
        ),
        None,
    )
    if video_path is None:
        raise FileNotFoundError(
            f"Could not find an MP4 or WebM reference video for: {video_name}"
        )

    # Set the Video stream
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not decode reference video: {video_path}. "
            "Check that this OpenCV build supports its video codec."
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    try:
        with create_hand_landmarker() as hand_landmarker:
            frame_index = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp_ms = int(frame_index * 1000 / fps)
                _, results = mediapipe_detection(
                    frame, hand_landmarker, timestamp_ms
                )
                pose, left_hand, right_hand = extract_landmarks(results)
                landmark_list["pose"].append(pose)
                landmark_list["left_hand"].append(left_hand)
                landmark_list["right_hand"].append(right_hand)
                frame_index += 1
    finally:
        cap.release()

    # Keep each video's landmarks under its label and relative video path.
    data_path = os.path.join(DATASET_FOLDER, video_name)
    os.makedirs(data_path, exist_ok=True)

    # Saving the landmark_list in the correct folder
    save_array(
        landmark_list["pose"], os.path.join(data_path, f"pose_{video_stem}.pickle")
    )
    save_array(
        landmark_list["left_hand"], os.path.join(data_path, f"lh_{video_stem}.pickle")
    )
    save_array(
        landmark_list["right_hand"], os.path.join(data_path, f"rh_{video_stem}.pickle")
    )


def save_array(arr, path):
    file = open(path, "wb")
    pkl.dump(arr, file)
    file.close()


def load_array(path):
    file = open(path, "rb")
    arr = pkl.load(file)
    file.close()
    return np.array(arr)
