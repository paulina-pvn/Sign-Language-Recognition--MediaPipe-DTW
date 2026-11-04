import os

import pandas as pd
from tqdm import tqdm

from models.sign_model import SignModel
from utils.landmark_utils import save_landmarks_from_video, load_array


DATASET_FOLDER = os.path.join("data", "dataset_tasks")
VIDEO_EXTENSIONS = (".mp4", ".webm")


def load_dataset():
    videos = sorted({
        os.path.splitext(file_name)[0]
        for root, dirs, files in os.walk(os.path.join("data", "videos"))
        for file_name in files
        if os.path.splitext(file_name)[1].lower() in VIDEO_EXTENSIONS
    })
    dataset = [
        file_name.replace(".pickle", "").replace("pose_", "")
        for root, dirs, files in os.walk(DATASET_FOLDER)
        for file_name in files
        if file_name.endswith(".pickle") and file_name.startswith("pose_")
    ]

    # Create the dataset from the reference videos
    videos_not_in_dataset = list(set(videos).difference(set(dataset)))
    n = len(videos_not_in_dataset)
    if n > 0:
        print(f"\nExtracting landmarks from new videos: {n} videos detected\n")

        for idx in tqdm(range(n)):
            save_landmarks_from_video(videos_not_in_dataset[idx])

    return videos


def load_reference_signs(videos):
    reference_signs = {"name": [], "sign_model": [], "distance": []}
    for video_name in videos:
        sign_name = os.path.splitext(video_name)[0].split("-")[0]
        path = os.path.join(DATASET_FOLDER, sign_name, video_name)

        left_hand_list = load_array(os.path.join(path, f"lh_{video_name}.pickle"))
        right_hand_list = load_array(os.path.join(path, f"rh_{video_name}.pickle"))

        reference_signs["name"].append(sign_name)
        reference_signs["sign_model"].append(SignModel(left_hand_list, right_hand_list))
        reference_signs["distance"].append(0)
    
    reference_signs = pd.DataFrame(reference_signs, dtype=object)
    print(
        f'Dictionary count: {reference_signs[["name", "sign_model"]].groupby(["name"]).count()}'
    )
    return reference_signs
