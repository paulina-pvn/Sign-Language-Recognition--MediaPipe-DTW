import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from models.sign_model import SignModel
from utils.landmark_utils import save_landmarks_from_video, load_array


DATASET_FOLDER = os.path.join("data", "dataset_tasks")
VIDEO_EXTENSIONS = (".mp4", ".webm", ".mov")


def load_dataset():
    video_root = Path("data/videos")
    videos = sorted(
        str(path.relative_to(video_root).with_suffix(""))
        for path in video_root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    dataset = [
        os.path.relpath(root, DATASET_FOLDER)
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
        sign_name = Path(video_name).parent.name
        path = os.path.join(DATASET_FOLDER, video_name)
        cache_name = Path(video_name).name

        left_hand_list = load_array(os.path.join(path, f"lh_{cache_name}.pickle"))
        right_hand_list = load_array(os.path.join(path, f"rh_{cache_name}.pickle"))

        reference_signs["name"].append(sign_name)
        reference_signs["sign_model"].append(SignModel(left_hand_list, right_hand_list))
        reference_signs["distance"].append(0)
    
    reference_signs = pd.DataFrame(reference_signs, dtype=object)
    print(
        f'Dictionary count: {reference_signs[["name", "sign_model"]].groupby(["name"]).count()}'
    )
    return reference_signs
