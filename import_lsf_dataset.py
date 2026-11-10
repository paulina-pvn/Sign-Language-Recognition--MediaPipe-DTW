"""Copy the downloaded Hugging Face LSF videos into the DTW folder layout."""

import argparse
import hashlib
import shutil
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".webm"}
LABEL_RENAMES = {"où _": "où", "comment _": "comment"}
SELECTED_LABELS = {
    "Bonjour",
    "merci",
    "de rien",
    "s'il vous plait",
    "au revoir",
    "oui",
    "non",
    "ça va",
    "pardon",
    "comprendre",
    "pas compris",
    "je ne sais pas",
    "je veux",
    "manger",
    "boire",
    "eau",
    "famille",
    "ami·e",
    "où",
    "comment",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=Path.home() / "Datasets" / "LSF_data" / "videos",
        help="Downloaded dataset videos folder (default: ~/Datasets/LSF_data/videos)",
    )
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    if not source.is_dir():
        parser.error(f"Dataset video folder not found: {source}")

    videos = sorted(
        path for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        parser.error(f"No MP4 or WebM files found under: {source}")

    destination_root = Path("data/videos")
    imported = skipped = 0
    for video in videos:
        # In this dataset, each filename is the French sign label; its parent
        # directory (for example "jauvert") identifies the video source.
        source_label = video.stem.strip()
        label = LABEL_RENAMES.get(source_label, source_label)
        if label not in SELECTED_LABELS:
            skipped += 1
            continue
        if not label or label in {".", ".."}:
            skipped += 1
            continue

        relative_source = video.relative_to(source).as_posix()
        digest = hashlib.sha1(relative_source.encode("utf-8")).hexdigest()[:8]
        author = video.parent.name.strip().replace(" ", "_") or "source"
        target_dir = destination_root / label
        target = target_dir / f"hf-{author}-{digest}{video.suffix.lower()}"
        if target.exists():
            skipped += 1
            continue

        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(video, target)
        imported += 1

    print(f"Imported {imported} videos into {destination_root}; skipped {skipped}.")
    print("The video files are ignored by Git and will stay local to this checkout.")


if __name__ == "__main__":
    main()
