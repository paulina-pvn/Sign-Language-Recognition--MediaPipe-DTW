# Sign Language Recognition with MediaPipe and DTW

This project recognizes signs by finding hand landmarks with MediaPipe and comparing hand movement over time with Dynamic Time Warping (DTW). The webcam view shows the detected hand points and the closest matching sign.

## Requirements

- macOS on Apple Silicon, with Python 3.14 already installed
- A webcam
- Internet access for the one-time MediaPipe model download

The application uses MediaPipe Tasks, which requires a separate hand-landmarker model file. The model file is not included in this repository.

## Set up

Open Terminal and go to this project folder (the folder containing `main.py`). Create an isolated environment with your existing Python 3.14 and install the project packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Download the MediaPipe hand-landmarker model into the `models` folder:

```bash
curl -L "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -o models/hand_landmarker.task
```

## Add reference signs

Put videos in a folder named for each sign. The video filename must start with the same sign name and a hyphen; the part before the first hyphen is used as the label. Both `.mp4` and `.webm` files are detected.

```text
data/
└── videos/
    ├── Bonjour/
    │   ├── Bonjour-video1.mp4
    │   └── Bonjour-video2.webm
    └── Merci/
        └── Merci-video1.mp4
```

You can use your own videos. WebM playback depends on whether the OpenCV video backend installed on your machine can decode the video's codec. OpenCV can use different video backends, and their supported formats vary. If a WebM cannot be opened, convert it to MP4 with `ffmpeg` or install an OpenCV build with a compatible FFmpeg/GStreamer backend. The optional `yt_download.py` script downloads clips listed in `yt_links.csv` as MP4; it requires `ffmpeg` (`brew install ffmpeg`).

This folder layout expects one isolated sign example per video. A dataset video that contains a whole lesson or several signs must first be split into short clips and labeled, using its annotation timestamps where available.

## Run

From the project folder, with `.venv` activated, run:

```bash
python main.py
```

The first run extracts hand landmarks from each reference video and stores them in `data/dataset_tasks/`. This separate cache keeps new MediaPipe Tasks landmarks apart from any dataset created by the older version.

When the webcam window appears:

- Press **r** to record a 50-frame sample and compare it with the reference signs.
- Press **q** to quit.

If macOS asks for camera access, allow it for the Terminal application you are using.

To run it again later:

```bash
cd Sign-Language-Recognition--MediaPipe-DTW
source .venv/bin/activate
python main.py
```

## How recognition works

MediaPipe finds up to two hands in each frame. The project turns each hand into a set of angles between hand connections, which makes the features less sensitive to hand size and orientation. DTW compares those angle sequences over time, allowing the same sign to take a little more or less time. The app ranks reference videos by distance and uses a vote among the closest matches to decide what to display.

To teach the app a new sign, add one or more reference videos using the folder and filename pattern above. The **r** key records an example to recognize; it does not add a reference sign.

## Reference

The original approach was inspired by [Sign Language Recognition using MediaPipe](https://www.sicara.ai/blog/sign-language-recognition-using-mediapipe-dtw).
