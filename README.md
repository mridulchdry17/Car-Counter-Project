# Car Counter Project

This project uses **YOLOv8** (You Only Look Once) for object detection combined with **OpenCV** for video processing and **SORT** (Simple Online and Realtime Tracking) to track vehicles in real-time. The goal of this project is to detect and count vehicles, such as cars, trucks, and motorbikes, in a video feed.

## Features

- Real-time vehicle detection and tracking.
- Use of **YOLOv8** for high-accuracy object detection.
- **SORT** for tracking objects across frames.
- Vehicle count is updated dynamically as vehicles cross a predefined line.
- Customizable mask for focusing on a specific region of interest in the video.

## Requirements

Make sure you have the following installed:

- Python 3.x
- OpenCV
- YOLOv8 (via the `ultralytics` package)
- NumPy
- cvzone (for text and bounding box rendering)
- SORT (for object tracking)

You can install the required dependencies using `pip`:

```bash
pip install opencv-python opencv-python-headless ultralytics numpy cvzone sort

