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


# Car Counting Using YOLOv8 and SORT

## Overview
This project demonstrates how to use the YOLOv8 model for vehicle detection and the SORT (Simple Online and Realtime Tracking) algorithm for tracking vehicles in a video. The vehicles are counted as they cross a predefined line in the video.

## Setup

### Prerequisites
1. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Download YOLOv8 weights:
    Download the YOLOv8 model weights (e.g., `yolov8l.pt`) from the official [YOLO repository](https://github.com/ultralytics/yolov8) or use a pre-trained model. Place the downloaded weights in the appropriate folder, as indicated in the code.

3. Ensure you have a video file for the car counting:
    The code is set up to work with `cars.mp4` by default. You can replace the path with your own video file.

## Usage

1. Run the Python script:
    ```bash
    python car-counter.py
    ```

2. The video will open, and you will see:
    - Bounding boxes around detected vehicles.
    - A unique ID displayed for each vehicle.
    - The vehicle count will be displayed on the screen as the vehicles cross a predefined line.

3. To exit the video window, press `q`.

## Explanation

### YOLOv8 Model
- The YOLOv8 model is used to detect objects (cars, trucks, and motorbikes) in each frame of the video. It provides the bounding box coordinates and confidence scores for detected objects.

### SORT Tracker
- After detecting the objects, the SORT (Simple Online and Realtime Tracking) algorithm tracks the objects across frames. This allows the program to assign unique IDs to each vehicle and track their movement.

### Vehicle Counting
- Vehicles are counted when they cross a predefined line on the video (represented by `limits`). The count is updated only once for each unique vehicle when it crosses the line.

## Customization

- **Masking**: You can define a region of interest (ROI) by providing a custom mask image. This allows you to focus detection on a specific area of the video.
- **Class Filtering**: The code filters for specific vehicle types like cars, trucks, and motorbikes. You can add or remove vehicle classes from the list to track other objects.
- **Line Position**: You can adjust the position of the line that vehicles must cross by modifying the `limits` list.

## Demo
Here is a quick demo of the car counting in action:

![image](https://github.com/user-attachments/assets/72c3379d-9da5-4d8f-b775-6091d515e9aa)

You can watch the demo video by clicking [here](project%20demo/image%2025-01-03%23-17-36.mp4).


## Conclusion
This project demonstrates how you can use YOLO for high-speed object detection and combine it with tracking algorithms like SORT to track and count vehicles. This can be extended to various applications like traffic monitoring and management.
