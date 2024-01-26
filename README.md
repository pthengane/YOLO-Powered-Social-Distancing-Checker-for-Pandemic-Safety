# YOLO-Powered Social Distancing Checker for Pandemic Safety

## Overview

This project introduces a Social Distancing Checker powered by YOLOv3, an advanced real-time object detection model trained on the COCO dataset. The primary focus is on monitoring adherence to social distancing in crowded areas, utilizing computer vision to detect individuals and assess their spatial relationships. The aim is to provide a tool that seamlessly integrates with CCTV cameras for efficient social distancing monitoring.

## Project Features

### Object Detection with YOLOv3

The heart of this project relies on YOLOv3, a cutting-edge object detection algorithm. YOLO, which stands for You Only Look Once, is renowned for its efficiency and accuracy in detecting multiple object classes in real-time. The model comes pre-trained on the COCO dataset, encompassing a diverse set of objects.

### Social Distancing Analysis

The system identifies individuals in a given video stream, evaluates their positions, and determines if social distancing norms are violated. It employs sophisticated spatial analysis to calculate pairwise distances between centroids of detected persons. Violations are flagged if the distance falls below a predefined threshold.

### Results Visualization

Real-time results are displayed with bounding boxes around detected individuals. The system employs color-coded indicators to highlight the risk level of each person – light green for safe, dark red for high risk, and orange for low risk. Connecting lines between individuals visually depict their closeness, with red indicating very close and yellow denoting close proximity.

## How to Use

### Prerequisites

Ensure you have the required Python packages installed. You can install them using the following command:

```bash
pip install numpy imutils opencv-python argparse scipy

```

### Running the Social Distancing Checker

1. **Download YOLOv3 Weights:**
   - Obtain the YOLOv3 weights file from [Here](https://drive.google.com/file/d/1zNcAkS4y2WVsBDwQbP0qoSImQoqxKwHS/view?usp=sharing?export=download).
   - Place the weights file in the `yolo-coco` folder.

2. **Run the System:**
   - Open the `main.py` script on your local machine or in Google Colab.
   - Execute the code cells sequentially.

3. **Customize Behavior:**
   - Use command line arguments to tailor the system behavior.
   - Example: `python main.py --input /path/to/input/video.mp4 --output /path/to/output/video.avi --display 1`

## Results Interpretation

The system delivers real-time feedback on social distancing adherence, displaying bounding boxes around detected individuals, centroid coordinates, and a count of social distancing violations. Violations are categorized based on risk level, offering a comprehensive visual analysis.

![Output avi gif](https://github.com/abd-shoumik/Social-distance-detection/blob/master/social%20distance%20detection.gif)

## Project Structure

- **main.py:** The main script for running the Social Distancing Checker.
- **src/utils.py:** Contains utility functions, including the detection of people in a frame.
- **src/model.py:** Contains the function to load the YOLOv3 model.

## Acknowledgments

This project builds upon the YOLOv3 object detection system and is designed for educational and awareness purposes. It showcases the potential of computer vision in addressing real-world challenges, especially during times of public health concern.

Feel free to explore, contribute, and adapt the project for specific use cases. Together, we can leverage technology to foster safer and healthier environments.
