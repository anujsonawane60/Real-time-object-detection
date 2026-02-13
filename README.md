# Real-time Object Detection System

## Overview
This Python-based real-time object detection system utilizes YOLOv3 to detect multiple objects in video streams or images, delivering fast and accurate results.

## Features
- 🎥 Real-time video processing
- 🖼️ Image processing
- 📦 Multi-object detection
- 🏷️ Object classification
- 📊 Confidence scoring
- ⚡ Optimized performance

## Prerequisites
- Python 3.7+
- OpenCV
- NumPy
- YOLOv3 weights file

## Installation

1. Clone the repository
```bash
git clone <your-repository-url>
cd Real-time-object-detection

Install required dependencies:
```
pip install -r requirements.txt
```

Download YOLOv3 weights and place the file in the project root directory.

**Project Structure**
```
Real-time-object-detection/
├── main.py
├── requirements.txt
├── .gitignore
├── output/
└── README.md
```

**Usage**
Run the object detection system with:
```
python main.py


**Configuration**
You can configure the system using the following parameters:
- Detection confidence threshold
- Non-maximum suppression threshold
- Input source selection (video/webcam)

**Detection Classes**
The system can detect:
- Persons
- Vehicles (cars, bikes, trucks)
- Animals
- Common indoor/outdoor objects

**Output**
Detection results include:
- Bounding boxes around detected objects
- Class labels
- Confidence scores
- Processed frames/images in the output directory

**Performance**
- Real-time processing capability
- GPU acceleration support (if available)
- Optimized for efficiency

**Contributing**
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

**License**
This project is licensed under the MIT License.

**Acknowledgments**
- YOLOv3 Algorithm
- OpenCV library
- NumPy library

