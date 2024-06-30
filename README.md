### **SMARTDRIVE_YOLOv8**

### Description
SMARTDRIVE_YOLOv8 is an advanced AI-driven system designed to enhance road safety by dynamically adjusting vehicle speed based on real-time detection of critical road conditions. Utilizing YOLOv8 object detection, this model identifies specific classes such as Bumpy Roads, Construction Work, Hospital Zones, Obstacles(like Blocked roads, Fallen trees, Potholes, Car, Bike,...), Pedestrians, School Zones, and Stop signs. 

Through the use of bounding box coordinates to precisely calculate the distance to each object detected and mathematical computations to estimate speed, the model guarantees timely and appropriate speed adjustments, thereby promoting safer driving environments. The system allows for the input of both images and videos and provides thorough reporting and analysis for better traffic control and speed limit observance.

## Table of Contents

Installation 

Deployment

Usage

Results

## Installation

#### Prerequisites
Python 3.7 or higher

pip (Python package installer)

#### Clone the Repository

```bash
  git clone https://github.com/SuryaKS27/SMARTDRIVE_YOLOv8.git
```
####  Install Required Dependencies

```bash
  pip install -r requirements.txt

```

#### Download YOLOv8 Weights

Download the YOLOv8 weights from the official repository or link provided in the documentation.


## Deployment


#### Run the Application
```bash
python speed_limitation_image.py

```
```bash
python speed_limitation_video.py

```
## Usage

#### Input Source:

The system supports both image and video inputs for comprehensive analysis.

#### Object Detection:

Detects classes including Bumpy Roads, Construction Work, Obstacles(like Blocked roads, Fallen trees, Potholes, Car, Bike,...), Pedestrians(Cyclists, Handicaps, children Strollers,...), School Zones, Hospital Zones and Stop signs. 

#### Distance Calculation:

Utilizes bounding box coordinates to calculate the distance to detected objects.

#### Speed Estimation:

Estimates the vehicle's speed using mathematical calculations.

#### Speed Adjustment:

Dynamically adjusts vehicle speed from the intial speed based on real-time analysis to ensure safer driving conditions.

## Results

### Detected Images:
#### Image 1:
![Screenshot 2024-06-15 212806](https://github.com/RamyaMN28/SMARTDRIVE_YOLOv8/assets/122740354/dfc84640-4b35-4496-8dae-8c439e1787d1)
#### Image 2:
![Screenshot 2024-06-15 213211](https://github.com/RamyaMN28/SMARTDRIVE_YOLOv8/assets/122740354/8443fe43-feb0-4872-a6bc-46cbcfbf5911)

### Detected Videos:
#### Video 1:
https://github.com/RamyaMN28/SMARTDRIVE_YOLOv8/assets/122740354/434ecb2f-5895-41d9-b5c7-b8141c481783

#### Video 2:
https://github.com/RamyaMN28/SMARTDRIVE_YOLOv8/assets/122740354/75e00aa0-9b5e-4c3e-8a4f-9b46c67e0a0b

## Author

- [@Surya K S ](https://github.com/SuryaKS27/)


