import os
import subprocess
import sys

HOME = os.getcwd()
print(HOME)

import ultralytics
ultralytics.checks()
from ultralytics import YOLO

#CUSTOM TRAINING

yolo_command = "yolo task=detect mode=train model=yolov8s.pt data=data1.yaml epochs=80 imgsz=416 plots=True"

subprocess.run(yolo_command, shell=True)

#VALIDATION

yolo_command = "yolo task=detect mode=val model=runs/detect/train5/weights/best.pt  data=data1.yaml"

subprocess.run(yolo_command, shell=True)

#TESTING

yolo_command = "yolo task=detect mode=predict model=runs/detect/train5/weights/best.pt conf=0.25 source=datasets/test/images save=True"

subprocess.run(yolo_command, shell=True)

#TESTING VIDEO
yolo_command = "yolo task=detect mode=predict model=runs/detect/train5/weights/best.pt conf=0.25 source=testing/videojaya.mp4 save=True"

subprocess.run(yolo_command, shell=True)

#TESTING IMAGE
yolo_command = "yolo task=detect mode=predict model=runs/detect/train5/weights/best.pt conf=0.25 source=testing/img1.jpeg save=True"  

subprocess.run(yolo_command, shell=True)
