import cv2 as cv
import math
from jinja2 import Template
import base64
import os
from ultralytics import YOLO

# Calculate the focal length using a reference object
def focal_length_finder(measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width
    return focal_length

# Calculate the distance to the object
def distance_finder(focal_length, real_object_width, width_in_frame):
    distance = (real_object_width * focal_length) / width_in_frame
    return distance

# Function to load YOLOv8 model
def load_yolov8_model(model_path):
    model = YOLO(model_path)
    return model

# Function to calculate adjusted speed based on distance
def calculate_adjusted_speed(distance, speed_limit):
    reaction_time = 1.5  # seconds
    friction_coefficient = 0.7
    gravity = 9.8  # m/s^2

    # Solve for the speed v using the quadratic formula
    a = 1 / (2 * friction_coefficient * gravity)
    b = reaction_time
    c = -distance

    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return 0  # No real solution, cannot stop in time
    else:
        v1 = (-b + math.sqrt(discriminant)) / (2*a)
        v2 = (-b - math.sqrt(discriminant)) / (2*a)
        adjusted_speed = max(v1, v2) * 3.6  # Convert m/s to km/h

        # Ensure adjusted speed does not exceed speed limit
        return min(adjusted_speed, speed_limit)

# Function to calculate distance based on object width in pixels
def distance_finder(focal_length, real_width, object_width_pixels):
    return (real_width * focal_length) / object_width_pixels

# Object detector function for a single frame
def object_detector(image, model, focal_length, real_width, closest_objects_info, confidence_threshold=0.5):
    results = model(image)
    data_list = []

    # Iterate over detections
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            if confidence > confidence_threshold:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                label = model.names[class_id]
                color = (255, 0, 0)  # Red color for bounding box
                cv.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv.putText(image, f"{label}: {confidence:.2f}", (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                object_width_pixels = x2 - x1  # Calculate width of the detected object in pixels
                distance = distance_finder(focal_length, real_width, object_width_pixels)
                
                # Find speed limit for the detected class
                closest_object = next((obj for obj in closest_objects_info if obj['Class'] == label), None)
                if closest_object:
                    speed_limit = closest_object['Speed Limit']
                    
                    # Calculate the adjusted speed
                    adjusted_speed = calculate_adjusted_speed(distance, speed_limit)
                    
                    data_list.append({
                        "Class": label,
                        "Object Width (pixels)": object_width_pixels,
                        "Distance (meters)": distance,
                        "Speed Limit (km/h)": speed_limit,
                        "Adjusted Speed (km/h)": adjusted_speed,
                        "Bounding Box": (x1, y1, x2, y2)
                    })

                    # Annotate frame with speed limit and distance
                    cv.putText(image, f"Speed Limit: {speed_limit} km/h", (x1, y2 + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv.putText(image, f"Distance: {distance:.2f} m", (x1, y2 + 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return data_list, image

# Process video frames
def process_video(video_path, model, focal_length, real_width, closest_objects_info, output_path, confidence_threshold=0.5):
    cap = cv.VideoCapture(video_path)
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        object_data, annotated_frame = object_detector(frame, model, focal_length, real_width, closest_objects_info, confidence_threshold)
        
        # Draw object details on the frame
        for obj in object_data:
            x1, y1, x2, y2 = obj["Bounding Box"]
            cv.putText(annotated_frame, f"Class: {obj['Class']}", (x1, y1 - 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv.putText(annotated_frame, f"Speed Limit: {obj['Speed Limit (km/h)']} km/h", (x1, y1 - 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv.putText(annotated_frame, f"Distance: {obj['Distance (meters)']:.2f} m", (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv.putText(annotated_frame, f"Adjusted Speed: {obj['Adjusted Speed (km/h)']:.2f} km/h", (x1, y1 + 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(annotated_frame)
        cv.imshow('Annotated Frame', annotated_frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv.destroyAllWindows()


model_path = 'C:/Users/Ramya M N/OneDrive/Desktop/SmartSpeedGuard_YOLOv8/runs/detect/train5/weights/best.pt'


# Example usage
if __name__ == "__main__":
    try:
        model = load_yolov8_model(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading YOLOv8 model: {e}")
        model = None
    if model is None:
        print("Error loading model. Exiting.")
        exit()

    # Specify your focal length and real width
    focal_length = 730  # example value, replace with your actual focal length
    real_width = 0.5  # example value in meters, replace with your actual known width


    video_path = "testing/hospital.mp4"
    output_path = "testing/hospital_detect.mp4"

    # Specify the closest_objects_info list with all relevant classes and speed limits
    closest_objects_info = [
        {"Class": "Pedestrian", "Speed Limit": 5},
        {"Class": "BumpyRoad", "Speed Limit": 10},
        {"Class": "ConstructionWork", "Speed Limit": 15},
        {"Class": "HospitalZone", "Speed Limit": 20},
        {"Class": "Obstacle", "Speed Limit": 5},
        {"Class": "SchoolZone", "Speed Limit": 25},
        {"Class": "Stop", "Speed Limit": 0},
    ]

    # Process the video
    process_video(video_path, model, focal_length, real_width, closest_objects_info, output_path)
