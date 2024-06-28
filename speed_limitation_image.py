import math
import cv2 as cv
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
        return adjusted_speed
    
# Perform object detection (without distance calculation)
def object_detector_without_distance(image, model, confidence_threshold=0.5):
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
                data_list.append([label, object_width_pixels, (x1, y1 - 10)])

    return data_list, image

# Object detector function
def object_detector(image, model, focal_length, real_width, closest_objects_info, initial_speed, confidence_threshold=0.5):
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

    return data_list, image



model_path = 'C:/Users/Ramya M N/OneDrive/Desktop/SmartSpeedGuard_YOLOv8/runs/detect/train5/weights/best.pt'
# Reference object known parameters (you need to measure these)
KNOWN_DISTANCE = 2.0  # Known distance to reference object in meters
KNOWN_WIDTH = 0.5  # Known width of reference object in meters

# Measured distance for adjusting speeds
MEASURED_DISTANCE = 3.5

# Closest object class information with respective speed limits
closest_objects_info = [
    {"Class": "BumpyRoad", "Speed Limit": 10},
    {"Class": "ConstructionWork", "Speed Limit": 20},
    {"Class": "HospitalZone", "Speed Limit": 25},
    {"Class": "Obstacle", "Speed Limit": 30},
    {"Class": "Pedestrian", "Speed Limit": 5},
    {"Class": "SchoolZone", "Speed Limit": 15},
    {"Class": "Stop", "Speed Limit": 0},
]
initial_speed = 50 # Initial speed in km/h
# Load the model
try:
    model = load_yolov8_model(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading YOLOv8 model: {e}")
    model = None

if model:
    # Load and process your reference image
    ref_image_path = "C:/Users/Ramya M N/OneDrive/Desktop/SmartSpeedGuard_YOLOv8/reference_ima/img2.jpg"
    ref_image = cv.imread(ref_image_path)

    if ref_image is None:
        print(f"Error loading reference image from {ref_image_path}")
    else:
        # Perform object detection on the reference image to find the reference width in pixels
        ref_object_data, _ = object_detector_without_distance(ref_image, model, confidence_threshold=0.5)
        if ref_object_data:
            # Assuming the reference object is the first detected object
            ref_object_width_pixels = ref_object_data[0][1]

            # Calculate the focal length
            focal_length = focal_length_finder(KNOWN_DISTANCE, KNOWN_WIDTH, ref_object_width_pixels)
            print(f"Focal Length: {focal_length}")

            # Load and process your test image
            image_path = "datasets/test/images/obbar6-resized_jpg.rf.de17ef49844908089bd086eee4eee96c.jpg"
            image = cv.imread(image_path)

            if image is None:
                print(f"Error loading image from {image_path}")
            else:
                # Perform object detection and distance estimation
                initial_speed = 50  # Initial speed in km/h
                object_data, detected_image = object_detector(image, model, focal_length, KNOWN_WIDTH, closest_objects_info, initial_speed)

                # Draw object details on the image
                for obj in object_data:
                    x1, y1, x2, y2 = obj["Bounding Box"]
                    cv.putText(detected_image, f"Class: {obj['Class']}", (x1, y1 - 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv.putText(detected_image, f"Distance: {obj['Distance (meters)']:.2f} m", (x1, y1 - 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv.putText(detected_image, f"Speed Limit: {obj['Speed Limit (km/h)']} km/h", (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv.putText(detected_image, f"Adjusted Speed: {obj['Adjusted Speed (km/h)']:.2f} km/h", (x1, y1 + 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Save or display the resulting image with annotations
                result_image_path = "C:/Users/Ramya M N/OneDrive/Desktop/SmartSpeedGuard_Check/SmartSpeedGuard_YOLOv8/datasets/train/images/sk6-resized_jpg.rf.ccb75f1a5d5b9e77b2b29d5c69e26254_annotated.jpg"
                cv.imwrite(result_image_path, detected_image)
                print(f"Result image saved to {result_image_path}")

                # Open the annotated image
                os.startfile(result_image_path)

                # Generate HTML report
                html_report_path = "C:/Users/Ramya M N/OneDrive/Desktop/SmartSpeedGuard_Check/SmartSpeedGuard_YOLOv8/datasets/train/images/sk6-resized_jpg.rf.ccb75f1a5d5b9e77b2b29d5c69e26254_report.html"
                template = Template("""
                <html>
                <head><title>Detection Report</title></head>
                <body>
                    <h1>Detection Report</h1>
                    <img src="data:image/jpeg;base64,{{ image_base64 }}" alt="Annotated Image"/>
                    <table border="1">
                        <tr>
                            <th>Class</th>
                            <th>Object Width (pixels)</th>
                            <th>Distance (meters)</th>
                            <th>Speed Limit (km/h)</th>
                            <th>Adjusted Speed (km/h)</th>
                        </tr>
                        {% for obj in object_data %}
                        <tr>
                            <td>{{ obj['Class'] }}</td>
                            <td>{{ obj['Object Width (pixels)'] }}</td>
                            <td>{{ obj['Distance (meters)'] }}</td>
                            <td>{{ obj['Speed Limit (km/h)'] }}</td>
                            <td>{{ obj['Adjusted Speed (km/h)'] }}</td>
                        </tr>
                        {% endfor %}
                    </table>
                </body>
                </html>
                """)

                # Convert image to base64 for embedding in HTML
                retval, buffer = cv.imencode('.jpg', detected_image)
                image_base64 = base64.b64encode(buffer).decode('utf-8')

                # Prepare data for rendering in HTML template
                data = {
                    'image_base64': image_base64,
                    'object_data': object_data
                }

                # Render HTML report
                rendered_report = template.render(data)

                # Write rendered HTML report to file
                with open(html_report_path, 'w') as report_file:
                    report_file.write(rendered_report)

                print(f"HTML report generated: {html_report_path}")

                # Open the HTML report
                os.startfile(html_report_path)

else:
    print("No model loaded. Exiting.")
