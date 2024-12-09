import cv2
import numpy as np

# Load YOLOv3 model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
classes = []

with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Define parameters for safe distance
SAFE_DISTANCE = 100  # Adjust according to requirements

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use your video source here (0 for webcam)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # Prepare image for YOLO
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Confidence threshold
                # Object details
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                label = str(classes[class_id])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {int(confidence * 100)}%", 
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Avoidance mechanism
                distance = SAFE_DISTANCE - (h * 0.1)  # Basic distance estimation
                if distance < SAFE_DISTANCE:
                    if center_x < width // 3:
                        avoidance_direction = "Move Right"
                    elif center_x > 2 * width // 3:
                        avoidance_direction = "Move Left"
                    else:
                        avoidance_direction = "Stop!"

                    # Display avoidance warning
                    cv2.putText(frame, avoidance_direction, (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    print(f"Warning: {avoidance_direction}")

    # Display the output
    cv2.imshow("Navigation Glasses", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
