import cv2
import numpy as np
import os
print(os.getcwd())

file_path = r"C:\Users\praveen v\Desktop\objectdet\coco.names.html"
print("File Exists:", os.path.exists(file_path))

cfg_path = r"C:\Users\praveen v\Desktop\objectdet\yolov3.cfg.txt"
weights_path = r"C:\Users\praveen v\Desktop\objectdet\yolov3.weights"
print("CFG Path Exists:", os.path.exists(cfg_path))
print("Weights Path Exists:", os.path.exists(weights_path))

net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)



# Load YOLO



layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Function to detect objects
def detect_objects(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    detected_objects = [(boxes[i], class_ids[i]) for i in indexes.flatten()] if len(indexes) > 0 else []
    return detected_objects

# Open the camera feed
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    # Detect objects in the frame
    detections = detect_objects(frame)

    # Draw bounding boxes and labels
    for (box, class_id) in detections:
        x, y, w, h = box
        label = str(classes[class_id])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Live Object Detection", frame)

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
