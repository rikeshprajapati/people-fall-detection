import cv2
import cvzone
import math
from ultralytics import YOLO

def fall_detection(x1, y1, x2, y2):
    height = y2 - y1
    width = x2 - x1
    return (height - width) < 0

# Load video and model
cap = cv2.VideoCapture('video_path')
model = YOLO('model_path')

# Load class names
classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for MP4
out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (980, 740))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (980, 740))
    results = model(frame)
    
    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_detect = int(box.cls[0])
            class_detect_name = classnames[class_detect]
            conf = math.ceil(confidence * 100)

            if conf > 80 and class_detect_name == 'person':
                cvzone.cornerRect(frame, [x1, y1, x2 - x1, y2 - y1], l=30, rt=6)
                cvzone.putTextRect(frame, f'{class_detect_name}', [x1 + 8, y1 - 12], thickness=2, scale=2)

                if fall_detection(x1, y1, x2, y2):
                    cvzone.putTextRect(frame, 'Fall-Detected', [x1, y1 - 50], thickness=2, scale=2)

    out.write(frame)  # Write the frame to the output video
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('t'):
        break

cap.release()
out.release()  # Release the VideoWriter object
cv2.destroyAllWindows()
