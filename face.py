import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Load known faces
path = 'images'
images = []
classNames = []

for filename in os.listdir(path):
    img = cv2.imread(os.path.join(path, filename))
    if img is not None:
        images.append(img)
        classNames.append(os.path.splitext(filename)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img_rgb)
        if encodings:
            encodeList.append(encodings[0])
    return encodeList

encodeListKnown = findEncodings(images)

# Track attendance and consistency
marked_names = set()
detection_counts = {}  # Tracks consecutive detections

def markAttendance(name):
    with open('attendance.csv', 'a') as f:
        time_now = datetime.now().strftime('%H:%M:%S')
        f.write(f'{name},{time_now}\n')
    print(f"{name} marked present at {datetime.now().strftime('%H:%M:%S')}")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        print("Camera not working")
        break

    imgS = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    imgS_rgb = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(imgS_rgb)
    face_encodings = face_recognition.face_encodings(imgS_rgb, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(encodeListKnown, face_encoding)
        face_distances = face_recognition.face_distance(encodeListKnown, face_encoding)

        name = "Unknown"
        if True in matches:
            match_index = np.argmin(face_distances)
            name = classNames[match_index]

            # Count stable recognition
            if name not in detection_counts:
                detection_counts[name] = 1
            else:
                detection_counts[name] += 1

            # Confirm detection after 5 consistent frames
            if detection_counts[name] == 5 and name not in marked_names:
                markAttendance(name)
                marked_names.add(name)

        else:
            name = "Unknown"

        # Reset count for unknown faces
        for key in list(detection_counts.keys()):
            if key != name:
                detection_counts[key] = 0

        # Draw box
        y1, x2, y2, x1 = [v * 4 for v in face_location]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Attendance System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()