import cv2
import numpy as np
import face_recognition
import os
import pandas as pd
from datetime import datetime

# ---------- Attendance Function ----------

def mark_attendance(name):
    filename = "attendance.xlsx"
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d")
    time_string = now.strftime("%H:%M:%S")

    # If file exists, load it, otherwise create new DataFrame
    if os.path.exists(filename):
        try:
            df = pd.read_excel(filename, engine="openpyxl")
        except Exception:
            # If file is corrupted, recreate it
            df = pd.DataFrame(columns=["Name", "Date", "Time"])
    else:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])

    # Check if already marked today
    if not ((df["Name"] == name) & (df["Date"] == dt_string)).any():
        new_entry = pd.DataFrame([[name, dt_string, time_string]],
                                 columns=["Name", "Date", "Time"])
        df = pd.concat([df, new_entry], ignore_index=True)

        # Save back to Excel
        df.to_excel(filename, index=False, engine="openpyxl")
        print(f"{name} marked present at {time_string}")
    else:
        print(f"{name} already marked today.")

# ---------- Load Known Faces ----------
path = 'Images'  # Folder where you store face images
images = []
classNames = []
myList = os.listdir(path)

print("Training faces...")

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

def find_encodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        except:
            print("No face found in image, skipping.")
    return encodeList

encodeListKnown = find_encodings(images)
print("Encoding complete.")

# ---------- Webcam ----------
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # Reduce size for speed
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4  # Scale back up

            # Draw rectangle and name
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            mark_attendance(name)  # Store in Excel
        else:
            # If unknown face
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, "UNKNOWN", (x1+6, y2-6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            print("Unknown face detected, not marked.")

    cv2.imshow('Webcam', img)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()