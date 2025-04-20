import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

# Setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Output CSV file
output_file = "emotion_dataset.csv"
if not os.path.exists(output_file):
    df = pd.DataFrame()
    df.to_csv(output_file, index=False)

# Define key-to-emotion mapping
emotion_map = {
    'h': 'happy',
    's': 'sad',
    'a': 'angry',
    'n': 'neutral',
    'f': 'fear',
    'd': 'disgust',
    'u': 'surprise'
}

print("[INFO] Press keys to record emotion: h=happy, s=sad, a=angry, n=neutral, f=fear, d=disgust, u=surprise | q to quit")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(frame_rgb)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            landmarks = []
            for lm in face_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])  # (x, y) only for simplicity

            # Draw face mesh
            mp.solutions.drawing_utils.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
            )

            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()
            elif chr(key) in emotion_map:
                label = emotion_map[chr(key)]
                print(f"[+] Captured {label} expression.")
                row = landmarks + [label]
                df = pd.DataFrame([row])
                df.to_csv(output_file, mode='a', index=False, header=not os.path.isfile(output_file))

    cv2.imshow("Facial Emotion Collector", frame)

cap.release()
cv2.destroyAllWindows()
