import streamlit as st
import face_recognition
import os
import cv2
import numpy as np
from datetime import datetime
import pandas as pd
from PIL import Image

# Title
st.title("🎓 SmartC - Face Recognition Attendance System")

# Load known faces
known_face_encodings = []
known_face_names = []

KNOWN_FACES_DIR = "known_faces"
for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(KNOWN_FACES_DIR, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(os.path.splitext(filename)[0])

# Upload image
uploaded_image = st.file_uploader("Upload a student photo", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert PIL to numpy array
    image_np = np.array(image)

    # Detect face
    uploaded_encoding = face_recognition.face_encodings(image_np)

    if uploaded_encoding:
        result = face_recognition.compare_faces(known_face_encodings, uploaded_encoding[0])
        distances = face_recognition.face_distance(known_face_encodings, uploaded_encoding[0])

        best_match_index = np.argmin(distances)
        if result[best_match_index]:
            name = known_face_names[best_match_index]
            st.success(f"✅ Attendance marked for: {name}")

            # Save to attendance log
            now = datetime.now()
            dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
            with open("attendance.csv", "a") as f:
                f.write(f"{name},{dt_string}\n")

        else:
            st.error("❌ No match found.")
    else:
        st.warning("😕 No face detected in uploaded image.")

# Download attendance log
if os.path.exists("attendance.csv"):
    df = pd.read_csv("attendance.csv", names=["Name", "Time"])
    st.download_button("📥 Download Attendance Log", df.to_csv(index=False), "attendance.csv", "text/csv")
