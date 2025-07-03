import streamlit as st
import face_recognition
from PIL import Image
import numpy as np
import io

st.title("🔍 Face Recognition App")

# Upload known face
known_img_file = st.file_uploader("Upload a known face image", type=["jpg", "jpeg", "png"])
# Upload image to check
check_img_file = st.file_uploader("Upload another image to check", type=["jpg", "jpeg", "png"])

if known_img_file and check_img_file:
    known_image = face_recognition.load_image_file(known_img_file)
    check_image = face_recognition.load_image_file(check_img_file)

    try:
        known_encoding = face_recognition.face_encodings(known_image)[0]
        check_encoding = face_recognition.face_encodings(check_image)[0]

        results = face_recognition.compare_faces([known_encoding], check_encoding)
        distance = face_recognition.face_distance([known_encoding], check_encoding)[0]

        col1, col2 = st.columns(2)
        col1.image(known_image, caption="Known Image", use_container_width=True)
        col2.image(check_image, caption="Image to Check", use_container_width=True)

        if results[0]:
            st.success(f"✅ Match found! (Distance: {distance:.2f})")
        else:
            st.error(f"❌ No match. (Distance: {distance:.2f})")

    except IndexError:
        st.warning("Face not detected in one of the images. Please try different images.")
