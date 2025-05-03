import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Dry Fish Detection",
    page_icon=" ",
    layout="wide"
)

# Title of the app
st.title("Dry Fish  Detection using YOLOv8")
st.sidebar.title("⚙️ Settings")

# Allow user to select from multiple models
model_options = {
    "YOLOv8 Small": "yolov8s.pt",
    "YOLOv8 Medium": "yolov8m.pt",
    "Custom Trained (Potholes)": "best.pt"
}


model_choice = st.sidebar.selectbox("Select Model Type", list(model_options.keys()))
model_path = model_options[model_choice]

# Confidence slider
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# Load the selected model
@st.cache_resource
def load_model(path):
    return YOLO(path)

model = load_model(model_path)

st.success(f"✅ Model `{model_path}` loaded successfully.")

# Upload image section
st.subheader("📷 Upload an Image to Detect Potholes")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Function to draw boxes
def draw_boxes(image, results):
    annotated_img = image.copy()

    if results and len(results.boxes) > 0:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = f"Pothole: {conf:.2f}"

            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated_img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return annotated_img

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    if st.button("🔍 Detect Potholes"):
        with st.spinner("Processing..."):
            results = model(image_np, conf=confidence_threshold)
            result_image = draw_boxes(image_np, results[0])

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)
            with col2:
                st.subheader("Detection Result")
                st.image(result_image, use_column_width=True)

            count = len(results[0].boxes)
            if count > 0:
                st.success(f"Detected {count} pothole(s).")
            else:
                st.info("No potholes detected.")

# About section
with st.expander("About this App"):
    st.write("""
    ### Dry Fish  Detection App (Image Upload Only)
    This app uses a YOLOv8 model trained for Dry Fish  detection.

    #### Features:
    - Upload an image to detect Dry Fish 
    - Bounding boxes highlight Dry Fish 
    - Adjustable confidence threshold

    #### How it works:
    The model scans the image for Dry Fish and draws red boxes with confidence scores.

    #### Use cases:
    - Road quality reporting
    - Civil engineering
    - Research and development
    """)
