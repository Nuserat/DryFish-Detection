import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Set environment variable
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Page config
st.set_page_config(page_title="Dry Fish Detection using YOLOv Models", page_icon="🛣️", layout="wide")
st.title("Dry Fish Detection using Multiple YOLO Models")

# Sidebar for settings
st.sidebar.title("Settings")

# Model options
model_options = {
    "YOLOv9": "yolov9.pt",
    "YOLOv10": "yolov10.pt",
    "YOLOv11": "yolov11.pt",
    "YOLOv12": "yolov12.pt"
}
selected_model_name = st.sidebar.selectbox("Select YOLO Model", list(model_options.keys()))
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

@st.cache_resource
def load_model(model_path):
    from ultralytics import YOLO
    return YOLO(model_path)

# Load the selected model
model_path = model_options[selected_model_name]
model = load_model(model_path)

# Function to draw circles
def draw_circles(image, results):
    annotated_img = image.copy()
    if results and len(results.boxes) > 0:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            radius = max((x2 - x1), (y2 - y1)) // 2

            cv2.circle(annotated_img, (center_x, center_y), radius, (0, 0, 255), 3)
            conf = float(box.conf[0])
            label = f"Pothole: {conf:.2f}"
            cv2.putText(annotated_img, label, (center_x - 10, center_y - radius - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return annotated_img

# Upload and detect
st.header("Upload Image for Detection")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    if st.button("Detect Dry Fish"):
        with st.spinner("Processing image..."):
            try:
                results = model(image_np, conf=confidence_threshold)
                result_image = draw_circles(image_np, results[0])
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original Image")
                    st.image(image, use_column_width=True)
                with col2:
                    st.subheader("Detection Results")
                    st.image(result_image, use_column_width=True)

                if len(results[0].boxes) > 0:
                    st.success(f"Detected {len(results[0].boxes)} Dry Fish(s)")
                else:
                    st.info("No Dry Fish detected in this image.")
            except Exception as e:
                st.error(f"Error processing image: {e}")
