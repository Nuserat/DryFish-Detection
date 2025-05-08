import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

os.environ["nnTh1SJfTal1EhvjW3Fi"] = "TRUE"

# Set page config
st.set_page_config(page_title="Fish Species Detection", page_icon="🐟", layout="wide")
st.title("Fish Species Detection using Multiple YOLO Models")

# Sidebar settings
st.sidebar.title("Settings")

# Define class names
CLASS_NAMES = [
    "Corica soborna", "Jamuna ailia", "Clupeidae", "Shrimp", "Chepa",
    "Chela", "Swamp barb", "Silond catfish", "Pale Carplet", "Bombay Duck", "Four-finger threadfin"
]

# Available models
model_options = {
    "YOLOv9": "yolov9.pt",
    "YOLOv10": "yolov10.pt",
    "YOLOv11": "yolov11.pt",
    "YOLOv12": "yolov12.pt"
}
selected_model_name = st.sidebar.selectbox("Select YOLO Model", list(model_options.keys()))
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# Load model
@st.cache_resource
def load_model(model_path):
    from ultralytics import YOLO
    return YOLO(model_path)

model_path = model_options[selected_model_name]
model = load_model(model_path)

# Function to draw bounding boxes and labels
def draw_boxes(image, results):
    annotated_img = image.copy()

    if results and len(results.boxes) > 0:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Get class and confidence
            class_id = int(box.cls[0]) if box.cls is not None else -1
            class_name = CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else "Unknown"
            conf = float(box.conf[0])
            label = f"{class_name}: {conf:.2f}"

            # Draw box
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # Draw label
            cv2.putText(annotated_img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return annotated_img

# Image upload section
st.header("Upload Image for Fish Detection")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    if st.button("Detect Fish"):
        with st.spinner("Processing image..."):
            try:
                results = model(image_np, conf=confidence_threshold)
                result_image = draw_boxes(image_np, results[0])

                # Show original and result images
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original Image")
                    st.image(image, use_column_width=True)
                with col2:
                    st.subheader("Detection Results")
                    st.image(result_image, use_column_width=True)

                # Summary
                if len(results[0].boxes) > 0:
                    detected_classes = [CLASS_NAMES[int(box.cls[0])] for box in results[0].boxes]
                    st.success(f"Detected {len(results[0].boxes)} object(s): {', '.join(detected_classes)}")
                else:
                    st.info("No fish detected in this image.")
            except Exception as e:
                st.error(f"Error processing image: {e}")
