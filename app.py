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

# Fix for torch.classes error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load model with caching
@st.cache_resource
def load_model():
    try:
        from ultralytics import YOLO
        return YOLO("best.pt")  # Ensure best.pt is in your project directory
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()

# Draw bounding boxes on image
def draw_boxes(image, results):
    annotated_img = image.copy()

    if results and len(results.boxes) > 0:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            # Draw red rectangle (bounding box)
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Add confidence label
            label = f"Dry Fish : {conf:.2f}"
            cv2.putText(annotated_img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return annotated_img

# Sidebar settings
st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# Image upload section
st.subheader("Upload an Image for Dry Fish  Detection")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    if st.button("Detect Dry Fish "):
        if model:
            with st.spinner("Processing image..."):
                try:
                    results = model(image_np, conf=confidence_threshold)
                    result_image = draw_boxes(image_np, results[0])

                    # Display results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Original Image")
                        st.image(image, use_column_width=True)
                    with col2:
                        st.subheader("Detection Results")
                        st.image(result_image, use_column_width=True)

                    # Detection count
                    count = len(results[0].boxes)
                    if count > 0:
                        st.success(f"Detected {count} Dry Fish (s)")
                    else:
                        st.info("No Dry Fish  detected in this image.")
                except Exception as e:
                    st.error(f"Error processing image: {e}")
        else:
            st.error("Model failed to load. Please refresh the page.")

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
