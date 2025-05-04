import os
os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNINGS"] = "true"
os.environ["PYTORCH_NO_WATCH"] = "true" 
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torch

# XAI libraries
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

# Set page configuration
st.set_page_config(
    page_title="Dry Fish Detection",
    page_icon="\U0001F41F",
    layout="wide"
)

# Title of the app
st.title("Dry Fish Detection using YOLOv Models")
st.sidebar.title("⚙️ Settings")

# Model selection dropdown
model_options = {
    "YOLOv9": "yolov9.pt",
    "YOLOv10": "yolov10.pt",
    "YOLOv11": "yolov11.pt",
    "YOLOv12": "yolov12.pt"
}
selected_model_name = st.sidebar.selectbox("Select Model", list(model_options.keys()))
model_path = model_options[selected_model_name]

# Load YOLO model with caching
@st.cache_resource
def load_model(path):
    model = YOLO(path)
    model.eval()
    return model

model = load_model(model_path)
st.success(f"Model `{model_path}` loaded successfully.")

# Draw bounding boxes around detections
def draw_boxes(image, results):
    annotated_img = image.copy()
    if results and len(results.boxes) > 0:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = f"Dry Fish: {conf:.2f}"
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated_img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return annotated_img

# Function to select target layers for CAM
def get_target_layers(model):
    try:
        return [model.model.model[-2], model.model.model[-3], model.model.model[-4]]
    except Exception as e:
        st.warning("Could not automatically select target layers for CAM. Adjust manually if needed.")
        return [model.model.model[-2]]

# Function to generate EigenCAM
@torch.no_grad()
def generate_eigencam(model, original_image_np):
    rgb_img = original_image_np.astype(np.float32) / 255.0
    input_tensor = preprocess_image(rgb_img, mean=[0, 0, 0], std=[1, 1, 1])

    target_layers = get_target_layers(model)
    cam = EigenCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
    grayscale_cam = cam(input_tensor=input_tensor)[0, :, :]

    cam.clear_hooks()  # 🧹 Clean up hooks explicitly

    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return cam_image

# Image upload section
st.subheader("📤 Upload an Image to Detect Dry Fish")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    if st.button("🔍 Detect Dry Fish"):
        with st.spinner("Processing..."):
            try:
                results = model(image_np)
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
                    st.success(f"Detected {count} Dry Fish instance(s).")
                else:
                    st.info("No Dry Fish detected.")

                # Show EigenCAM
                eigencam_image = generate_eigencam(model, image_np)

                with st.expander("🧠 Explainability - EigenCAM"):
                    col3, col4 = st.columns(2)
                    with col3:
                        st.subheader("EigenCAM Heatmap")
                        st.image(eigencam_image, use_column_width=True)
                    with col4:
                        st.subheader("Overlay Comparison")
                        overlay = np.hstack((image_np, eigencam_image))
                        st.image(overlay, use_column_width=True)

            except Exception as e:
                st.error(f"Error during detection: {e}")

# About section
with st.expander("About this App"):
    st.write("""
    ### Dry Fish Detection App (Image Upload Only)
    This app uses YOLOv Models trained for detecting dry fish from images.

    #### Features:
    - Upload an image for dry fish detection
    - Bounding boxes with confidence scores
    - EigenCAM for model explainability

    #### How it works:
    The model processes the uploaded image and detects regions containing dry fish using pre-trained YOLO weights.

    #### Use cases:
    - Quality control in seafood processing
    - Marine life classification
    - Research and monitoring in fisheries
    """)
