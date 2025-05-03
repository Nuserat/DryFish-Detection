import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torch
from torchvision import transforms
from torchcam.methods import EigenCAM
from torchcam.utils import overlay_mask

# Set Streamlit page config
st.set_page_config(
    page_title="Dry Fish Detection with EigenCAM",
    page_icon="🐟",
    layout="wide"
)

st.title("Dry Fish Detection using YOLOv Models and EigenCAM")
st.sidebar.title("⚙️ Settings")

# Model selection
model_options = {
    "YOLOv9": "yolov9.pt",
    "YOLOv10": "yolov10.pt",
    "YOLOv11": "yolov11.pt",
    "YOLOv12": "yolov12.pt"
}
selected_model_name = st.sidebar.selectbox("Select Model", list(model_options.keys()))
model_path = model_options[selected_model_name]

# Load YOLO model
@st.cache_resource
def load_model(path):
    return YOLO(path)

model = load_model(model_path)
st.success(f"✅ Model `{model_path}` loaded successfully.")

# Draw bounding boxes
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

# Get EigenCAM heatmap
def generate_eigen_cam(image_np, model):
    # Convert image to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    input_tensor = transform(image_np).unsqueeze(0)

    # Get backbone for CAM
    backbone = model.model.model.model  # YOLOv8 → TaskModule → Model → backbone
    target_layer = backbone[-2]         # Pick one of the last conv layers

    cam_extractor = EigenCAM(backbone, target_layer=target_layer)

    with torch.no_grad():
        _ = backbone(input_tensor)  # Forward pass

    cam = cam_extractor(torch.argmax(_[0], dim=1).item())  # Use highest score

    # Convert CAM and overlay
    cam_resized = cv2.resize(cam[0].numpy(), (image_np.shape[1], image_np.shape[0]))
    heatmap = overlay_mask(Image.fromarray(image_np), Image.fromarray((cam_resized * 255).astype(np.uint8)), alpha=0.5)
    return heatmap

# Upload image
st.subheader("📷 Upload an Image to Detect Dry Fish")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    if st.button("🔍 Detect Dry Fish"):
        with st.spinner("Running detection and generating EigenCAM..."):
            try:
                # Run detection
                results = model(image_np)
                result_image = draw_boxes(image_np, results[0])

                # Generate EigenCAM
                cam_image = generate_eigen_cam(image_np, model)

                # Display results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.subheader("Original Image")
                    st.image(image, use_column_width=True)
                with col2:
                    st.subheader("Detection Result")
                    st.image(result_image, use_column_width=True)
                with col3:
                    st.subheader("EigenCAM Visualization")
                    st.image(cam_image, use_column_width=True)

                count = len(results[0].boxes)
                if count > 0:
                    st.success(f"Detected {count} Dry Fish instance(s).")
                else:
                    st.info("No Dry Fish detected.")

            except Exception as e:
                st.error(f"❌ Error: {e}")

# About section
with st.expander("About this App"):
    st.write("""
    ### Dry Fish Detection with YOLOv Models and EigenCAM
    This app detects dry fish in images using YOLOv9,YOLOv10,YOLOv11,YOLOv12 and visualizes model focus using EigenCAM.

    #### Features:
    - Upload image for dry fish detection
    - Draws bounding boxes on detection
    - Overlays heatmap using EigenCAM for model interpretability

    #### Use cases:
    - Fish quality assessment
    - Fish categorization and research
    - Visual explainability in model decisions
    """)
