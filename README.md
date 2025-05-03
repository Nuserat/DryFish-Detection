# Dry Fish Detection and Explainable AI

This repository contains a web-based application for Detecting dry fish images and providing explainable AI (XAI) visualizations using Eigen-CAM.

## Demo App: 
- Link: https://dry-fish-xai-caeyevtxdywvun96usmgbq.streamlit.app/

## Features
- **Dry Fish Detection**: Uses a Yolov models trained on 11 different dry fish categories.
- **Explainable AI (XAI)**: Provides  Eigen-CAM visualizations for model interpretability.
- **User-Friendly UI**: Built with Streamlit for easy image upload and analysis.
- **Custom Styling**: Right-side background is light gray with black text for better readability.

### Clone the Repository
```sh
git clone https://github.com/your-username/dry-fish-classification.git
cd dry-fish-classification
```

### Install Dependencies
```sh
pip install -r requirements.txt
```

### Run the Application
```sh
streamlit run app.py
```

## Model
- **Architecture**: MobileNetV2
- **Dataset**: 11 categories of dry fish images
- **Explainability**: Uses Grad-CAM techniques to highlight important regions for classification

## Usage
1. Upload an image of a dry fish.
2. The model predicts the category of the fish.
3. XAI visualizations ( Eigen-CAM) are generated to show model decision areas.
4. Predictions and visualizations are displayed in the interface.

## Folder Structure
```
📂 dry-fish-classification
│── 📄 app.py  # Main application file
│── 📄 best.pth  # Trained model
│── 📂 models  # Contains model-related scripts
│── 📂 static  # Static assets like images, CSS
│── 📂 templates  # HTML templates for UI customization
│── 📂 requirements.txt  # Required dependencies
│── 📂 README.md  # Project documentation
```
