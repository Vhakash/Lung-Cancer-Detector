import streamlit as st
from preprocessing import preprocess_image, ensure_color_channels
from utils import init_db, add_to_history, calculate_prediction_confidence, read_dicom_file, display_dicom_info
from visualization import visualize_prediction, visualize_model_performance
from model import create_model, load_pretrained_model
import numpy as np
import cv2

st.set_page_config(page_title="Lung Cancer Detector", layout="wide")

# Initialize DB safely
init_db()

st.title("🫁 Lung Cancer Detection System")

# Model selection
model_type = st.selectbox("Select Model", ["Basic CNN", "InceptionV3 Transfer Learning"])

@st.cache_resource
def load_model(selected_model):
    if selected_model == "Basic CNN":
        return create_model()
    else:
        return load_pretrained_model()

# File upload
uploaded_file = st.file_uploader("Upload a Lung Scan (JPG, PNG, or DICOM)", type=["jpg", "png", "dcm"])

if uploaded_file:
    file_bytes = uploaded_file.read()
    if uploaded_file.name.lower().endswith(".dcm"):
        image, ds = read_dicom_file(file_bytes)
        display_dicom_info(ds)
    else:
        file_arr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(file_arr, cv2.IMREAD_UNCHANGED)

    image = ensure_color_channels(image)
    processed_image = preprocess_image(image)

    # Load model lazily
    model = load_model(model_type)

    # Simulate prediction
    prediction = model.predict(np.expand_dims(processed_image, axis=0))[0][0]
    label, confidence = calculate_prediction_confidence(prediction)

    st.subheader(f"Prediction: {label} ({confidence}%)")
    visualize_prediction(prediction)

    # Save to DB (safe)
    try:
        add_to_history(file_bytes, model_type, label, confidence, "None")
    except Exception as e:
        st.warning(f"Could not save analysis: {e}")

    st.subheader("Model Performance Overview")
    visualize_model_performance(model_type)
