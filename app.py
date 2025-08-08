import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import tempfile
import time

st.set_page_config(
    page_title="Lung Cancer Detection",
    page_icon="🫁",
    layout="wide"
)

st.write("🚀 App starting... UI loaded.")

# --- Lazy imports ---
@st.cache_resource
def get_model(option):
    from model import create_model, load_pretrained_model
    if option == "Basic CNN":
        return create_model()
    return load_pretrained_model()

@st.cache_resource
def init_database():
    from utils import init_db
    try:
        init_db()
    except Exception as e:
        st.error(f"Database init failed: {e}")

# Sidebar
st.sidebar.markdown("## 🔧 Analysis Settings")
model_option = st.sidebar.selectbox(
    "Choose AI Model",
    ["Basic CNN", "InceptionV3 Transfer Learning"], index=0
)
visualization_option = st.sidebar.selectbox(
    "Choose Visualization",
    ["Prediction Confidence", "Class Activation Maps", "Feature Maps"], index=0
)

# Initialize DB (safe)
init_database()

# Load model only when needed
if 'model' not in st.session_state or st.session_state.get('model_option') != model_option:
    with st.spinner("Loading model... Please wait"):
        st.session_state.model = get_model(model_option)
    st.session_state.model_option = model_option

# Main UI
st.markdown("# 🫁 Lung Cancer Detection AI")
st.markdown("### Early detection saves lives. Upload medical images or try sample cases.")

uploaded_file = st.file_uploader(
    "Choose a lung CT scan or X-ray image file", 
    type=["jpg", "jpeg", "png", "dcm"]
)

if uploaded_file:
    from utils import read_dicom_file, display_dicom_info
    from preprocessing import preprocess_image, ensure_color_channels
    from visualization import visualize_prediction, visualize_activation_maps, visualize_feature_maps, visualize_model_performance
    from utils import calculate_prediction_confidence, add_to_history

    col1, col2 = st.columns(2)

    try:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        if file_ext == 'dcm':
            with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name
            dicom_data, pixel_array = read_dicom_file(temp_path)
            os.unlink(temp_path)
            with col1:
                st.image(pixel_array, caption="Original DICOM Image", use_container_width=True)
                display_dicom_info(dicom_data)
            processed = preprocess_image(pixel_array)
        else:
            img = Image.open(uploaded_file)
            with col1:
                st.image(img, caption="Original Image", use_container_width=True)
            img_arr = ensure_color_channels(np.array(img))
            processed = preprocess_image(img_arr)

        with st.spinner("Analyzing image..."):
            start = time.time()
            pred = st.session_state.model.predict(np.expand_dims(processed, axis=0))
            end = time.time()

        label, conf = calculate_prediction_confidence(pred[0][0])
        with col2:
            if label == "Cancer":
                st.error(f"⚠️ {label} detected - Confidence: {conf:.2f}%")
            else:
                st.success(f"✅ {label} - Confidence: {conf:.2f}%")
            st.info(f"Processing time: {(end - start)*1000:.1f} ms")

            if visualization_option == "Prediction Confidence":
                visualize_prediction(pred[0][0])
            elif visualization_option == "Class Activation Maps":
                visualize_activation_maps(st.session_state.model, processed)
            elif visualization_option == "Feature Maps":
                visualize_feature_maps(st.session_state.model, processed)

            visualize_model_performance(model_option)

    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>⚠️ <b>Medical Disclaimer:</b> "
    "This tool is for educational purposes only and not a substitute for professional diagnosis.</div>",
    unsafe_allow_html=True
)
