import psycopg2
import streamlit as st
import os
import io
import pydicom
import numpy as np
from datetime import datetime

# ---------------------------
# Database Connection Helper
# ---------------------------
def get_connection_string():
    # Check Streamlit secrets first (deployment)
    if "postgres" in st.secrets:
        return st.secrets["postgres"].get("connection_string", "")
    # Fallback for local development
    return os.getenv("POSTGRES_URL", "")

def init_db():
    conn_str = get_connection_string()
    if not conn_str:
        st.warning("Database connection not configured. History features disabled.")
        return False

    try:
        conn = psycopg2.connect(conn_str)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_history (
                id SERIAL PRIMARY KEY,
                image BYTEA,
                model_type TEXT,
                prediction_label TEXT,
                confidence FLOAT,
                enhancement TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        st.warning(f"Database unavailable: {e}")
        return False

def add_to_history(image_bytes, model_type, prediction_label, confidence, enhancement):
    conn_str = get_connection_string()
    if not conn_str:
        return
    try:
        conn = psycopg2.connect(conn_str)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO analysis_history (image, model_type, prediction_label, confidence, enhancement)
            VALUES (%s, %s, %s, %s, %s)
        """, (psycopg2.Binary(image_bytes), model_type, prediction_label, confidence, enhancement))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        st.warning(f"Could not save to database: {e}")

def get_analysis_history():
    conn_str = get_connection_string()
    if not conn_str:
        return []
    try:
        conn = psycopg2.connect(conn_str)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, model_type, prediction_label, confidence, enhancement, timestamp
            FROM analysis_history
            ORDER BY timestamp DESC
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return rows
    except Exception as e:
        st.warning(f"Could not retrieve history: {e}")
        return []

def clear_analysis_history():
    conn_str = get_connection_string()
    if not conn_str:
        return
    try:
        conn = psycopg2.connect(conn_str)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM analysis_history")
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        st.warning(f"Could not clear history: {e}")

# ---------------------------
# DICOM Handling
# ---------------------------
def read_dicom_file(file_bytes):
    file_stream = io.BytesIO(file_bytes)
    ds = pydicom.dcmread(file_stream)
    image = ds.pixel_array.astype(float)

    image = (np.maximum(image, 0) / image.max()) * 255.0
    image = np.uint8(image)
    return image, ds

def display_dicom_info(ds):
    st.write("**Patient Name:**", getattr(ds, "PatientName", "Unknown"))
    st.write("**Patient ID:**", getattr(ds, "PatientID", "Unknown"))
    st.write("**Modality:**", getattr(ds, "Modality", "Unknown"))
    st.write("**Study Date:**", getattr(ds, "StudyDate", "Unknown"))

# ---------------------------
# Prediction Confidence
# ---------------------------
def calculate_prediction_confidence(prediction):
    label = "Cancer" if prediction >= 0.5 else "Normal"
    confidence = round(prediction * 100, 2) if label == "Cancer" else round((1 - prediction) * 100, 2)
    return label, confidence
