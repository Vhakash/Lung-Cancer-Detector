import streamlit as st
import pydicom
import numpy as np
import pandas as pd
from datetime import datetime
from preprocessing import normalize_dicom_image
import time
import psycopg2
from psycopg2.extras import RealDictCursor
from PIL import Image
import os

def get_db_connection():
    """"Establish a connection to the postgreSQL database."""
    conn = psycopg2.connect(**st.secrets["postgres"])
    return conn

def init_db():
    """Initialize the PostgreSQL database and create the history table."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
              id SERIAL PRIMARY KEY,
              timestamp TIMESTAMP,
              image_path VARCHAR(255),
              model_type VARCHAR(50),
              prediction_value REAL,
              prediction_label VARCHAR(50),
              confidence REAL,
              enhancement VARCHAR(50)
              );
''')
    conn.commit()
    c.close()
    conn.close()

    if not os.path.exists('history_images'):
        os.makedirs('history_images')

def add_to_history(image, model_type, prediction, enhancement = None):
    label, confidence = calculate_prediction_confidence(prediction[0][0])

    #save the image file
    current_time = datetime.now()
    timestamp_str = current_time.strftime("%Y%m%d_%H%M%S")
    image_path = f"history_images/{timestamp_str}.png"
    Image.fromarray((image * 255).astype(np.uint8)).save(image_path)

    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        '''
        INSERT INTO history (timestamp, image_path, model_type, prediction_value, prediction_label, confidence, enhancement)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ''',
        (current_time, image_path, model_type, float(prediction[0][0]), label, float(confidence), enhancement)
    )
    conn.commit()
    c.close()
    conn.close()        

def get_analysis_history():
    conn = get_db_connection()
    c = conn.cursor(cursor_factory = RealDictCursor)
    c.execute("SELECT * FROM history ORDER BY timestamp DESC LIMIT 10")
    history = c.fetchall()
    c.close()
    conn.close()
    return history

def clear_analysis_history():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("TRUNCATE TABLE history RESTART IDENTITY")
    conn.commit()
    c.close()
    conn.close()

    #optional : also delete saved image files
    for f in os.listdir('history_images'):
        os.remove(os.path.join('history_images', f))    

def read_dicom_file(file_path):
    """Read and process DICOM file"""
    try:
        # Read DICOM file
        dicom_data = pydicom.dcmread(file_path)
        
        # Get pixel array
        pixel_array = dicom_data.pixel_array
        
        # Normalize for display
        normalized_array = normalize_dicom_image(pixel_array)
        
        return dicom_data, normalized_array
    
    except Exception as e:
        st.error(f"Error reading DICOM file: {str(e)}")
        return None, None

def display_dicom_info(dicom_data):
    """Display DICOM metadata information"""
    if dicom_data is None:
        return
    
    st.subheader("DICOM Information")
    
    # Extract relevant DICOM tags
    info_dict = {}
    
    # Patient information
    if hasattr(dicom_data, 'PatientName') and dicom_data.PatientName:
        info_dict['Patient Name'] = str(dicom_data.PatientName)
    if hasattr(dicom_data, 'PatientID') and dicom_data.PatientID:
        info_dict['Patient ID'] = dicom_data.PatientID
    if hasattr(dicom_data, 'PatientAge') and dicom_data.PatientAge:
        info_dict['Patient Age'] = dicom_data.PatientAge
    if hasattr(dicom_data, 'PatientSex') and dicom_data.PatientSex:
        info_dict['Patient Sex'] = dicom_data.PatientSex
    
    # Study information
    if hasattr(dicom_data, 'StudyDate') and dicom_data.StudyDate:
        info_dict['Study Date'] = dicom_data.StudyDate
    if hasattr(dicom_data, 'Modality') and dicom_data.Modality:
        info_dict['Modality'] = dicom_data.Modality
    if hasattr(dicom_data, 'BodyPartExamined') and dicom_data.BodyPartExamined:
        info_dict['Body Part'] = dicom_data.BodyPartExamined
    
    # Image information
    if hasattr(dicom_data, 'Rows') and dicom_data.Rows:
        info_dict['Image Height'] = dicom_data.Rows
    if hasattr(dicom_data, 'Columns') and dicom_data.Columns:
        info_dict['Image Width'] = dicom_data.Columns
    if hasattr(dicom_data, 'PixelSpacing') and dicom_data.PixelSpacing:
        info_dict['Pixel Spacing'] = f"{dicom_data.PixelSpacing[0]:.2f} x {dicom_data.PixelSpacing[1]:.2f} mm"
    
    # Display information in a nice format
    if info_dict:
        df = pd.DataFrame(list(info_dict.items()), columns=['Property', 'Value'])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No metadata available for this DICOM file.")

def calculate_prediction_confidence(prediction_value):
    """Calculate prediction label and confidence percentage"""
    # prediction_value is between 0 and 1
    # 0 = Normal, 1 = Cancer
    
    if prediction_value > 0.5:
        label = "Cancer"
        confidence = prediction_value * 100
    else:
        label = "Normal"
        confidence = (1 - prediction_value) * 100
    
    return label, confidence

def compare_model_performances():
    """Compare performance metrics between different models"""
    # Simulated performance data
    models_data = {
        'Model Type': ['Basic CNN', 'InceptionV3 Transfer Learning'],
        'Accuracy': [0.87, 0.94],
        'Precision': [0.84, 0.91],
        'Recall': [0.82, 0.89],
        'F1-Score': [0.83, 0.90],
        'AUC': [0.88, 0.95],
        'Processing Time (ms)': [120, 180]
    }
    
    df = pd.DataFrame(models_data)
    return df

def format_file_size(size_bytes):
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB"]
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"

def validate_image_format(file):
    """Validate if uploaded file is a supported image format"""
    allowed_extensions = ['jpg', 'jpeg', 'png', 'dcm']
    file_extension = file.name.split('.')[-1].lower()
    return file_extension in allowed_extensions

def calculate_processing_stats():
    """Calculate processing statistics from history"""
    history = get_analysis_history()
    
    if not history:
        return None
    
    stats = {
        'total_analyses': len(history),
        'cancer_detections': sum(1 for h in history if h['prediction_label'] == 'Cancer'),
        'normal_predictions': sum(1 for h in history if h['prediction_label'] == 'Normal'),
        'avg_confidence': np.mean([h['confidence'] for h in history]),
        'models_used': list(set(h['model_type'] for h in history))
    }
    
    return stats
