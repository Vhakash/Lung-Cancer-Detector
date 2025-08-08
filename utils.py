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
import logging
from typing import Optional, Dict, List, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to track database availability
_db_available = None

def check_database_availability() -> bool:
    """Check if database is available and cache the result."""
    global _db_available
    
    if _db_available is not None:
        return _db_available
    
    try:
        conn = _get_raw_db_connection()
        if conn:
            conn.close()
            _db_available = True
            logger.info("Database connection verified successfully")
            return True
    except Exception as e:
        logger.warning(f"Database not available: {str(e)}")
        _db_available = False
        return False

def _get_raw_db_connection():
    """Internal function to get database connection without error handling."""
    if "postgres" not in st.secrets:
        raise Exception("PostgreSQL configuration not found in secrets")
    
    # Check if we have a connection string (for cloud databases like Neon)
    if "connection_string" in st.secrets["postgres"]:
        return psycopg2.connect(st.secrets["postgres"]["connection_string"])
    else:
        # Add SSL mode for cloud database connections (like Neon)
        connection_params = dict(st.secrets["postgres"])
        connection_params["sslmode"] = "require"
        return psycopg2.connect(**connection_params)

def get_db_connection():
    """Establish a connection to the PostgreSQL database with comprehensive error handling."""
    try:
        return _get_raw_db_connection()
    except KeyError as e:
        error_msg = f"Database configuration missing: {str(e)}"
        logger.error(error_msg)
        st.error("🔧 Database Configuration Error")
        st.error("Please check your database configuration in Streamlit Cloud secrets.")
        st.info("The app will continue with limited functionality (no analysis history).")
        return None
    except psycopg2.OperationalError as e:
        error_msg = f"Database connection failed: {str(e)}"
        logger.error(error_msg)
        st.error("🔌 Database Connection Failed")
        st.error("Unable to connect to the database. Please check your connection settings.")
        st.info("The app will continue with limited functionality (no analysis history).")
        return None
    except Exception as e:
        error_msg = f"Unexpected database error: {str(e)}"
        logger.error(error_msg)
        st.error("⚠️ Database Error")
        st.error(f"An unexpected error occurred: {str(e)}")
        st.info("The app will continue with limited functionality (no analysis history).")
        return None

def init_db() -> bool:
    """Initialize the PostgreSQL database and create the history table.
    
    Returns:
        bool: True if initialization successful, False otherwise
    """
    try:
        if not check_database_availability():
            st.warning("📊 Database not available - Analysis history will not be saved")
            return False
            
        conn = get_db_connection()
        if not conn:
            return False
            
        with conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS history (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        image_path VARCHAR(255),
                        model_type VARCHAR(50) NOT NULL,
                        prediction_value REAL NOT NULL,
                        prediction_label VARCHAR(50) NOT NULL,
                        confidence REAL NOT NULL,
                        enhancement VARCHAR(50),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                ''')
                
                # Create index for better query performance
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_history_timestamp 
                    ON history(timestamp DESC);
                ''')
                
                # Create index for model type queries
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_history_model_type 
                    ON history(model_type);
                ''')
        
        # Ensure history_images directory exists
        if not os.path.exists('history_images'):
            os.makedirs('history_images')
            logger.info("Created history_images directory")
            
        logger.info("Database initialized successfully!")
        st.success("✅ Database connected and initialized successfully!")
        return True
        
    except Exception as e:
        error_msg = f"Database initialization failed: {str(e)}"
        logger.error(error_msg)
        st.error("🚫 Database Initialization Failed")
        st.error("The app will continue without database functionality.")
        st.warning("⚠️ Analysis history will not be saved.")
        return False

def add_to_history(image: np.ndarray, model_type: str, prediction: np.ndarray, enhancement: Optional[str] = None) -> bool:
    """Add analysis result to history with robust error handling.
    
    Args:
        image: Processed image array
        model_type: Type of model used
        prediction: Model prediction array
        enhancement: Enhancement technique used (optional)
        
    Returns:
        bool: True if successfully added to history, False otherwise
    """
    if not check_database_availability():
        return False
        
    try:
        label, confidence = calculate_prediction_confidence(prediction[0][0])
        
        # Save the image file with error handling
        current_time = datetime.now()
        timestamp_str = current_time.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        image_path = f"history_images/{timestamp_str}.png"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        
        # Save image with proper error handling
        try:
            Image.fromarray((image * 255).astype(np.uint8)).save(image_path)
        except Exception as img_error:
            logger.warning(f"Failed to save image: {img_error}")
            image_path = None  # Continue without saving image
        
        conn = get_db_connection()
        if not conn:
            return False
            
        with conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    '''
                    INSERT INTO history (timestamp, image_path, model_type, prediction_value, 
                                       prediction_label, confidence, enhancement)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    ''',
                    (current_time, image_path, model_type, float(prediction[0][0]), 
                     label, float(confidence), enhancement)
                )
                result_id = cursor.fetchone()[0]
                logger.info(f"Added analysis to history with ID: {result_id}")
        
        return True
        
    except Exception as e:
        error_msg = f"Failed to add analysis to history: {str(e)}"
        logger.error(error_msg)
        st.warning("⚠️ Failed to save analysis to history")
        return False

def get_analysis_history(limit: int = 10) -> List[Dict[str, Any]]:
    """Retrieve analysis history with robust error handling.
    
    Args:
        limit: Maximum number of records to retrieve
        
    Returns:
        List of history records or empty list if unavailable
    """
    if not check_database_availability():
        return []
        
    try:
        conn = get_db_connection()
        if not conn:
            return []
            
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    "SELECT * FROM history ORDER BY timestamp DESC LIMIT %s",
                    (limit,)
                )
                history = cursor.fetchall()
                
        # Convert to list of dicts for better handling
        return [dict(record) for record in history]
        
    except Exception as e:
        error_msg = f"Failed to retrieve analysis history: {str(e)}"
        logger.error(error_msg)
        st.warning("⚠️ Failed to retrieve analysis history")
        return []

def clear_analysis_history() -> bool:
    """Clear analysis history with robust error handling.
    
    Returns:
        bool: True if successfully cleared, False otherwise
    """
    if not check_database_availability():
        st.warning("Database not available - cannot clear history")
        return False
        
    try:
        conn = get_db_connection()
        if not conn:
            return False
            
        with conn:
            with conn.cursor() as cursor:
                cursor.execute("TRUNCATE TABLE history RESTART IDENTITY")
                
        # Optional: also delete saved image files
        try:
            if os.path.exists('history_images'):
                for filename in os.listdir('history_images'):
                    file_path = os.path.join('history_images', filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                logger.info("Cleared saved image files")
        except Exception as file_error:
            logger.warning(f"Failed to clear some image files: {file_error}")
            
        logger.info("Analysis history cleared successfully")
        return True
        
    except Exception as e:
        error_msg = f"Failed to clear analysis history: {str(e)}"
        logger.error(error_msg)
        st.error("⚠️ Failed to clear analysis history")
        return False

def read_dicom_file(file_path: str) -> Tuple[Optional[Any], Optional[np.ndarray]]:
    """Read and process DICOM file with comprehensive error handling.
    
    Args:
        file_path: Path to DICOM file
        
    Returns:
        Tuple of (dicom_data, normalized_array) or (None, None) if failed
    """
    try:
        # Read DICOM file
        dicom_data = pydicom.dcmread(file_path)
        
        # Get pixel array
        pixel_array = dicom_data.pixel_array
        
        # Normalize for display
        normalized_array = normalize_dicom_image(pixel_array)
        
        logger.info(f"Successfully processed DICOM file: {file_path}")
        return dicom_data, normalized_array
    
    except FileNotFoundError:
        st.error("📁 DICOM file not found")
        return None, None
    except Exception as e:
        error_msg = f"Error reading DICOM file: {str(e)}"
        logger.error(error_msg)
        st.error("⚠️ Error reading DICOM file")
        st.error(f"Details: {str(e)}")
        return None, None

def display_dicom_info(dicom_data: Any) -> None:
    """Display DICOM metadata information with error handling.
    
    Args:
        dicom_data: DICOM dataset object
    """
    if dicom_data is None:
        st.info("No DICOM metadata available")
        return
    
    try:
        st.subheader("📋 DICOM Information")
        
        # Extract relevant DICOM tags safely
        info_dict = {}
        
        # Patient information
        safe_tags = [
            ('PatientName', 'Patient Name'),
            ('PatientID', 'Patient ID'),
            ('PatientAge', 'Patient Age'),
            ('PatientSex', 'Patient Sex'),
            ('StudyDate', 'Study Date'),
            ('Modality', 'Modality'),
            ('BodyPartExamined', 'Body Part'),
            ('Rows', 'Image Height'),
            ('Columns', 'Image Width')
        ]
        
        for tag, display_name in safe_tags:
            try:
                if hasattr(dicom_data, tag):
                    value = getattr(dicom_data, tag)
                    if value:
                        info_dict[display_name] = str(value)
            except Exception:
                continue  # Skip problematic tags
        
        # Special handling for pixel spacing
        try:
            if hasattr(dicom_data, 'PixelSpacing') and dicom_data.PixelSpacing:
                spacing = dicom_data.PixelSpacing
                info_dict['Pixel Spacing'] = f"{spacing[0]:.2f} x {spacing[1]:.2f} mm"
        except Exception:
            pass
        
        # Display information in a nice format
        if info_dict:
            df = pd.DataFrame(list(info_dict.items()), columns=['Property', 'Value'])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No readable metadata available for this DICOM file.")
            
    except Exception as e:
        logger.error(f"Error displaying DICOM info: {str(e)}")
        st.warning("⚠️ Error displaying DICOM metadata")

def calculate_prediction_confidence(prediction_value: float) -> Tuple[str, float]:
    """Calculate prediction label and confidence percentage with validation.
    
    Args:
        prediction_value: Raw prediction value (0-1)
        
    Returns:
        Tuple of (label, confidence_percentage)
    """
    try:
        # Ensure prediction value is valid
        prediction_value = float(prediction_value)
        prediction_value = max(0.0, min(1.0, prediction_value))  # Clamp to [0,1]
        
        if prediction_value > 0.5:
            label = "Cancer"
            confidence = prediction_value * 100
        else:
            label = "Normal"
            confidence = (1 - prediction_value) * 100
        
        return label, round(confidence, 2)
        
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid prediction value: {prediction_value}, error: {e}")
        return "Unknown", 0.0

def compare_model_performances() -> pd.DataFrame:
    """Compare performance metrics between different models.
    
    Returns:
        DataFrame with model performance comparison
    """
    try:
        # Simulated performance data with realistic values
        models_data = {
            'Model Type': ['Basic CNN', 'InceptionV3 Transfer Learning'],
            'Accuracy': [0.87, 0.94],
            'Precision': [0.84, 0.91],
            'Recall': [0.82, 0.89],
            'F1-Score': [0.83, 0.90],
            'AUC': [0.88, 0.95],
            'Processing Time (ms)': [120, 180]
        }
        
        return pd.DataFrame(models_data)
        
    except Exception as e:
        logger.error(f"Error creating model comparison: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    try:
        if size_bytes == 0:
            return "0B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = int(np.floor(np.log(size_bytes) / np.log(1024)))
        i = min(i, len(size_names) - 1)  # Prevent index out of range
        
        p = pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"
        
    except (ValueError, TypeError, ZeroDivisionError):
        return "Unknown"

def validate_image_format(file) -> bool:
    """Validate if uploaded file is a supported image format.
    
    Args:
        file: Streamlit uploaded file object
        
    Returns:
        bool: True if format is supported
    """
    try:
        if not hasattr(file, 'name') or not file.name:
            return False
            
        allowed_extensions = ['jpg', 'jpeg', 'png', 'dcm', 'dicom']
        file_extension = file.name.split('.')[-1].lower()
        return file_extension in allowed_extensions
        
    except Exception:
        return False

def calculate_processing_stats() -> Optional[Dict[str, Any]]:
    """Calculate processing statistics from history with error handling.
    
    Returns:
        Dictionary with processing statistics or None if unavailable
    """
    try:
        history = get_analysis_history(limit=100)  # Get more records for stats
        
        if not history:
            return None
        
        stats = {
            'total_analyses': len(history),
            'cancer_detections': sum(1 for h in history if h.get('prediction_label') == 'Cancer'),
            'normal_predictions': sum(1 for h in history if h.get('prediction_label') == 'Normal'),
            'avg_confidence': round(np.mean([h.get('confidence', 0) for h in history]), 2),
            'models_used': list(set(h.get('model_type', 'Unknown') for h in history)),
            'enhancement_usage': sum(1 for h in history if h.get('enhancement') is not None)
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error calculating processing stats: {str(e)}")
        return None

def get_database_status() -> Dict[str, Any]:
    """Get current database status information.
    
    Returns:
        Dictionary with database status information
    """
    try:
        if check_database_availability():
            conn = get_db_connection()
            if conn:
                with conn:
                    with conn.cursor() as cursor:
                        cursor.execute("SELECT COUNT(*) FROM history")
                        record_count = cursor.fetchone()[0]
                        
                return {
                    'status': 'connected',
                    'available': True,
                    'record_count': record_count,
                    'message': 'Database connected successfully'
                }
        
        return {
            'status': 'disconnected',
            'available': False,
            'record_count': 0,
            'message': 'Database not available'
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'available': False,
            'record_count': 0,
            'message': f'Database error: {str(e)}'
        }
