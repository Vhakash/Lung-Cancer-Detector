import numpy as np
from PIL import Image
import cv2

def ensure_color_channels(image_array):
    """Ensure image has 3 color channels (RGB)"""
    if len(image_array.shape) == 2:
        # Grayscale image - convert to RGB
        image_array = np.stack([image_array] * 3, axis=-1)
    elif len(image_array.shape) == 3:
        if image_array.shape[2] == 1:
            # Single channel image - convert to RGB
            image_array = np.repeat(image_array, 3, axis=2)
        elif image_array.shape[2] == 4:
            # RGBA image - remove alpha channel
            image_array = image_array[:, :, :3]
    
    return image_array

def preprocess_image(image):
    """Preprocess image for model input"""
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Ensure proper data type
    if image.dtype != np.uint8:
        # Normalize to 0-255 range if needed
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # Ensure RGB channels
    image = ensure_color_channels(image)
    
    # Resize to model input size (224x224)
    if image.shape[:2] != (224, 224):
        image = cv2.resize(image, (224, 224))
    
    # Normalize pixel values to 0-1 range
    image = image.astype(np.float32) / 255.0
    
    return image

def normalize_dicom_image(pixel_array):
    """Normalize DICOM pixel array for display and processing"""
    # Handle different DICOM data types
    if pixel_array.dtype == np.uint16:
        # Convert to float and normalize
        pixel_array = pixel_array.astype(np.float32)
        
    # Apply windowing for better contrast
    # Use lung window settings by default
    window_center = 40
    window_width = 400
    
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    
    # Apply windowing
    pixel_array = np.clip(pixel_array, img_min, img_max)
    
    # Normalize to 0-255
    pixel_array = ((pixel_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    
    return pixel_array

def enhance_contrast(image, alpha=1.5, beta=0):
    """Enhance image contrast"""
    # Apply contrast enhancement
    enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return enhanced

def apply_clahe(image):
    """Apply Contrast Limited Adaptive Histogram Equalization"""
    if len(image.shape) == 3:
        # For color images, apply CLAHE to each channel
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        # For grayscale images
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(image)
    
    return enhanced
