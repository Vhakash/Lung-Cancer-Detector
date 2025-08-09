import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
from skimage import exposure, filters

def get_available_enhancements():
    """Get list of available image enhancement techniques"""
    return [
        "Contrast Enhancement",
        "CLAHE",
        "Gamma Correction",
        "Noise Reduction"
    ]

def apply_enhancement(image, enhancement_type, strength=1.0):
    """Apply image enhancement technique"""
    # Convert to proper format for processing
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Ensure image is in the right format
    if image.dtype == np.float32 or image.dtype == np.float64:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    enhanced_image = image.copy()
    
    try:
        if enhancement_type == "Contrast Enhancement":
            enhanced_image = enhance_contrast(image, strength)
            
        elif enhancement_type == "CLAHE":
            enhanced_image = apply_clahe_enhancement(image, strength)
            
        elif enhancement_type == "Gamma Correction":
            enhanced_image = apply_gamma_correction(image, strength)
            
        elif enhancement_type == "Noise Reduction":
            enhanced_image = apply_noise_reduction(image, strength)
    
    except Exception as e:
        print(f"Enhancement error: {e}")
        return image
    
    # Convert back to float32 for model input
    enhanced_image = enhanced_image.astype(np.float32) / 255.0
    
    return enhanced_image

def enhance_contrast(image, strength):
    """Enhance image contrast"""
    alpha = 1.0 + (strength - 1.0) * 0.5  # Scale strength
    beta = 0
    enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return enhanced

def apply_clahe_enhancement(image, strength):
    """Apply Contrast Limited Adaptive Histogram Equalization"""
    clip_limit = 2.0 * strength
    
    if len(image.shape) == 3:
        # For color images, apply CLAHE to each channel in LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        # For grayscale images
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
        enhanced = clahe.apply(image)
    
    return enhanced

def apply_gamma_correction(image, strength):
    """Apply gamma correction"""
    # Convert strength to gamma value (0.5 to 2.0)
    gamma = 0.5 + strength * 1.5
    
    # Build lookup table
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 
                     for i in np.arange(0, 256)]).astype("uint8")
    
    # Apply gamma correction
    enhanced = cv2.LUT(image, table)
    
    return enhanced

def apply_noise_reduction(image, strength):
    """Apply noise reduction using bilateral filter"""
    d = int(5 + strength * 5)  # Neighborhood diameter
    sigma_color = 75 * strength
    sigma_space = 75 * strength
    
    if len(image.shape) == 3:
        enhanced = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    else:
        enhanced = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    return enhanced

def adaptive_enhancement(image):
    """Apply adaptive enhancement based on image characteristics"""
    # Analyze image characteristics
    mean_intensity = np.mean(image)
    std_intensity = np.std(image)
    
    # Choose enhancement based on characteristics
    if mean_intensity < 50:
        # Dark image - apply gamma correction
        return apply_gamma_correction(image, 1.5)
    elif std_intensity < 30:
        # Low contrast - apply CLAHE
        return apply_clahe_enhancement(image, 1.2)
    else:
        # Normal image - apply mild contrast enhancement
        return enhance_contrast(image, 1.1)
