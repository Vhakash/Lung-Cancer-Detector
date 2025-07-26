import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
from skimage import exposure, filters

def get_available_enhancements():
    """Get list of available image enhancement techniques"""
    return [
        "Contrast Enhancement",
        "CLAHE",
        "Histogram Equalization", 
        "Gaussian Blur",
        "Sharpening",
        "Edge Enhancement",
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
            
        elif enhancement_type == "Histogram Equalization":
            enhanced_image = apply_histogram_equalization(image)
            
        elif enhancement_type == "Gaussian Blur":
            enhanced_image = apply_gaussian_blur(image, strength)
            
        elif enhancement_type == "Sharpening":
            enhanced_image = apply_sharpening(image, strength)
            
        elif enhancement_type == "Edge Enhancement":
            enhanced_image = apply_edge_enhancement(image, strength)
            
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

def apply_histogram_equalization(image):
    """Apply histogram equalization"""
    if len(image.shape) == 3:
        # For color images, equalize each channel
        enhanced = np.zeros_like(image)
        for i in range(3):
            enhanced[:,:,i] = cv2.equalizeHist(image[:,:,i])
    else:
        # For grayscale images
        enhanced = cv2.equalizeHist(image)
    
    return enhanced

def apply_gaussian_blur(image, strength):
    """Apply Gaussian blur"""
    kernel_size = int(3 + (strength - 1.0) * 4)  # Scale kernel size with strength
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure odd kernel size
    
    enhanced = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return enhanced

def apply_sharpening(image, strength):
    """Apply image sharpening"""
    # Create sharpening kernel
    kernel_strength = strength * 0.5
    kernel = np.array([[-1,-1,-1],
                      [-1, 9 + kernel_strength, -1],
                      [-1,-1,-1]])
    
    enhanced = cv2.filter2D(image, -1, kernel)
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    
    return enhanced

def apply_edge_enhancement(image, strength):
    """Apply edge enhancement"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Create edge mask
    edges_colored = np.zeros_like(image)
    if len(image.shape) == 3:
        for i in range(3):
            edges_colored[:,:,i] = edges
    else:
        edges_colored = edges
    
    # Blend with original image
    alpha = strength * 0.3
    enhanced = cv2.addWeighted(image, 1.0, edges_colored, alpha, 0)
    
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
