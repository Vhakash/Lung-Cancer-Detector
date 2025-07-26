import numpy as np
from PIL import Image, ImageDraw
import random

def create_sample_lung_image(image_type="normal", size=(224, 224)):
    """Create a synthetic lung image for demonstration purposes"""
    # Create base image
    img = Image.new('L', size, color=20)  # Dark background
    draw = ImageDraw.Draw(img)
    
    # Draw lung-like shapes
    if image_type == "normal":
        # Draw normal lung structures
        # Left lung
        draw.ellipse([20, 40, 100, 180], fill=60, outline=80)
        # Right lung  
        draw.ellipse([124, 40, 204, 180], fill=60, outline=80)
        
        # Add some normal lung texture
        for _ in range(50):
            x = random.randint(25, 95)
            y = random.randint(45, 175)
            draw.point((x, y), fill=random.randint(70, 90))
            
        for _ in range(50):
            x = random.randint(129, 199)
            y = random.randint(45, 175)
            draw.point((x, y), fill=random.randint(70, 90))
            
    elif image_type == "suspicious":
        # Draw lungs with suspicious areas
        # Left lung
        draw.ellipse([20, 40, 100, 180], fill=60, outline=80)
        # Right lung
        draw.ellipse([124, 40, 204, 180], fill=60, outline=80)
        
        # Add suspicious nodules
        draw.ellipse([65, 80, 85, 100], fill=120, outline=140)  # Nodule in left lung
        draw.ellipse([150, 120, 165, 135], fill=115, outline=135)  # Nodule in right lung
        
        # Add normal texture
        for _ in range(40):
            x = random.randint(25, 95)
            y = random.randint(45, 175)
            if not (65 <= x <= 85 and 80 <= y <= 100):  # Avoid nodule area
                draw.point((x, y), fill=random.randint(70, 90))
                
        for _ in range(40):
            x = random.randint(129, 199)
            y = random.randint(45, 175)
            if not (150 <= x <= 165 and 120 <= y <= 135):  # Avoid nodule area
                draw.point((x, y), fill=random.randint(70, 90))
    
    # Convert to RGB
    img_rgb = Image.new('RGB', size)
    img_rgb.paste(img)
    
    return img_rgb

def create_sample_ct_scan(image_type="normal", size=(224, 224)):
    """Create a synthetic CT scan image"""
    # Create base CT-like image
    img = Image.new('L', size, color=30)
    draw = ImageDraw.Draw(img)
    
    # Draw body outline
    draw.ellipse([10, 10, 214, 214], fill=50, outline=70)
    
    # Draw lung areas
    # Left lung
    draw.ellipse([40, 60, 100, 140], fill=25, outline=40)
    # Right lung
    draw.ellipse([124, 60, 184, 140], fill=25, outline=40)
    
    # Draw ribs
    for i in range(5):
        y = 50 + i * 25
        draw.arc([20, y, 204, y+20], start=0, end=180, fill=80, width=2)
    
    # Draw spine
    draw.rectangle([108, 140, 116, 200], fill=100, outline=120)
    
    if image_type == "suspicious":
        # Add tumor-like mass
        draw.ellipse([70, 85, 90, 105], fill=80, outline=100)
        # Add some infiltration
        draw.ellipse([140, 95, 155, 110], fill=70, outline=90)
    
    # Convert to RGB
    img_rgb = Image.new('RGB', size)
    img_rgb.paste(img)
    
    return img_rgb

def get_sample_image_names():
    """Get list of available sample image names"""
    return [
        "Normal Chest X-ray",
        "Suspicious Chest X-ray", 
        "Normal CT Scan",
        "Suspicious CT Scan",
        "Clear Lung Fields",
        "Nodular Pattern"
    ]

def get_sample_image(image_name):
    """Get a sample image by name"""
    image_mapping = {
        "Normal Chest X-ray": lambda: create_sample_lung_image("normal"),
        "Suspicious Chest X-ray": lambda: create_sample_lung_image("suspicious"),
        "Normal CT Scan": lambda: create_sample_ct_scan("normal"),
        "Suspicious CT Scan": lambda: create_sample_ct_scan("suspicious"),
        "Clear Lung Fields": lambda: create_sample_lung_image("normal"),
        "Nodular Pattern": lambda: create_sample_lung_image("suspicious")
    }
    
    if image_name in image_mapping:
        return image_mapping[image_name]()
    else:
        return None

def get_sample_image_description(image_name):
    """Get description for sample images"""
    descriptions = {
        "Normal Chest X-ray": "A typical normal chest X-ray showing clear lung fields with no abnormalities.",
        "Suspicious Chest X-ray": "Chest X-ray with suspicious nodular opacities that may require further investigation.",
        "Normal CT Scan": "Normal chest CT scan showing typical lung anatomy without concerning findings.",
        "Suspicious CT Scan": "CT scan showing areas of concern that may indicate pathological changes.",
        "Clear Lung Fields": "X-ray demonstrating completely clear bilateral lung fields.",
        "Nodular Pattern": "Image showing nodular patterns that may be associated with various lung conditions."
    }
    
    return descriptions.get(image_name, "Sample medical image for testing the AI detection system.")
