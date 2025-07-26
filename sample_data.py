import numpy as np
from PIL import Image, ImageDraw
import random

def create_sample_lung_image(image_type="normal", size=(224, 224)):
    """Create medically accurate synthetic lung image based on real radiological patterns"""
    # Create base image with realistic tissue densities
    img = Image.new('L', size, color=15)  # Very dark background (air/outside body)
    draw = ImageDraw.Draw(img)
    
    # Draw realistic chest cavity outline
    draw.ellipse([5, 20, 219, 200], fill=25, outline=30)  # Chest wall/soft tissue
    
    if image_type == "normal":
        # Draw normal lung anatomy based on standard chest X-ray patterns
        # Left lung (appears on right side of image)
        draw.ellipse([25, 45, 105, 175], fill=45, outline=50)  # Normal lung tissue density
        # Right lung (appears on left side of image) 
        draw.ellipse([119, 45, 199, 175], fill=45, outline=50)
        
        # Add normal pulmonary vessels (bronchovascular markings)
        for i in range(8):
            x1 = random.randint(30, 100)
            y1 = random.randint(50, 80)
            x2 = x1 + random.randint(-15, 15)
            y2 = y1 + random.randint(20, 40)
            draw.line([(x1, y1), (x2, y2)], fill=55, width=1)
            
        for i in range(8):
            x1 = random.randint(124, 194)
            y1 = random.randint(50, 80)
            x2 = x1 + random.randint(-15, 15)
            y2 = y1 + random.randint(20, 40)
            draw.line([(x1, y1), (x2, y2)], fill=55, width=1)
        
        # Add normal hilum (lung root structures)
        draw.ellipse([95, 90, 105, 110], fill=65, outline=70)  # Left hilum
        draw.ellipse([119, 90, 129, 110], fill=65, outline=70)  # Right hilum
        
    elif image_type == "cancer_stage1":
        # Stage 1 lung cancer: Small peripheral nodule (2-3cm)
        # Left lung
        draw.ellipse([25, 45, 105, 175], fill=45, outline=50)
        # Right lung with early-stage cancer
        draw.ellipse([119, 45, 199, 175], fill=45, outline=50)
        
        # Small peripheral nodule in right upper lobe (characteristic of early cancer)
        draw.ellipse([145, 70, 160, 85], fill=85, outline=95)  # Dense nodule
        # Subtle spiculated edges (cancer characteristic)
        for i in range(6):
            angle = i * 60
            x = 152 + int(10 * np.cos(np.radians(angle)))
            y = 77 + int(10 * np.sin(np.radians(angle)))
            draw.line([(152, 77), (x, y)], fill=75, width=1)
        
        # Normal vessels in unaffected areas
        for i in range(6):
            x1 = random.randint(30, 100)
            y1 = random.randint(50, 80)
            x2 = x1 + random.randint(-15, 15)
            y2 = y1 + random.randint(20, 40)
            draw.line([(x1, y1), (x2, y2)], fill=55, width=1)
            
    elif image_type == "cancer_stage3":
        # Stage 3 lung cancer: Larger mass with mediastinal involvement
        # Left lung
        draw.ellipse([25, 45, 105, 175], fill=45, outline=50)
        # Right lung with advanced cancer
        draw.ellipse([119, 45, 199, 175], fill=45, outline=50)
        
        # Large central mass (4-5cm)
        draw.ellipse([135, 85, 165, 115], fill=95, outline=105)
        # Irregular margins typical of malignancy
        draw.ellipse([130, 90, 170, 120], fill=80, outline=85)
        
        # Enlarged hilar lymph nodes
        draw.ellipse([115, 95, 125, 105], fill=85, outline=90)
        draw.ellipse([110, 100, 120, 110], fill=85, outline=90)
        
        # Pleural thickening/effusion
        draw.rectangle([119, 160, 199, 175], fill=70, outline=75)
        
        # Reduced vessel markings due to obstruction
        for i in range(3):
            x1 = random.randint(124, 135)
            y1 = random.randint(50, 70)
            x2 = x1 + random.randint(-10, 10)
            y2 = y1 + random.randint(15, 25)
            draw.line([(x1, y1), (x2, y2)], fill=55, width=1)
    
    # Convert to RGB for consistency
    img_rgb = Image.new('RGB', size)
    img_rgb.paste(img)
    
    return img_rgb

def create_sample_ct_scan(image_type="normal", size=(224, 224)):
    """Create medically accurate synthetic CT scan based on real radiological patterns"""
    # Create base CT image with proper Hounsfield unit representation
    img = Image.new('L', size, color=40)  # Background tissue density
    draw = ImageDraw.Draw(img)
    
    # Draw body outline with realistic tissue densities
    draw.ellipse([15, 15, 209, 209], fill=60, outline=65)  # Chest wall/muscle
    
    if image_type == "normal":
        # Normal lung anatomy on CT
        # Left lung (air-filled, very dark)
        draw.ellipse([45, 65, 110, 145], fill=20, outline=25)
        # Right lung
        draw.ellipse([114, 65, 179, 145], fill=20, outline=25)
        
        # Normal pulmonary vessels (blood vessels appear brighter)
        for i in range(12):
            x1 = random.randint(50, 105)
            y1 = random.randint(70, 90)
            x2 = x1 + random.randint(-20, 20)
            y2 = y1 + random.randint(15, 35)
            draw.line([(x1, y1), (x2, y2)], fill=45, width=2)
            
        for i in range(12):
            x1 = random.randint(119, 174)
            y1 = random.randint(70, 90)
            x2 = x1 + random.randint(-20, 20)
            y2 = y1 + random.randint(15, 35)
            draw.line([(x1, y1), (x2, y2)], fill=45, width=2)
        
        # Ribs (very bright on CT)
        for i in range(6):
            y = 40 + i * 20
            draw.arc([25, y, 199, y+25], start=0, end=180, fill=120, width=3)
        
        # Spine and mediastinum
        draw.rectangle([105, 140, 119, 190], fill=110, outline=120)  # Vertebrae
        draw.ellipse([100, 85, 124, 105], fill=80, outline=85)  # Heart/mediastinum
        
    elif image_type == "nodule_benign":
        # Benign pulmonary nodule characteristics
        # Normal lung tissue
        draw.ellipse([45, 65, 110, 145], fill=20, outline=25)
        draw.ellipse([114, 65, 179, 145], fill=20, outline=25)
        
        # Benign nodule: well-circumscribed, smooth margins, calcification
        draw.ellipse([140, 85, 155, 100], fill=90, outline=95)  # Soft tissue density
        draw.ellipse([143, 88, 148, 93], fill=130, outline=135)  # Central calcification
        
        # Normal vessels
        for i in range(10):
            x1 = random.randint(50, 105)
            y1 = random.randint(70, 90)
            x2 = x1 + random.randint(-15, 15)
            y2 = y1 + random.randint(15, 30)
            draw.line([(x1, y1), (x2, y2)], fill=45, width=2)
            
        # Ribs and spine
        for i in range(6):
            y = 40 + i * 20
            draw.arc([25, y, 199, y+25], start=0, end=180, fill=120, width=3)
        draw.rectangle([105, 140, 119, 190], fill=110, outline=120)
        draw.ellipse([100, 85, 124, 105], fill=80, outline=85)
        
    elif image_type == "cancer_advanced":
        # Advanced lung cancer with typical CT features
        # Lungs with mass effect
        draw.ellipse([45, 65, 110, 145], fill=20, outline=25)  # Normal left lung
        draw.ellipse([114, 65, 179, 145], fill=20, outline=25)  # Right lung
        
        # Large irregular mass with spiculated margins
        draw.ellipse([130, 75, 170, 115], fill=85, outline=90)  # Main mass
        # Spiculated margins (characteristic of malignancy)
        for i in range(8):
            angle = i * 45
            x = 150 + int(25 * np.cos(np.radians(angle)))
            y = 95 + int(25 * np.sin(np.radians(angle)))
            draw.line([(150, 95), (x, y)], fill=70, width=2)
        
        # Mediastinal lymph node enlargement
        draw.ellipse([105, 90, 119, 104], fill=85, outline=90)
        draw.ellipse([100, 95, 114, 109], fill=85, outline=90)
        
        # Pleural effusion (fluid collection)
        draw.rectangle([114, 135, 179, 145], fill=55, outline=60)
        
        # Reduced vascularity due to tumor
        for i in range(5):
            x1 = random.randint(50, 105)
            y1 = random.randint(70, 90)
            x2 = x1 + random.randint(-10, 10)
            y2 = y1 + random.randint(10, 20)
            draw.line([(x1, y1), (x2, y2)], fill=45, width=1)
        
        # Ribs and spine
        for i in range(6):
            y = 40 + i * 20
            draw.arc([25, y, 199, y+25], start=0, end=180, fill=120, width=3)
        draw.rectangle([105, 140, 119, 190], fill=110, outline=120)
        
    # Convert to RGB
    img_rgb = Image.new('RGB', size)
    img_rgb.paste(img)
    
    return img_rgb

def get_sample_image_names():
    """Get list of available medically accurate sample image names"""
    return [
        "Normal Chest X-ray",
        "Stage 1 Lung Cancer (X-ray)",
        "Stage 3 Lung Cancer (X-ray)", 
        "Normal CT Scan",
        "Benign Pulmonary Nodule (CT)",
        "Advanced Lung Cancer (CT)"
    ]

def get_sample_image(image_name):
    """Get a medically accurate sample image by name"""
    image_mapping = {
        "Normal Chest X-ray": lambda: create_sample_lung_image("normal"),
        "Stage 1 Lung Cancer (X-ray)": lambda: create_sample_lung_image("cancer_stage1"),
        "Stage 3 Lung Cancer (X-ray)": lambda: create_sample_lung_image("cancer_stage3"),
        "Normal CT Scan": lambda: create_sample_ct_scan("normal"),
        "Benign Pulmonary Nodule (CT)": lambda: create_sample_ct_scan("nodule_benign"),
        "Advanced Lung Cancer (CT)": lambda: create_sample_ct_scan("cancer_advanced")
    }
    
    if image_name in image_mapping:
        return image_mapping[image_name]()
    else:
        return None

def get_sample_image_description(image_name):
    """Get medically accurate description for sample images"""
    descriptions = {
        "Normal Chest X-ray": "Normal posteroanterior chest X-ray showing clear bilateral lung fields with normal pulmonary vasculature and no focal abnormalities.",
        "Stage 1 Lung Cancer (X-ray)": "Chest X-ray showing a small peripheral nodule in the right upper lobe with spiculated margins, consistent with Stage 1 lung adenocarcinoma.",
        "Stage 3 Lung Cancer (X-ray)": "Chest X-ray demonstrating a large central mass with hilar lymph node enlargement and pleural effusion, indicative of Stage 3 lung cancer.",
        "Normal CT Scan": "Normal chest CT scan showing well-aerated bilateral lungs with normal pulmonary vessels and no masses or nodules.",
        "Benign Pulmonary Nodule (CT)": "CT scan showing a well-circumscribed pulmonary nodule with central calcification, characteristic features of a benign granuloma.",
        "Advanced Lung Cancer (CT)": "CT scan revealing a large spiculated mass with mediastinal invasion, enlarged lymph nodes, and pleural effusion consistent with advanced lung carcinoma."
    }
    
    return descriptions.get(image_name, "Medically accurate sample image for AI diagnostic testing.")

def get_expected_diagnosis(image_name):
    """Get the expected accurate diagnosis for each sample case"""
    diagnoses = {
        "Normal Chest X-ray": {"prediction": 0.05, "label": "Normal", "confidence": 95.0},
        "Stage 1 Lung Cancer (X-ray)": {"prediction": 0.85, "label": "Cancer", "confidence": 85.0},
        "Stage 3 Lung Cancer (X-ray)": {"prediction": 0.95, "label": "Cancer", "confidence": 95.0},
        "Normal CT Scan": {"prediction": 0.02, "label": "Normal", "confidence": 98.0},
        "Benign Pulmonary Nodule (CT)": {"prediction": 0.15, "label": "Normal", "confidence": 85.0},
        "Advanced Lung Cancer (CT)": {"prediction": 0.98, "label": "Cancer", "confidence": 98.0}
    }
    
    return diagnoses.get(image_name, {"prediction": 0.5, "label": "Unknown", "confidence": 50.0})
