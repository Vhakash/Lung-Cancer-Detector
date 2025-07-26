import numpy as np
import time

class MockModel:
    """Mock model for demonstration purposes with accurate sample predictions"""
    def __init__(self, model_type="basic"):
        self.model_type = model_type
        self.current_sample_case = None
        
    def set_sample_case(self, sample_case_name):
        """Set the current sample case for accurate prediction"""
        self.current_sample_case = sample_case_name
        
    def predict(self, image):
        """Simulate prediction with accurate values for sample cases"""
        # Simulate processing time
        time.sleep(0.1 if self.model_type == "basic" else 0.2)
        
        # If this is a sample case, return accurate prediction
        if self.current_sample_case:
            from sample_data import get_expected_diagnosis
            diagnosis = get_expected_diagnosis(self.current_sample_case)
            
            # Add small random variation to simulate model uncertainty
            base_prediction = diagnosis["prediction"]
            variation = np.random.uniform(-0.05, 0.05)
            prediction = np.clip(base_prediction + variation, 0.01, 0.99)
            
            # Reset sample case after prediction
            self.current_sample_case = None
            
            return np.array([[prediction]])
        
        # For uploaded images, generate realistic random predictions
        if self.model_type == "transfer":
            # Transfer learning model should be more accurate
            prediction = np.random.uniform(0.1, 0.9)
        else:
            # Basic model with slightly lower accuracy
            prediction = np.random.uniform(0.2, 0.8)
            
        return np.array([[prediction]])

def create_model():
    """Create a basic CNN model for lung cancer detection"""
    print("Creating Basic CNN model (mock version)")
    return MockModel("basic")

def load_pretrained_model():
    """Load InceptionV3 transfer learning model"""
    print("Loading InceptionV3 Transfer Learning model (mock version)")
    return MockModel("transfer")

def get_model_info(model):
    """Get information about the model"""
    if isinstance(model, MockModel):
        return {
            "type": "Mock Model",
            "parameters": "Simulated",
            "layers": "Simulated"
        }
    else:
        return {
            "type": "Unknown Model",
            "parameters": "Unknown",
            "layers": "Unknown"
        }