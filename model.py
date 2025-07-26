import numpy as np
import time

class MockModel:
    """Mock model for demonstration purposes"""
    def __init__(self, model_type="basic"):
        self.model_type = model_type
        
    def predict(self, image):
        """Simulate prediction with random but realistic values"""
        # Simulate processing time
        time.sleep(0.1 if self.model_type == "basic" else 0.2)
        
        # Generate realistic prediction values
        # For cancer detection: 0 = Normal, 1 = Cancer
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