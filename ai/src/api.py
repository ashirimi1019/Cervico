from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
from PIL import Image
import io
from typing import Dict, Any

from .preprocessing import UltrasoundPreprocessor
from .model import DilationPredictor
from .synthetic_data import UltrasoundGenerator

app = FastAPI(title="Cervico AI API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
preprocessor = UltrasoundPreprocessor()
predictor = DilationPredictor()
generator = UltrasoundGenerator()

@app.post("/predict")
async def predict_dilation(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Predict cervical dilation from ultrasound image
    
    Args:
        file: Uploaded ultrasound image
        
    Returns:
        Dictionary containing prediction results
    """
    # Read and preprocess image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    image_array = np.array(image)
    
    # Preprocess image
    processed_image = preprocessor.preprocess_image(image_array)
    image_tensor = torch.from_numpy(processed_image).float()
    
    # Get prediction
    class_prediction, precise_measurement = predictor.predict(image_tensor)
    
    return {
        "dilation_class": class_prediction,
        "dilation_cm": round(precise_measurement, 1),
        "confidence": "high" if abs(class_prediction - precise_measurement) < 0.5 else "medium"
    }

@app.post("/generate")
async def generate_synthetic(dilation_cm: float) -> Dict[str, Any]:
    """
    Generate synthetic ultrasound image for specified dilation
    
    Args:
        dilation_cm: Target dilation in centimeters
        
    Returns:
        Dictionary containing generated image data
    """
    # Generate synthetic image
    images = generator.generate_dilation_image(dilation_cm, num_images=1)
    
    # Convert to bytes
    img_byte_arr = io.BytesIO()
    images[0].save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    return {
        "image": img_byte_arr,
        "dilation_cm": dilation_cm
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
