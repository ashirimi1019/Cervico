import os
import torch
import logging
from PIL import Image
import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt

from .model import DilationPredictor
from .preprocessing import UltrasoundPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def predict_dilation(
    image_path: str,
    model_path: str,
    save_dir: Optional[str] = None,
    device: str = None
) -> Tuple[float, float]:
    """
    Predict cervical dilation from an ultrasound image
    
    Args:
        image_path: Path to the ultrasound image
        model_path: Path to the trained model checkpoint
        save_dir: Optional directory to save visualization
        device: Device to run inference on
        
    Returns:
        Tuple of (class prediction, precise measurement)
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    
    preprocessor = UltrasoundPreprocessor(training=False)
    processed_image = preprocessor.preprocess_image(image)
    
    # Convert to torch tensor
    image_tensor = torch.tensor(processed_image).float()
    if image_tensor.shape[-1] == 3:  # If channels are last
        image_tensor = image_tensor.permute(2, 0, 1)  # Move channels to first dimension
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    
    # Load model and predict
    predictor = DilationPredictor(model_path=model_path, device=device)
    class_pred, precise_pred = predictor.predict(image_tensor)
    
    logger.info(f"Predicted dilation class: {class_pred}cm")
    logger.info(f"Precise measurement: {precise_pred:.2f}cm")
    
    # Visualize if save_dir is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 5))
        
        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        # Processed image
        plt.subplot(1, 2, 2)
        plt.imshow(processed_image.transpose(2, 0, 1).transpose(1, 2, 0))
        plt.title(f'Processed Image\nPredicted: {precise_pred:.2f}cm')
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        save_path = os.path.join(save_dir, f'prediction_{os.path.basename(image_path)}')
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"Saved visualization to {save_path}")
    
    return class_pred, precise_pred

if __name__ == "__main__":
    # Example usage
    predict_dilation(
        image_path="/Users/harimanivannan/Documents/GitHub/Cervico/ai/data/images/3cm.jpg",
        model_path="/Users/harimanivannan/Documents/GitHub/Cervico/ai/models/best_model.pt",
        save_dir="/Users/harimanivannan/Documents/GitHub/Cervico/ai/predictions"
    )
