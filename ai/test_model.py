import os
import random
from PIL import Image
import matplotlib.pyplot as plt
from src.predict import predict_dilation
import torch

def test_random_image(val_dir: str, model_path: str, output_dir: str = "predictions"):
    """
    Test model on a random image from validation set
    
    Args:
        val_dir: Directory containing validation images
        model_path: Path to the model checkpoint
        output_dir: Directory to save visualization
    """
    # Get list of validation images
    val_images = [f for f in os.listdir(val_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not val_images:
        raise ValueError(f"No images found in {val_dir}")
    
    # Select a random image
    test_image = random.choice(val_images)
    image_path = os.path.join(val_dir, test_image)
    
    # Get true dilation from filename (assuming format: {dilation}cm_*.jpg)
    true_dilation = float(test_image.split('cm')[0])
    
    print(f"\nTesting image: {test_image}")
    print(f"True dilation: {true_dilation}cm")
    
    # Make prediction
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    class_pred, precise_pred = predict_dilation(
        image_path=image_path,
        model_path=model_path,
        save_dir=output_dir,
        device=device
    )
    
    print(f"Predicted class: {class_pred}cm")
    print(f"Precise prediction: {precise_pred:.2f}cm")
    print(f"Absolute error: {abs(true_dilation - precise_pred):.2f}cm")

if __name__ == "__main__":
    # Paths
    val_dir = "data/dataset/val"
    model_path = "models/best_model.pt"
    output_dir = "predictions"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Test the model
    test_random_image(val_dir, model_path, output_dir)
