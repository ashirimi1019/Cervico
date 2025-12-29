import os
import glob
import random
from pathlib import Path
import sys

# Add the parent directory to the Python path to fix imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from ai.src.predict import predict_dilation

def test_model_on_val_images():
    # Get validation images from the dataset
    data_dir = os.path.join('data', 'dataset')  # Updated path
    image_files = []
    
    # Search for images in all directories
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(os.path.join(data_dir, '**', ext), recursive=True))
    
    if not image_files:
        print("No images found!")
        print(f"Searched in: {os.path.abspath(data_dir)}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Select 3 random images to test
    test_images = random.sample(image_files, min(3, len(image_files)))
    
    print("\nTesting AI Model on Images")
    print("=========================")
    
    model_path = os.path.join('models', 'best_model.pt')  # Using best_model.pt
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {os.path.abspath(model_path)}")
        return
        
    print(f"Using model: {model_path}")
    
    for image_path in test_images:
        print(f"\nTesting image: {Path(image_path).name}")
        try:
            # Get predictions from the model
            class_pred, precise_pred = predict_dilation(
                image_path=image_path,
                model_path=model_path,
                device='cpu'  # Explicitly use CPU
            )
            
            print(f"Results:")
            print(f"- Classification: {class_pred:.1f} cm")
            print(f"- Precise Measurement: {precise_pred:.2f} cm")
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            print(f"Full error details:", e.__class__.__name__)
            import traceback
            print(traceback.format_exc())

if __name__ == "__main__":
    test_model_on_val_images()
