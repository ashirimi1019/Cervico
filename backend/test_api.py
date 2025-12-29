import requests
import os
import glob
from pathlib import Path

def test_ultrasound_endpoint():
    # Get validation images from the dataset
    val_dir = os.path.join('..', 'ai', 'data', 'val')
    image_files = []
    
    # Search for images in validation directory
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(os.path.join(val_dir, '**', ext), recursive=True))
    
    if not image_files:
        print("No validation images found!")
        return
        
    print(f"Found {len(image_files)} validation images")
    
    # Test first 3 images
    for image_path in image_files[:3]:
        print(f"\nTesting image: {Path(image_path).name}")
        
        # Prepare the file for upload
        with open(image_path, 'rb') as img:
            files = {'file': (Path(image_path).name, img, 'image/jpeg')}
            
            try:
                # Make the request to our API
                response = requests.post('http://localhost:8001/process-ultrasound', files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"Success!")
                    print(f"Class Prediction: {result['class_prediction']:.1f} cm")
                    print(f"Precise Prediction: {result['precise_prediction']:.2f} cm")
                    print(f"Image ID: {result['id']}")
                else:
                    print(f"Error: {response.status_code}")
                    print(response.text)
                    
            except Exception as e:
                print(f"Request failed: {str(e)}")

if __name__ == "__main__":
    test_ultrasound_endpoint()
