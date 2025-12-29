import os
import shutil
import random
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_dataset(
    augmented_dir: str,
    output_dir: str,
    train_split: float = 0.8,
    seed: int = 42
):
    """
    Organize augmented images into train/val splits
    
    Args:
        augmented_dir: Directory containing augmented images by dilation
        output_dir: Directory to save organized dataset
        train_split: Fraction of data to use for training
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Process each dilation directory
    total_train = 0
    total_val = 0
    
    for dilation_dir in os.listdir(augmented_dir):
        if not dilation_dir.endswith('cm'):
            continue
            
        src_dir = os.path.join(augmented_dir, dilation_dir)
        if not os.path.isdir(src_dir):
            continue
            
        # Get all augmented images
        images = [f for f in os.listdir(src_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)
        
        # Split into train/val
        split_idx = int(len(images) * train_split)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Copy images to respective directories
        for img in train_images:
            src = os.path.join(src_dir, img)
            dst = os.path.join(train_dir, f"{dilation_dir}_{img}")
            shutil.copy2(src, dst)
            total_train += 1
            
        for img in val_images:
            src = os.path.join(src_dir, img)
            dst = os.path.join(val_dir, f"{dilation_dir}_{img}")
            shutil.copy2(src, dst)
            total_val += 1
    
    logger.info(f"Dataset prepared with {total_train} training and {total_val} validation images")
    return train_dir, val_dir

if __name__ == "__main__":
    # Prepare the dataset
    augmented_dir = "/Users/harimanivannan/Documents/GitHub/Cervico/ai/data/augmented"
    output_dir = "/Users/harimanivannan/Documents/GitHub/Cervico/ai/data/dataset"
    
    train_dir, val_dir = prepare_dataset(augmented_dir, output_dir)
