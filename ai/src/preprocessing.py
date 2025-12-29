import cv2
import numpy as np
import albumentations as A
from PIL import Image
from typing import Union, Tuple
import logging

logger = logging.getLogger(__name__)

class UltrasoundPreprocessor:
    def __init__(self, target_size: Tuple[int, int] = (224, 224), training: bool = False):
        """
        Preprocessor for ultrasound images
        
        Args:
            target_size: Target image size
            training: Whether to apply training augmentations
        """
        self.target_size = target_size
        self.training = training
        
        # Base transform pipeline
        self.base_transform = A.Compose([
            A.Resize(height=target_size[0], width=target_size[1]),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
            A.GaussianBlur(blur_limit=(3, 3), p=0.5),
        ])
        
        # Training augmentations
        self.train_transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
            ], p=0.5),
            A.GaussNoise(p=0.5),
            A.ElasticTransform(alpha=120, sigma=6, p=0.3),
        ])
        
        # Normalization as final step
        self.normalize = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess an input image.
        
        Args:
            image (np.ndarray): Input image in RGB format
            
        Returns:
            np.ndarray: Preprocessed image
        """
        # Ensure image is in RGB format
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
        # Apply base transformations
        transformed = self.base_transform(image=image)['image']
        
        # Apply training augmentations if in training mode
        if self.training:
            transformed = self.train_transform(image=transformed)['image']
        
        # Apply normalization
        transformed = self.normalize(image=transformed)['image']
        
        return transformed

    def batch_preprocess(self, images: list) -> np.ndarray:
        """
        Preprocess a batch of images
        
        Args:
            images: List of image paths or numpy arrays
            
        Returns:
            Batch of preprocessed images
        """
        processed = [self.preprocess_image(img) for img in images]
        return np.stack(processed)
