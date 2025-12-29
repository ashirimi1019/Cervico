import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
from torchvision import transforms
import os
from PIL import Image
import numpy as np
from typing import Tuple, Optional
import logging

from .preprocessing import UltrasoundPreprocessor
from .model import CervicalDilationModel
from .utils import TrainingMonitor, evaluate_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CervicalDataset(Dataset):
    def __init__(self, data_dir: str, transform=None, training: bool = False):
        """
        Dataset for cervical dilation images
        
        Args:
            data_dir: Directory containing the images
            transform: Optional transform to apply
            training: Whether this is for training (enables augmentation)
        """
        self.data_dir = data_dir
        self.transform = transform or UltrasoundPreprocessor(training=training)
        self.samples = []
        
        # Collect all image files and their labels
        for file in os.listdir(data_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Only process files that start with a number and end with 'cm'
                if file.lower().split('cm')[0].replace('.', '').isdigit():
                    try:
                        dilation = float(file.split('cm')[0])
                        if 0 <= dilation <= 10:  # Valid dilation range
                            self.samples.append((file, dilation))
                    except ValueError:
                        continue
        
        if not self.samples:
            raise ValueError(f"No valid cervical dilation images found in {data_dir}")
            
        logger.info(f"Found {len(self.samples)} valid samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        filename, dilation = self.samples[idx]
        image_path = os.path.join(self.data_dir, filename)
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        
        if self.transform:
            image = self.transform.preprocess_image(image)
        
        # Convert to torch tensor and ensure correct shape (C, H, W)
        image = torch.tensor(image).float()
        if image.shape[-1] == 3:  # If channels are last
            image = image.permute(2, 0, 1)  # Move channels to first dimension
        
        # Create labels
        class_label = int(round(dilation))  # Round to nearest cm for classification
        regression_label = dilation  # Keep exact value for regression
        
        return image, torch.tensor(class_label), torch.tensor(regression_label)

def train_model(
    train_dir: str,
    val_dir: str,
    model_save_dir: str,
    num_epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    device: Optional[str] = None,
    patience: int = 10  # Early stopping patience
):
    """
    Train the cervical dilation model
    
    Args:
        train_dir: Directory containing training images
        val_dir: Directory containing validation images
        model_save_dir: Directory to save model checkpoints
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        device: Device to train on (cuda/cpu)
        patience: Number of epochs to wait for improvement before early stopping
    """
    # Create model save directory
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    logger.info(f"Training on device: {device}")
    
    # Create datasets
    train_dataset = CervicalDataset(train_dir, training=True)
    val_dataset = CervicalDataset(val_dir, training=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model and move to device
    model = CervicalDilationModel()
    model = model.to(device)
    
    # Loss functions
    classification_criterion = torch.nn.CrossEntropyLoss()
    regression_criterion = torch.nn.MSELoss()
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training monitoring
    monitor = TrainingMonitor(model_save_dir)
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = []
        
        for batch_idx, (images, class_labels, reg_labels) in enumerate(train_loader):
            images = images.to(device)
            class_labels = class_labels.to(device)
            reg_labels = reg_labels.to(device)
            
            optimizer.zero_grad()
            class_out, reg_out = model(images)
            
            # Calculate losses
            class_loss = classification_criterion(class_out, class_labels)
            reg_loss = regression_criterion(reg_out.squeeze(), reg_labels)
            total_loss = class_loss + reg_loss
            
            total_loss.backward()
            optimizer.step()
            
            train_losses.append(total_loss.item())
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, "
                          f"Loss: {total_loss.item():.4f}")
        
        avg_train_loss = sum(train_losses) / len(train_losses)
        
        # Validation phase
        model.eval()
        val_losses = []
        val_metrics = []
        
        with torch.no_grad():
            for images, class_labels, reg_labels in val_loader:
                images = images.to(device)
                class_labels = class_labels.to(device)
                reg_labels = reg_labels.to(device)
                
                class_out, reg_out = model(images)
                
                # Calculate losses
                class_loss = classification_criterion(class_out, class_labels)
                reg_loss = regression_criterion(reg_out.squeeze(), reg_labels)
                total_loss = class_loss + reg_loss
                
                val_losses.append(total_loss.item())
                
                # Calculate metrics
                metrics = evaluate_model(class_out, reg_out, class_labels, reg_labels)
                val_metrics.append(metrics)
        
        avg_val_loss = sum(val_losses) / len(val_losses)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Log metrics
        avg_metrics = {k: sum(m[k] for m in val_metrics) / len(val_metrics) 
                      for k in val_metrics[0].keys()}
        
        logger.info(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, "
                   f"Val Loss = {avg_val_loss:.4f}")
        logger.info(f"Validation Metrics: {avg_metrics}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'metrics': avg_metrics
            }, os.path.join(model_save_dir, 'best_model.pt'))
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if epochs_without_improvement >= patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    logger.info("Training completed!")
    return model

if __name__ == "__main__":
    # Train the model
    train_model(
        train_dir="/Users/harimanivannan/Documents/GitHub/Cervico/ai/data/dataset/train",
        val_dir="/Users/harimanivannan/Documents/GitHub/Cervico/ai/data/dataset/val",
        model_save_dir="/Users/harimanivannan/Documents/GitHub/Cervico/ai/models",
        num_epochs=50,
        batch_size=16,
        learning_rate=1e-4
    )
