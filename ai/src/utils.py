import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
import os
import logging
from sklearn.metrics import mean_absolute_error, accuracy_score
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

class TrainingMonitor:
    def __init__(self, save_dir: str):
        """
        Monitor and visualize training progress
        
        Args:
            save_dir: Directory to save plots and metrics
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'class_accuracy': [],
            'regression_mae': [],
            'accuracy': [],
            'mae': [],
            'within_1cm': [],
            'within_2cm': []
        }
    
    def update(self, metrics: Dict[str, float]) -> None:
        """Update training history with new metrics"""
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
    
    def plot_losses(self) -> None:
        """Plot training and validation losses"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_loss'], label='Training Loss')
        if self.history['val_loss']:
            plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, 'loss_curves.png'))
        plt.close()
    
    def plot_metrics(self) -> None:
        """Plot accuracy and MAE metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(self.history['class_accuracy'], label='Classification Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Classification Accuracy')
        
        ax2.plot(self.history['regression_mae'], label='Regression MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Mean Absolute Error')
        ax2.set_title('Regression Error')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'metrics.png'))
        plt.close()

def visualize_predictions(
    images: List[np.ndarray],
    true_dilations: List[float],
    pred_dilations: List[float],
    save_path: str
) -> None:
    """
    Visualize model predictions on sample images
    
    Args:
        images: List of ultrasound images
        true_dilations: True dilation values
        pred_dilations: Predicted dilation values
        save_path: Path to save visualization
    """
    n_samples = len(images)
    fig, axes = plt.subplots(2, n_samples, figsize=(5*n_samples, 10))
    
    # Plot images with predictions
    for i in range(n_samples):
        axes[0, i].imshow(images[i], cmap='gray')
        axes[0, i].set_title(f'True: {true_dilations[i]:.1f}cm\nPred: {pred_dilations[i]:.1f}cm')
        axes[0, i].axis('off')
    
    # Plot scatter plot of predictions vs truth
    axes[1, 0].scatter(true_dilations, pred_dilations)
    axes[1, 0].plot([0, 10], [0, 10], 'r--')  # Perfect prediction line
    axes[1, 0].set_xlabel('True Dilation (cm)')
    axes[1, 0].set_ylabel('Predicted Dilation (cm)')
    axes[1, 0].set_title('Prediction vs Ground Truth')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def evaluate_model(
    class_output: torch.Tensor,
    reg_output: torch.Tensor,
    class_labels: torch.Tensor,
    reg_labels: torch.Tensor
) -> Dict[str, float]:
    """
    Evaluate model performance
    
    Args:
        class_output: Classification logits
        reg_output: Regression predictions
        class_labels: True class labels
        reg_labels: True regression values
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Move tensors to CPU for metric calculation
    class_output = class_output.detach().cpu()
    reg_output = reg_output.detach().cpu()
    class_labels = class_labels.cpu()
    reg_labels = reg_labels.cpu()
    
    # Classification metrics
    class_preds = torch.argmax(class_output, dim=1).numpy()
    accuracy = accuracy_score(class_labels.numpy(), class_preds)
    
    # Regression metrics
    mae = mean_absolute_error(reg_labels.numpy(), reg_output.squeeze().numpy())
    
    # Calculate percentage of predictions within 1cm and 2cm
    abs_diff = torch.abs(reg_output.squeeze() - reg_labels)
    within_1cm = (abs_diff <= 1.0).float().mean().item()
    within_2cm = (abs_diff <= 2.0).float().mean().item()
    
    return {
        'accuracy': accuracy,
        'mae': mae,
        'within_1cm': within_1cm,
        'within_2cm': within_2cm
    }

def evaluate_model_dataloader(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate model performance
    
    Args:
        model: Trained PyTorch model
        dataloader: DataLoader with test/validation data
        device: Device to run evaluation on
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    true_classes = []
    pred_classes = []
    true_dilations = []
    pred_dilations = []
    
    with torch.no_grad():
        for images, class_labels, reg_labels in dataloader:
            images = images.to(device)
            class_output, reg_output = model(images)
            
            # Handle classification output
            pred_class = torch.argmax(class_output, dim=1)
            true_classes.extend(class_labels.cpu().numpy().tolist())
            pred_classes.extend(pred_class.cpu().numpy().tolist())
            
            # Handle regression output
            pred_dilation = reg_output.squeeze()
            if isinstance(pred_dilation, torch.Tensor):
                if pred_dilation.dim() == 0:  # scalar tensor
                    pred_dilations.append(pred_dilation.item())
                else:
                    pred_dilations.extend(pred_dilation.cpu().numpy().tolist())
            else:  # already a scalar
                pred_dilations.append(float(pred_dilation))
                
            true_dilations.extend(reg_labels.cpu().numpy().tolist())
    
    # Convert lists to numpy arrays
    true_classes = np.array(true_classes)
    pred_classes = np.array(pred_classes)
    true_dilations = np.array(true_dilations)
    pred_dilations = np.array(pred_dilations)
    
    # Handle empty arrays
    if len(true_classes) == 0 or len(pred_classes) == 0:
        return {
            'classification_accuracy': 0.0,
            'regression_mae': float('inf')
        }
    
    metrics = {
        'classification_accuracy': accuracy_score(true_classes, pred_classes),
        'regression_mae': mean_absolute_error(true_dilations, pred_dilations)
    }
    
    return metrics
