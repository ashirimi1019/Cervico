import os
import torch
import logging
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json

from .train import CervicalDataset
from .model import DilationPredictor
from .utils import evaluate_model, visualize_predictions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model(
    model_path: str,
    test_dir: str,
    results_dir: str,
    batch_size: int = 16,
    device: str = None
) -> Dict[str, float]:
    """
    Test the trained model on a test dataset
    
    Args:
        model_path: Path to the trained model checkpoint
        test_dir: Directory containing test images
        results_dir: Directory to save test results
        batch_size: Batch size for testing
        device: Device to run inference on
    
    Returns:
        Dictionary of test metrics
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    predictor = DilationPredictor(model_path=model_path, device=device)
    model = predictor.model
    
    # Create test dataset and dataloader
    test_dataset = CervicalDataset(test_dir, training=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Test metrics
    test_metrics = []
    all_images = []
    all_true_dilations = []
    all_pred_dilations = []
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        for images, class_labels, reg_labels in test_loader:
            images = images.to(device)
            class_labels = class_labels.to(device)
            reg_labels = reg_labels.to(device)
            
            # Get predictions
            class_out, reg_out = model(images)
            
            # Calculate metrics
            batch_metrics = evaluate_model(class_out, reg_out, class_labels, reg_labels)
            test_metrics.append(batch_metrics)
            
            # Store results for visualization
            all_images.extend([img.cpu().numpy().transpose(1, 2, 0) for img in images])
            all_true_dilations.extend(reg_labels.cpu().numpy())
            all_pred_dilations.extend(reg_out.cpu().squeeze().numpy())
    
    # Calculate average metrics
    avg_metrics = {
        k: sum(m[k] for m in test_metrics) / len(test_metrics)
        for k in test_metrics[0].keys()
    }
    
    # Log results
    logger.info("Test Results:")
    for metric, value in avg_metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # Save metrics
    with open(os.path.join(results_dir, 'test_metrics.json'), 'w') as f:
        json.dump(avg_metrics, f, indent=4)
    
    # Visualize sample predictions
    n_samples = min(5, len(all_images))
    sample_indices = torch.randperm(len(all_images))[:n_samples]
    
    sample_images = [all_images[i] for i in sample_indices]
    sample_true = [all_true_dilations[i] for i in sample_indices]
    sample_pred = [all_pred_dilations[i] for i in sample_indices]
    
    visualize_predictions(
        sample_images,
        sample_true,
        sample_pred,
        save_path=os.path.join(results_dir, 'sample_predictions.png')
    )
    
    return avg_metrics

if __name__ == "__main__":
    # Test the model
    test_model(
        model_path="/Users/harimanivannan/Documents/GitHub/Cervico/ai/models/best_model.pt",
        test_dir="/Users/harimanivannan/Documents/GitHub/Cervico/ai/data/dataset/val",
        results_dir="/Users/harimanivannan/Documents/GitHub/Cervico/ai/results"
    )
