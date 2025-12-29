import torch
import os
import logging
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import Optional, Dict, List
import json

from .model import CervicalDilationModel, DilationPredictor
from .preprocessing import UltrasoundPreprocessor
from .utils import evaluate_model, visualize_predictions
from .train import CervicalDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_and_visualize(
    model_path: str,
    data_dir: str,
    output_dir: str,
    batch_size: int = 4,
    device: Optional[str] = None
) -> Dict[str, float]:
    """
    Evaluate model performance and generate visualizations
    
    Args:
        model_path: Path to trained model checkpoint
        data_dir: Directory containing test images
        output_dir: Directory to save results
        batch_size: Batch size for evaluation
        device: Device to run evaluation on
        
    Returns:
        Dictionary of evaluation metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model and load weights
    model = CervicalDilationModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Create dataset and dataloader
    dataset = CervicalDataset(data_dir, transform=UltrasoundPreprocessor(training=False))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Evaluate model
    metrics = evaluate_model(model, dataloader, device)
    logger.info(f"Evaluation metrics: {metrics}")
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Generate visualizations for a few samples
    with torch.no_grad():
        images, true_classes, true_regs = next(iter(dataloader))
        images = images.to(device)
        class_output, reg_output = model(images)
        
        # Convert to numpy for visualization
        images_np = images.cpu().numpy()
        true_dilations = true_regs.cpu().numpy()
        pred_dilations = reg_output.squeeze().cpu().numpy()
        
        # Visualize predictions
        visualize_predictions(
            images=images_np,
            true_dilations=true_dilations,
            pred_dilations=pred_dilations,
            save_path=os.path.join(output_dir, 'predictions.png')
        )
    
    return metrics

def export_model(model_path: str, output_path: str, device: Optional[str] = None) -> None:
    """
    Export model to TorchScript format
    
    Args:
        model_path: Path to trained model checkpoint
        output_path: Path to save exported model
        device: Device to run export on
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = CervicalDilationModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Create example input
    example_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Export to TorchScript
    traced_model = torch.jit.trace(model, example_input)
    torch.jit.save(traced_model, output_path)
    logger.info(f"Model exported to {output_path}")

if __name__ == "__main__":
    # Example usage
    evaluate_and_visualize(
        model_path="/Users/harimanivannan/Documents/GitHub/Cervico/ai/models/model_latest.pth",
        data_dir="/Users/harimanivannan/Documents/GitHub/Cervico/ai/data/images",
        output_dir="/Users/harimanivannan/Documents/GitHub/Cervico/ai/evaluation"
    )
    
    # Export model
    export_model(
        model_path="/Users/harimanivannan/Documents/GitHub/Cervico/ai/models/model_latest.pth",
        output_path="/Users/harimanivannan/Documents/GitHub/Cervico/ai/models/model_traced.pt"
    )
