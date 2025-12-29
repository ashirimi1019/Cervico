import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Tuple

class CervicalDilationModel(nn.Module):
    def __init__(self, num_classes: int = 11, pretrained: bool = True):
        """
        Cervical dilation prediction model based on EfficientNet-B0
        
        Args:
            num_classes: Number of dilation classes (0-10cm)
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        
        # Use EfficientNet-B0 as backbone
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        
        # Replace classifier head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=in_features, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=512, out_features=num_classes)
        )
        
        # Add regression head for precise measurements
        self.regression_head = nn.Sequential(
            nn.Linear(in_features=num_classes, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            classification_output: Class probabilities
            regression_output: Precise dilation measurement
        """
        classification_output = self.backbone(x)
        regression_output = self.regression_head(classification_output) * 10.0
        return classification_output, regression_output

class DilationPredictor:
    def __init__(self, model_path: Optional[str] = None, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = CervicalDilationModel().to(self.device)
        
        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> Tuple[float, float]:
        """
        Predict cervical dilation
        
        Args:
            image: Preprocessed image tensor
            
        Returns:
            class_prediction: Predicted dilation class
            precise_measurement: Precise dilation measurement in cm
        """
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        class_output, regression_output = self.model(image)
        
        class_prediction = torch.argmax(class_output, dim=1)[0].item()
        precise_measurement = regression_output[0].item()
        
        return class_prediction, precise_measurement
