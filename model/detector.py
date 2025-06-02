import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pretrained ResNet18 with default weights
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Replace the final layer for binary classification (real/fake)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x):
        # Output: Raw logits (no sigmoid yet)
        return self.model(x)
