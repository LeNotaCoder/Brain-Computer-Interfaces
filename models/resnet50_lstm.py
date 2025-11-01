import torch.nn as nn
import torchvision.models as models
from .base import BaseModel


class ResNet50LSTM(BaseModel):
    """ResNet50-LSTM with transfer learning (Section 4.2)"""
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        # Remove the final FC layer, keep feature extraction
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze ResNet
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Trainable LSTM and FC
        self.lstm = nn.LSTM(2048, 250, batch_first=True)
        self.bn_lstm = nn.BatchNorm1d(250)
        self.fc1 = nn.Linear(250, 100)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, 4)
    
    def forward(self, x):
        # Extract features
        x = self.features(x)
        # Flatten
        x = x.view(x.size(0), -1)
        # Reshape for LSTM
        x = x.unsqueeze(1)
        # LSTM
        x, _ = self.lstm(x)
        x = self.bn_lstm(x[:, -1, :])
        # FC layers
        x = torch.nn.functional.elu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
