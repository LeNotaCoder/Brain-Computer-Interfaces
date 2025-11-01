import torch.nn as nn
import torchvision.models as models
from .base import BaseModel


class InceptionV3LSTM(BaseModel):
    """Inception-v3-LSTM with transfer learning (Section 4.2)"""
    def __init__(self):
        super().__init__()
        # Load pretrained Inception-v3
        inception = models.inception_v3(pretrained=True)
        inception.aux_logits = False
        
        # Remove the final layers (fc and avgpool)
        # Keep only the feature extraction part
        self.features = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
        )
        
        # Freeze all feature layers
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Add adaptive pooling to get fixed size output
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Trainable LSTM and FC
        self.lstm = nn.LSTM(2048, 250, batch_first=True)
        self.bn_lstm = nn.BatchNorm1d(250)
        self.fc1 = nn.Linear(250, 100)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, 4)
    
    def forward(self, x):
        # Extract features
        x = self.features(x)
        # Pool to fixed size
        x = self.adaptive_pool(x)
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
