import torch.nn as nn
from .base import BaseModel


class CustomCNNLSTM(BaseModel):
    """Custom CNN-LSTM (Section 4.1)"""
    def __init__(self):
        super().__init__()
        # Two convolutional-pooling blocks
        self.conv1 = nn.Conv2d(3, 5, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(5)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(5, 10, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(10)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # LSTM (250 units)
        self.lstm = nn.LSTM(10, 250, batch_first=True)
        self.bn_lstm = nn.BatchNorm1d(250)
        
        # Fully connected
        self.fc1 = nn.Linear(250, 100)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, 4)
        
        self._init_weights()
    
    def forward(self, x):
        # Conv blocks with ELU
        x = self.pool1(torch.nn.functional.elu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.nn.functional.elu(self.bn2(self.conv2(x))))
        
        # Reshape for LSTM
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, -1, c)
        
        # LSTM
        x, _ = self.lstm(x)
        x = self.bn_lstm(x[:, -1, :])
        
        # FC layers
        x = torch.nn.functional.elu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
