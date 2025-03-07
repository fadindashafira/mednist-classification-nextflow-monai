#!/usr/bin/env python

import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121, resnet18

class SimpleCNN(nn.Module):
    def __init__(self, spatial_dims=2, in_channels=1, out_channels=6):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, out_channels)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def get_model(model_type, num_classes, in_channels=1):
    """
    Factory function to return the specified model architecture
    
    Args:
        model_type: String specifying model type (simple_cnn, densenet, resnet)
        num_classes: Number of output classes
        in_channels: Number of input channels
        
    Returns:
        A PyTorch model
    """
    if model_type == "simple_cnn":
        return SimpleCNN(in_channels=in_channels, out_channels=num_classes)
    elif model_type == "densenet":
        return DenseNet121(spatial_dims=2, in_channels=in_channels, out_channels=num_classes)
    elif model_type == "resnet":
        return resnet18(spatial_dims=2, in_channels=in_channels, out_channels=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from simple_cnn, densenet, resnet")