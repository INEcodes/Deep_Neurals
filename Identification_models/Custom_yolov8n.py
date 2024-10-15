import torch
import torch.nn as nn
from ultralytics import YOLO

class CustomModel(nn.Module):
    def __init__(self, pretrained_model):
        super(CustomModel, self).__init__()
        self.pretrained = pretrained_model  # Load the pre-trained YOLOv8 model
        self.new_layer = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            # Add more layers as needed
        )
    
    def forward(self, x):
        x = self.pretrained(x)
        x = self.new_layer(x)
        return x

# Load the pre-trained YOLOv8 model
model = YOLO('yolov8n.pt').model

# Wrap the pre-trained model with your custom model
custom_model = CustomModel(model)
print(custom_model)