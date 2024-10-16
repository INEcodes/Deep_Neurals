import torch
from ultralytics import YOLO

# Load the pre-trained model object
pretrained_model = YOLO('yolov8n.pt').model

def add_custom_head(model):
    # Add your custom layers here
    model.add_module('custom_head', torch.nn.Sequential(
        torch.nn.Conv2d(1024, 512, 3, stride=1, padding=1),  # Example new layer
        torch.nn.ReLU()))
    return model

# Update the model architecture
updated_model = add_custom_head(pretrained_model)
print(updated_model)
