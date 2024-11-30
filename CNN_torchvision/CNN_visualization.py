import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from graphviz import Digraph
import torch.nn.functional as F


# Define the CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# Function to visualize the CNN architecture
def visualize_cnn(model):
    """
    Visualize a CNN model architecture using graphviz.
    """
    dot = Digraph(format='png', graph_attr={'rankdir': 'LR'})
    dot.attr(dpi='100')

    # Input layer
    dot.node('Input', label='Input\n(3x32x32)', shape='box', style='filled', fillcolor='lightblue')

    # Add convolutional layers
    dot.node('Conv1', label='Conv2D\n32 filters\n3x3 kernel', shape='box', style='filled', fillcolor='lightgreen')
    dot.node('Conv2', label='Conv2D\n64 filters\n3x3 kernel', shape='box', style='filled', fillcolor='lightgreen')
    dot.node('Conv3', label='Conv2D\n128 filters\n3x3 kernel', shape='box', style='filled', fillcolor='lightgreen')

    # Add pooling layers
    dot.node('Pool1', label='MaxPool2D\n2x2', shape='ellipse', style='filled', fillcolor='lightyellow')
    dot.node('Pool2', label='MaxPool2D\n2x2', shape='ellipse', style='filled', fillcolor='lightyellow')
    dot.node('Pool3', label='MaxPool2D\n2x2', shape='ellipse', style='filled', fillcolor='lightyellow')

    # Flatten layer
    dot.node('Flatten', label='Flatten\n128x4x4 → 2048', shape='box', style='filled', fillcolor='lightpink')

    # Fully connected layers
    dot.node('FC1', label='Dense\n512 units\nReLU', shape='box', style='filled', fillcolor='lightgrey')
    dot.node('Dropout', label='Dropout\n0.25', shape='ellipse', style='filled', fillcolor='lightcoral')
    dot.node('FC2', label='Dense\n10 units\nSoftmax', shape='box', style='filled', fillcolor='lightgrey')

    # Define edges
    dot.edges([('Input', 'Conv1'), ('Conv1', 'Pool1'), 
               ('Pool1', 'Conv2'), ('Conv2', 'Pool2'),
               ('Pool2', 'Conv3'), ('Conv3', 'Pool3'),
               ('Pool3', 'Flatten'), ('Flatten', 'FC1'),
               ('FC1', 'Dropout'), ('Dropout', 'FC2')])

    return dot


# Initialize the model
cnn_model = Net()

# Visualize the CNN
cnn_graph = visualize_cnn(cnn_model)

# Save and render the visualization
cnn_graph.render('cnn_architecture', view=True)  # Save and open the PNG file
