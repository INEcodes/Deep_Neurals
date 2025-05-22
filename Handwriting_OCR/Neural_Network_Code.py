import os

folder = 'C:/Users/Abhay/Documents/GitHub/Deep_Neurals/Handwriting_OCR/Dataset/data_subset'
image_files = os.listdir(folder)
print(f"Total images found: {len(image_files)}")
print(image_files[:10])  # print first 10 filenames

from PIL import Image
import matplotlib.pyplot as plt
import os

image_folder = 'C:/Users/Abhay/Documents/GitHub/Deep_Neurals/Handwriting_OCR/Dataset/data_subset'
image_name = 'b06-093-s01-05.png'  # pick one image you want to load

image_path = os.path.join(image_folder, image_name)
img = Image.open(image_path)

plt.imshow(img)
plt.axis('off')
plt.show()



import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Example: Suppose you have a dict mapping filename -> text label
# Replace this with your actual label loading logic
labels_dict = {
    'b06-093-s01-05.png': "This is example text",
    'c03-087c-s01-00.png': "Another sample sentence",
    # ...
}

class IAMHandwritingDataset(Dataset):
    def __init__(self, images_folder, labels_dict, transform=None):
        self.images_folder = images_folder
        self.labels_dict = labels_dict
        self.filenames = list(labels_dict.keys())
        self.transform = transform
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img_path = os.path.join(self.images_folder, filename)
        image = Image.open(img_path).convert('L')  # grayscale
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels_dict[filename]
        
        return image, label

# Define transforms: resize to 128x32, convert to tensor, normalize
transform = transforms.Compose([
    transforms.Resize((32, 128)),  # (height, width)
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# Folder path where images are stored
images_folder = "C:/Users/Abhay/Documents/GitHub/Deep_Neurals/Handwriting_OCR/Dataset/data_subset"

# Create Dataset and DataLoader
dataset = IAMHandwritingDataset(images_folder, labels_dict, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Test iteration
for imgs, labels in dataloader:
    print(imgs.shape)  # Expect [batch_size, 1, 32, 128]
    print(labels)
    break

import string
import torch.nn.utils.rnn as rnn_utils

# Step 1: Build vocabulary from all label strings
all_text = " ".join(labels_dict.values())
vocab = sorted(list(set(all_text)))  # unique characters sorted

# Add special tokens if needed
vocab = ['<blank>'] + vocab  # For CTC blank token at index 0

# Create char to index and index to char maps
char2idx = {c: i for i, c in enumerate(vocab)}
idx2char = {i: c for i, c in enumerate(vocab)}

print("Vocabulary:", vocab)

# Step 2: Encode labels to sequences of indices
def encode_label(text):
    return [char2idx[c] for c in text]

encoded_labels = [encode_label(label) for label in labels_dict.values()]

# Step 3: Create a collate_fn to pad sequences when loading batches
def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, 0)
    
    label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    encoded = [torch.tensor(encode_label(label), dtype=torch.long) for label in labels]
    
    padded_labels = rnn_utils.pad_sequence(encoded, batch_first=True, padding_value=0)
    return images, padded_labels, label_lengths

# Use this collate function in your DataLoader
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# Test batch with encoded labels
for imgs, labels_padded, lengths in dataloader:
    print("Image batch shape:", imgs.shape)
    print("Padded labels shape:", labels_padded.shape)
    print("Label lengths:", lengths)
    break

import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, img_height=32, n_channels=1, n_classes=len(vocab)):
        super(CRNN, self).__init__()
        # CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1),  # (batch,64,32,128)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (batch,64,16,64)
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # (batch,128,16,64)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (batch,128,8,32)
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # (batch,256,8,32)
            nn.ReLU(),
            nn.MaxPool2d((2,1), (2,1)),  # (batch,256,4,32)
        )
        
        # Calculate feature size for RNN input
        self.rnn_input_size = 256 * 4  # channels * height after conv
        
        # RNN layers (2-layer BiLSTM)
        self.rnn = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        # Fully connected output layer
        self.fc = nn.Linear(256*2, n_classes)  # bidirectional doubles hidden size
    
    def forward(self, x):
        # x shape: (batch, 1, 32, 128)
        conv_out = self.cnn(x)  # (batch, 256, 4, 32)
        
        batch_size, c, h, w = conv_out.size()
        # Permute and reshape for RNN: (batch, width, channels*height)
        rnn_in = conv_out.permute(0, 3, 1, 2).contiguous()  # (batch, width, channels, height)
        rnn_in = rnn_in.view(batch_size, w, c * h)  # (batch, width, feature_size)
        
        rnn_out, _ = self.rnn(rnn_in)  # (batch, width, 512)
        output = self.fc(rnn_out)  # (batch, width, n_classes)
        
        # For CTC loss, output shape should be (width, batch, n_classes)
        output = output.permute(1, 0, 2)
        
        return output
    
model = CRNN(n_classes=len(vocab))

criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

from torch.utils.data import random_split, DataLoader

# Assuming 'dataset' is your full dataset object

# Split dataset: 80% train, 20% val (adjust as needed)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders with your collate_fn
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)


import torch
import torch.nn.functional as F

# ✅ Set device and move model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

num_epochs = 5  # change as needed

for epoch in range(num_epochs):
    # --- Training ---
    model.train()
    train_loss = 0.0
    for imgs, labels_padded, label_lengths in train_loader:
        imgs = imgs.to(device)
        labels_padded = labels_padded.to(device)
        label_lengths = label_lengths.to(device)

        optimizer.zero_grad()

        outputs = model(imgs)  # (seq_len, batch, n_classes)
        log_probs = F.log_softmax(outputs, dim=2)

        batch_size = imgs.size(0)
        input_lengths = torch.full(size=(batch_size,), fill_value=log_probs.size(0), dtype=torch.long).to(device)

        loss = criterion(log_probs, labels_padded, input_lengths, label_lengths)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, labels_padded, label_lengths in val_loader:
            imgs = imgs.to(device)
            labels_padded = labels_padded.to(device)
            label_lengths = label_lengths.to(device)

            outputs = model(imgs)
            log_probs = F.log_softmax(outputs, dim=2)

            batch_size = imgs.size(0)
            input_lengths = torch.full(size=(batch_size,), fill_value=log_probs.size(0), dtype=torch.long).to(device)

            loss = criterion(log_probs, labels_padded, input_lengths, label_lengths)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    
from torch.utils.data import random_split

total_size = len(dataset)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

from torch.utils.data import DataLoader

test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

# Load processor and model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load a test image from the IAM dataset
image_path = "C:/Users/Abhay/Documents/GitHub/Deep_Neurals/Handwriting_OCR/Dataset/data_subset/b06-093-s01-05.png"
image = Image.open(image_path).convert("RGB")

# Preprocess
pixel_values = processor(images=image, return_tensors="pt").pixel_values
pixel_values = pixel_values.to(device)

# Generate prediction
generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("Predicted Text:", generated_text)

import os

folder = "C:/Users/Abhay/Documents/GitHub/Deep_Neurals/Handwriting_OCR/Dataset/data_subset"
image_files = os.listdir(folder)

# Print some sample files
print("Available images:", image_files[:100])  # See first 10 image files

# Pick a different image
image_path = os.path.join(folder, image_files[0])

image = Image.open(image_path).convert("RGB")
pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("Predicted Text:", generated_text)