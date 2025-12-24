# =================================================================================================
# KAGGLE TRAINING SCRIPT - CUSTOM ARCFACE
# =================================================================================================
# Copy this entire script into a Kaggle Notebook cell to train your own Face Recognition model.
#
# PREREQUISITES (Kaggle Environment):
# - GPU: Enabled (Tesla T4 x2 or P100 recommended)
# - Internet: Enabled (to download pretrained weights if needed)
# - Dataset: You must upload your 'faces' folder as a Kaggle Dataset.
# =================================================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import os
import math
import numpy as np
import time
from tqdm import tqdm  # Progress bar

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "image_size": 112,       # Standard ArcFace input size
    "embedding_size": 512,   # Output vector size
    "batch_size": 64,        # Adjust based on GPU memory
    "epochs": 20,
    "lr": 0.01,
    "dataset_path": "/kaggle/input/your-dataset-name",  # UPDATE THIS LINE ON KAGGLE
    "model_save_path": "custom_arcface.pth",
    "onnx_save_path": "custom_arcface.onnx"
}

print(f"Running on device: {CONFIG['device']}")

# ==========================================
# 2. MODEL ARCHITECTURE (Backbone)
# ==========================================
class MobileFaceNet(nn.Module):
    """
    A lightweight backbone optimized for Mobile/Edge devices (like Raspberry Pi).
    We use a modified ResNet18 for simplicity and learning purposes here, 
    but MobileFaceNet is the goal for efficiency.
    """
    def __init__(self, embedding_size=512, pretrained=True):
        super(MobileFaceNet, self).__init__()
        # Load a standard ResNet18
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Modify the last layer to output 512-d embedding instead of 1000 classes
        # The 'fc' (Fully Connected) layer is the bottleneck
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, embedding_size)
        
        # Batch Norm to normalize embeddings (crucial for ArcFace)
        self.bn = nn.BatchNorm1d(embedding_size)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.bn(x)
        # Normalize the features to inputs of unit length
        # This is strictly required for ArcFace Loss
        return F.normalize(x, p=2, dim=1)

# ==========================================
# 3. ARCFACE HEAD (The "Magic" Loss Function)
# ==========================================
class ArcFaceHead(nn.Module):
    """
    Implements the ArcFace loss margin.
    It forces the model to push faces of different people APART 
    and pull faces of the same person TOGETHER in 512-d space.
    """
    def __init__(self, embedding_size, num_classes, s=64.0, m=0.50):
        super(ArcFaceHead, self).__init__()
        self.s = s  # Scale factor
        self.m = m  # Margin
        self.W = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.W)

    def forward(self, embeddings, labels):
        # 1. Normalize weights
        W_norm = F.normalize(self.W, p=2, dim=1)
        
        # 2. Calculate Cosine Similarity (Embeddings . Weights)
        cosine = F.linear(embeddings, W_norm)
        
        # 3. Get the cosine of the target labels
        # We only apply the margin to the ground truth class
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        
        # 4. Add the angular margin (m) to the target theta
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        target_logit = torch.cos(theta + self.m)
        
        # 5. Combine: Use target_logit for correct class, cosine for others
        output = one_hot * target_logit + (1.0 - one_hot) * cosine
        
        # 6. Scale by s
        output *= self.s
        return output

# ==========================================
# 4. DATASET LOADER
# ==========================================
class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_names = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.class_names)}
        
        print(f"Found {len(self.class_names)} classes (people).")
        
        for cls in self.class_names:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(cls_dir, img_name))
                    self.labels.append(self.class_to_idx[cls])
        
        print(f"Total images: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# ==========================================
# 5. TRAINING LOOP
# ==========================================
def train():
    # Transforms (Crucial: Normalization)
    transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # [-1, 1] range
    ])
    
    # Load Data
    try:
        dataset = FaceDataset(CONFIG['dataset_path'], transform=transform)
        dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure you updated CONFIG['dataset_path']!")
        return

    # Initialize Model & Loss
    num_classes = len(dataset.class_names)
    backbone = MobileFaceNet(embedding_size=CONFIG['embedding_size']).to(CONFIG['device'])
    head = ArcFaceHead(embedding_size=CONFIG['embedding_size'], num_classes=num_classes).to(CONFIG['device'])
    
    # Optimizer (Optimizes both Backbone and Head)
    optimizer = optim.SGD([
        {'params': backbone.parameters()},
        {'params': head.parameters()}
    ], lr=CONFIG['lr'], momentum=0.9, weight_decay=5e-4)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    
    print("\nStarting Training...")
    
    for epoch in range(CONFIG['epochs']):
        backbone.train()
        head.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}"):
            images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
            
            optimizer.zero_grad()
            
            # Forward Pass
            embeddings = backbone(images)      # Get 512-d feature
            outputs = head(embeddings, labels) # Apply ArcFace margin
            
            # Loss Calculation
            loss = criterion(outputs, labels)
            
            # Backward Pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        scheduler.step()
        epoch_acc = 100 * correct / total
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}: Loss={total_loss/len(dataloader):.4f} | Acc={epoch_acc:.2f}% | Time={epoch_time:.1f}s")
        
    print("\nTraining Complete!")
    
    # ==========================================
    # 6. EXPORT TO ONNX (For Raspberry Pi)
    # ==========================================
    print(f"Exporting to ONNX: {CONFIG['onnx_save_path']}")
    backbone.eval()
    
    dummy_input = torch.randn(1, 3, 112, 112).to(CONFIG['device'])
    
    torch.onnx.export(backbone, 
                      dummy_input, 
                      CONFIG['onnx_save_path'],
                      input_names=['input'], 
                      output_names=['embedding'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'embedding': {0: 'batch_size'}},
                      opset_version=12)
                      
    print("Export Success! You can now download the .onnx file.")

if __name__ == "__main__":
    train()
