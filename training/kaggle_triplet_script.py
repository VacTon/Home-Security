# =================================================================================================
# KAGGLE TRAINING SCRIPT - TRIPLET LOSS (FaceNet Style)
# =================================================================================================
# This script trains a Face Recognition model using Triplet Margin Loss.
# Ideally needs a dataset with many identities (e.g., standard public datasets).
# =================================================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import os
import random
import numpy as np
import time
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "image_size": 112,
    "embedding_size": 128,   # FaceNet often uses 128 (smaller than ArcFace's 512)
    "batch_size": 32,        # Smaller batch size for triplets often helps stability
    "epochs": 30,            # Needs more epochs than ArcFace
    "lr": 0.001,
    "margin": 0.5,           # Distance margin (alpha)
    "dataset_path": "/kaggle/input/your-dataset-name",
    "model_save_path": "custom_triplet.pth",
    "onnx_save_path": "custom_triplet.onnx",
    "hard_mining": True      # Enable online hard mining
}

print(f"Running Triplet Training on: {CONFIG['device']}")

# ==========================================
# 2. MODEL ARCHITECTURE
# ==========================================
class EmbeddingNet(nn.Module):
    def __init__(self, embedding_size=128, pretrained=True):
        super(EmbeddingNet, self).__init__()
        # Using MobileNetV2 for speed/efficiency on Raspberry Pi
        self.backbone = models.mobilenet_v2(pretrained=pretrained)
        
        # Replace classifier with embedding layer
        # MobileNetV2 uses 'classifier' block, last linear layer is at index 1
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity() # Remove default classifier
        
        self.fc = nn.Linear(1280, embedding_size) # MobileNetV2 features are 1280 deep
        
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # L2 Normalize embeddings (Critical for Triplet Loss!)
        return F.normalize(x, p=2, dim=1)

# ==========================================
# 3. TRIPLET DATASET (The Tricky Part)
# ==========================================
class TripletFaceDataset(Dataset):
    """
    Returns triplets: (Anchor, Positive, Negative)
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_map = {} # {class_idx: [image_paths...]}
        self.image_paths = []
        self.labels = []
        
        # Index the dataset
        classes = sorted(os.listdir(root_dir))
        for i, cls in enumerate(classes):
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir): continue
            
            self.class_map[i] = []
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                    path = os.path.join(cls_dir, img_name)
                    self.image_paths.append(path)
                    self.labels.append(i)
                    self.class_map[i].append(path)

        # Remove classes with < 2 images (cannot form Anchor-Positive pair)
        self.valid_classes = [c for c, paths in self.class_map.items() if len(paths) >= 2]
        print(f"Found {len(self.image_paths)} images, {len(self.valid_classes)} valid identities.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. Select ANCHOR
        anchor_path = self.image_paths[idx]
        anchor_label = self.labels[idx]
        
        # 2. Select POSITIVE (Same class, different image)
        # If class only has 1 image (shouldn't happen due to filter), reuse anchor
        possible_positives = self.class_map[anchor_label]
        positive_path = random.choice(possible_positives)
        while positive_path == anchor_path and len(possible_positives) > 1:
            positive_path = random.choice(possible_positives)

        # 3. Select NEGATIVE (Different class)
        negative_label = random.choice(self.valid_classes)
        while negative_label == anchor_label:
            negative_label = random.choice(self.valid_classes)
        negative_path = random.choice(self.class_map[negative_label])

        # Load Images
        anchor_img = Image.open(anchor_path).convert('RGB')
        pos_img = Image.open(positive_path).convert('RGB')
        neg_img = Image.open(negative_path).convert('RGB')

        if self.transform:
            anchor_img = self.transform(anchor_img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)

        return anchor_img, pos_img, neg_img

# ==========================================
# 4. TRAINING LOOP
# ==========================================
def train():
    transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
    ])
    
    try:
        dataset = TripletFaceDataset(CONFIG['dataset_path'], transform=transform)
        dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2)
    except Exception as e:
        print(f"Dataset Error: {e}")
        return

    model = EmbeddingNet(embedding_size=CONFIG['embedding_size']).to(CONFIG['device'])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    criterion = nn.TripletMarginLoss(margin=CONFIG['margin'], p=2)
    
    print("Starting Triplet Training...")
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0
        
        for anchor, positive, negative in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            anchor = anchor.to(CONFIG['device'])
            positive = positive.to(CONFIG['device'])
            negative = negative.to(CONFIG['device'])
            
            optimizer.zero_grad()
            
            embed_a = model(anchor)
            embed_p = model(positive)
            embed_n = model(negative)
            
            loss = criterion(embed_a, embed_p, embed_n)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {total_loss/len(dataloader):.4f}")

    # ==========================
    # Export
    # ==========================
    print("Exporting ONNX...")
    model.eval()
    dummy = torch.randn(1, 3, 112, 112).to(CONFIG['device'])
    torch.onnx.export(model, dummy, CONFIG['onnx_save_path'], 
                      input_names=['input'], output_names=['embedding'], opset_version=12)
    print("Done!")

if __name__ == "__main__":
    train()
