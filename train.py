"""
Retinal AI: Lesion Detector - Dual Architecture Training Pipeline
Fine-tunes EfficientNet-B3 and ResNet-50 for multi-label retinal disease detection:
Classes:
  0: Diabetic Retinopathy (DR)
  1: Glaucoma
  2: Cataract
  3: AMD (Age-Related Macular Degeneration)
"""

import os
import argparse
import time
import numpy as np
import pandas as pd
from PIL import Image
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

CLASSES = ['Diabetic Retinopathy (DR)', 'Glaucoma', 'Cataract', 'AMD']

def apply_clahe(img_np):
    """Enhance fundus local contrast in LAB color space."""
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

class RetinalDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, use_clahe=True):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.use_clahe = use_clahe

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert('RGB')
        img_np = np.array(img)
        if self.use_clahe:
            img_np = apply_clahe(img_np)
        img_pil = Image.fromarray(img_np)
        
        if self.transform:
            tensor = self.transform(img_pil)
        else:
            tensor = transforms.ToTensor()(img_pil)
            
        target = torch.tensor(self.labels[idx], dtype=torch.float32)
        return tensor, target

class RetinalScreener(nn.Module):
    def __init__(self, arch="effnet", num_classes=4, pretrained=True):
        super().__init__()
        self.arch = arch
        if arch == "effnet":
            weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
            self.base = models.efficientnet_b3(weights=weights)
            in_features = self.base.classifier[1].in_features
            self.base.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, num_classes)
            )
        else:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.base = models.resnet50(weights=weights)
            in_features = self.base.fc.in_features
            self.base.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, num_classes)
            )

    def forward(self, x):
        return self.base(x)

def get_transforms(img_size=300):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            running_loss += loss.item() * inputs.size(0)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.cpu().numpy())

    total_loss = running_loss / len(dataloader.dataset)
    all_targets = np.vstack(all_targets)
    all_probs = np.vstack(all_probs)
    all_preds = (all_probs >= 0.5).astype(int)

    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    micro_f1 = f1_score(all_targets, all_preds, average='micro', zero_division=0)
    subset_acc = accuracy_score(all_targets, all_preds)

    return total_loss, macro_f1, micro_f1, subset_acc

def train_model(arch="effnet", epochs=15, batch_size=16, lr=1e-4, device='cuda'):
    print(f"\n=======================================================")
    print(f"🚀 Initializing Fine-Tuning Pipeline for: {arch.upper()}")
    print(f"=======================================================")
    
    model = RetinalScreener(arch=arch, num_classes=4, pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Output directory
    os.makedirs("models", exist_ok=True)
    save_path = f"models/best_{'efficientnet_b3' if arch == 'effnet' else 'resnet50'}.pth"

    print(f"Model initialized on {device}. Checkpoints will save to {save_path}")
    return model, save_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Retinal Disease Screener")
    parser.add_argument("--arch", type=str, default="effnet", choices=["effnet", "resnet"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Torch Device: {device}")
