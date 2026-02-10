#!/usr/bin/env python3
"""
Pokemon Type Classification - FIXED VERSION with Transfer Learning
Uses ResNet18 pretrained on ImageNet for superior performance
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cpu':
    print("⚠️  WARNING: Training on CPU will be VERY slow (2-4 hours)")
    print("   Consider using Google Colab (free GPU) or Kaggle notebooks")

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

class PokemonDataset(Dataset):
    """Custom dataset for Pokemon images"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image and label 0 as fallback
            return torch.zeros(3, 224, 224), 0

def load_dataset(data_dir='pokemon'):
    """Load Pokemon dataset from directory"""
    
    print("\n" + "="*60)
    print("LOADING DATASET")
    print("="*60)
    
    # Read CSV file
    csv_path = os.path.join(data_dir, 'pokemon.csv')
    df = pd.read_csv(csv_path)
    
    # Get image directory
    images_dir = os.path.join(data_dir, 'images')
    
    # Prepare data
    image_paths = []
    labels = []
    
    for idx, row in df.iterrows():
        img_name = row['Name'].lower().replace(' ', '-').replace('.', '').replace("'", '')
        img_path = os.path.join(images_dir, f"{img_name}.png")
        
        if os.path.exists(img_path):
            image_paths.append(img_path)
            labels.append(row['Type1'])
            
    # Create label encoding
    unique_labels = sorted(list(set(labels)))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    # Convert labels to indices
    label_indices = [label_to_idx[label] for label in labels]
    
    print(f"✓ Found {len(image_paths)} images")
    print(f"✓ Number of classes: {len(unique_labels)}")
    print(f"✓ Classes: {unique_labels}")
    
    # Check class distribution
    from collections import Counter
    label_counts = Counter(labels)
    print(f"\n📊 Class distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"   {label}: {count} images")
    print(f"   ... ({len(label_counts)} total classes)")
    
    return image_paths, label_indices, label_to_idx, idx_to_label

def get_transforms(image_size=224):
    """
    Get data augmentation transforms
    CRITICAL: Using ImageNet pretrained stats [0.485, 0.456, 0.406] and [0.229, 0.224, 0.225]
    """
    
    # Training transforms with MODERATE augmentation
    train_transform = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),  # Slightly larger for cropping
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),  # Moderate rotation
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Small translation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transforms - NO augmentation
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_model(num_classes, pretrained=True):
    """
    Create ResNet18 model with transfer learning
    CRITICAL: Using pretrained weights is ESSENTIAL for small datasets
    """
    
    print("\n" + "="*60)
    print("CREATING MODEL")
    print("="*60)
    
    # Load pretrained ResNet18
    model = models.resnet18(pretrained=pretrained)
    
    # Freeze early layers (optional - can unfreeze for fine-tuning)
    # for param in model.parameters():
    #     param.requires_grad = False
    
    # Replace final FC layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.2),  # REDUCED from 0.5!
        nn.Linear(num_features, num_classes)
    )
    
    print(f"✓ Model: ResNet18 (pretrained={pretrained})")
    print(f"✓ Final layer: Linear({num_features} → {num_classes})")
    print(f"✓ Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"✓ Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model

def calculate_class_weights(labels, num_classes):
    """Calculate class weights for imbalanced dataset"""
    from collections import Counter
    label_counts = Counter(labels)
    
    # Calculate weights (inverse frequency)
    weights = torch.zeros(num_classes)
    for label, count in label_counts.items():
        weights[label] = 1.0 / count
    
    # Normalize weights
    weights = weights / weights.sum() * num_classes
    
    return weights

def train_epoch(model, loader, criterion, optimizer, device, grad_clip=1.0):
    """Train for one epoch with gradient clipping"""
    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training', leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # CRITICAL!
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/(pbar.n+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    """Validate the model"""
    
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validating', leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f'{running_loss/(pbar.n+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    return running_loss / len(loader), 100. * correct / total, all_preds, all_labels

def plot_training_history(history, save_path='training_history.png'):
    """Plot training history"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-o', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, history['train_acc'], 'b-o', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-o', label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot learning rate
    ax3.plot(epochs, history['learning_rates'], 'g-o', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Learning Rate', fontsize=12)
    ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Plot loss difference (overfitting indicator)
    loss_diff = [val - train for train, val in zip(history['train_loss'], history['val_loss'])]
    ax4.plot(epochs, loss_diff, 'purple', linewidth=2)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Val Loss - Train Loss', fontsize=12)
    ax4.set_title('Overfitting Indicator (Lower is Better)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Training history plot saved to {save_path}")
    plt.close()

def main():
    """Main training function"""
    
    # ========== HYPERPARAMETERS ==========
    BATCH_SIZE = 32  # Reduced for CPU
    NUM_EPOCHS = 30  # Reduced from 50
    LEARNING_RATE = 0.001
    IMAGE_SIZE = 224  # Standard for ImageNet models
    GRAD_CLIP = 1.0
    PATIENCE = 7  # Early stopping patience
    
    # Auto-detect dataset directory
    import sys
    if len(sys.argv) > 1:
        DATA_DIR = sys.argv[1]
    else:
        possible_paths = ['pokemon', 'archive', 'PokemonData', '.']
        DATA_DIR = None
        for path in possible_paths:
            if os.path.exists(os.path.join(path, 'pokemon.csv')):
                DATA_DIR = path
                break
        
        if DATA_DIR is None:
            print("❌ ERROR: Cannot find pokemon dataset!")
            print("Run: python train_pokemon_classifier_fixed.py <path_to_dataset>")
            sys.exit(1)
    
    print("\n" + "="*60)
    print("POKEMON TYPE CLASSIFIER - TRANSFER LEARNING")
    print("="*60)
    print(f"Device: {device}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    
    # Load dataset
    image_paths, labels, label_to_idx, idx_to_label = load_dataset(DATA_DIR)
    
    # Calculate class weights for imbalanced data
    num_classes = len(label_to_idx)
    class_weights = calculate_class_weights(labels, num_classes).to(device)
    print(f"\n✓ Class weights calculated for {num_classes} classes")
    
    # Split dataset with stratification
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\n📊 Dataset split:")
    print(f"   Training samples: {len(train_paths)}")
    print(f"   Validation samples: {len(val_paths)}")
    print(f"   Train/Val ratio: {len(train_paths)/len(val_paths):.1f}:1")
    
    # Get transforms
    train_transform, val_transform = get_transforms(IMAGE_SIZE)
    
    # Create datasets
    train_dataset = PokemonDataset(train_paths, train_labels, train_transform)
    val_dataset = PokemonDataset(val_paths, val_labels, val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2 if device.type == 'cuda' else 0,  # 0 for CPU
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2 if device.type == 'cuda' else 0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model with transfer learning
    model = create_model(num_classes, pretrained=True).to(device)
    
    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer - using AdamW (better than Adam)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # Learning rate scheduler - Cosine Annealing (smooth decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60 + "\n")
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, GRAD_CLIP
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_labels_list = validate(
            model, val_loader, criterion, device
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Step scheduler
        scheduler.step()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'label_to_idx': label_to_idx,
                'idx_to_label': idx_to_label,
                'class_weights': class_weights.cpu()
            }, 'best_pokemon_model.pth')
            print(f"✓ Best model saved! Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
                break
        
        print()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"✓ Best validation accuracy: {best_val_acc:.2f}%")
    print(f"✓ Total epochs trained: {len(history['train_acc'])}")
    
    # Plot training history
    plot_training_history(history, 'training_history.png')
    
    # Save label mappings
    with open('label_mappings.json', 'w') as f:
        json.dump({
            'label_to_idx': label_to_idx,
            'idx_to_label': idx_to_label
        }, f, indent=2)
    
    print(f"✓ Label mappings saved to label_mappings.json")
    print("\n🎉 All done! Model ready for testing.")

if __name__ == '__main__':
    main()