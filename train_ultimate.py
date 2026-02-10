#!/usr/bin/env python3
"""
Pokemon Type Classification - ULTIMATE FIX
Handles small datasets, class imbalance, and CPU training optimally
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
    print("⚠️  Training on CPU - Will take 30-60 minutes")

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

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
            image = Image.open(img_path).convert('RGB')  # RGBA -> RGB
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(3, 224, 224), 0

def load_dataset(data_dir='pokemon', min_samples_per_class=5):
    """Load Pokemon dataset and handle classes with too few samples"""
    
    print("\n" + "="*60)
    print("LOADING DATASET")
    print("="*60)
    
    csv_path = os.path.join(data_dir, 'pokemon.csv')
    df = pd.read_csv(csv_path)
    
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
    
    # Count samples per class
    from collections import Counter
    label_counts = Counter(labels)
    
    # CRITICAL FIX: Remove or merge classes with too few samples
    print(f"\n📊 Original class distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        status = "❌ TOO SMALL" if count < min_samples_per_class else "✓"
        print(f"   {label}: {count} {status}")
    
    # Find classes to remove
    small_classes = [label for label, count in label_counts.items() 
                     if count < min_samples_per_class]
    
    if small_classes:
        print(f"\n⚠️  Removing {len(small_classes)} classes with <{min_samples_per_class} samples:")
        print(f"   {small_classes}")
        
        # Filter out small classes
        filtered_paths = []
        filtered_labels = []
        for path, label in zip(image_paths, labels):
            if label not in small_classes:
                filtered_paths.append(path)
                filtered_labels.append(label)
        
        image_paths = filtered_paths
        labels = filtered_labels
        
        print(f"   Remaining samples: {len(image_paths)}")
    
    # Create label encoding
    unique_labels = sorted(list(set(labels)))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    # Convert labels to indices
    label_indices = [label_to_idx[label] for label in labels]
    
    print(f"\n✓ Final dataset:")
    print(f"   Total images: {len(image_paths)}")
    print(f"   Number of classes: {len(unique_labels)}")
    print(f"   Classes: {unique_labels}")
    
    # Show final distribution
    final_counts = Counter(labels)
    print(f"\n📊 Final class distribution:")
    for label in sorted(unique_labels):
        count = final_counts[label]
        print(f"   {label}: {count}")
    
    return image_paths, label_indices, label_to_idx, idx_to_label

def get_transforms(image_size=224, augment_level='moderate'):
    """Get data augmentation transforms"""
    
    if augment_level == 'light':
        # Light augmentation for very small datasets
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    elif augment_level == 'moderate':
        # Moderate augmentation
        train_transform = transforms.Compose([
            transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:  # heavy
        # Heavy augmentation for small datasets
        train_transform = transforms.Compose([
            transforms.Resize((int(image_size * 1.2), int(image_size * 1.2))),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_model(num_classes, freeze_backbone=False):
    """Create ResNet18 model with optional backbone freezing"""
    
    print("\n" + "="*60)
    print("CREATING MODEL")
    print("="*60)
    
    # Use new API to avoid warnings
    from torchvision.models import ResNet18_Weights
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    
    # Freeze backbone if requested (faster training, less overfitting)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        print("   ✓ Backbone frozen (only training classifier)")
    
    # Replace final layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),  # Slightly higher dropout for small dataset
        nn.Linear(num_features, num_classes)
    )
    
    # Unfreeze classifier
    for param in model.fc.parameters():
        param.requires_grad = True
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"   ✓ Model: ResNet18 (pretrained on ImageNet)")
    print(f"   ✓ Final layer: Linear({num_features} → {num_classes})")
    print(f"   ✓ Total parameters: {total_params:,}")
    print(f"   ✓ Trainable parameters: {trainable_params:,}")
    print(f"   ✓ Frozen: {freeze_backbone}")
    
    return model

def calculate_sample_weights(labels):
    """Calculate sample weights for balanced sampling"""
    from collections import Counter
    label_counts = Counter(labels)
    
    # Weight inversely proportional to class frequency
    weights = [1.0 / label_counts[label] for label in labels]
    
    # Normalize
    weights = np.array(weights)
    weights = weights / weights.sum() * len(weights)
    
    return torch.DoubleTensor(weights)

def train_epoch(model, loader, criterion, optimizer, device, grad_clip=1.0):
    """Train for one epoch"""
    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training', leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
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
    
    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-o', label='Val', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(epochs, history['train_acc'], 'b-o', label='Train', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-o', label='Val', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Learning rate
    ax3.plot(epochs, history['learning_rates'], 'g-o', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Learning Rate', fontsize=12)
    ax3.set_title('Learning Rate', fontsize=14, fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Overfitting indicator
    loss_diff = [val - train for train, val in zip(history['train_loss'], history['val_loss'])]
    ax4.plot(epochs, loss_diff, 'purple', linewidth=2)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Val Loss - Train Loss', fontsize=12)
    ax4.set_title('Overfitting Check', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to {save_path}")
    plt.close()

def main():
    """Main training function"""
    
    # ========== OPTIMIZED HYPERPARAMETERS FOR SMALL DATASET ==========
    BATCH_SIZE = 16  # Smaller for CPU and small dataset
    NUM_EPOCHS = 50  # More epochs for small dataset
    LEARNING_RATE = 0.0001  # Lower LR for stability
    IMAGE_SIZE = 224
    GRAD_CLIP = 0.5  # Tighter gradient clipping
    PATIENCE = 15  # More patience
    MIN_SAMPLES_PER_CLASS = 10  # Minimum samples to keep a class
    FREEZE_BACKBONE = True  # Freeze for first phase
    AUGMENT_LEVEL = 'heavy'  # Heavy augmentation for small data
    
    import sys
    if len(sys.argv) > 1:
        DATA_DIR = sys.argv[1]
    else:
        possible_paths = ['pokemon', 'archive', '.']
        DATA_DIR = None
        for path in possible_paths:
            if os.path.exists(os.path.join(path, 'pokemon.csv')):
                DATA_DIR = path
                break
        
        if DATA_DIR is None:
            print("❌ Dataset not found!")
            sys.exit(1)
    
    print("\n" + "="*60)
    print("POKEMON TYPE CLASSIFIER - OPTIMIZED FOR SMALL DATASETS")
    print("="*60)
    print(f"Device: {device}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Augmentation: {AUGMENT_LEVEL}")
    print(f"Freeze backbone: {FREEZE_BACKBONE}")
    
    # Load dataset (removes classes with too few samples)
    image_paths, labels, label_to_idx, idx_to_label = load_dataset(
        DATA_DIR, min_samples_per_class=MIN_SAMPLES_PER_CLASS
    )
    
    num_classes = len(label_to_idx)
    
    # Split with stratification
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\n📊 Dataset split:")
    print(f"   Training: {len(train_paths)}")
    print(f"   Validation: {len(val_paths)}")
    
    # Get transforms
    train_transform, val_transform = get_transforms(IMAGE_SIZE, AUGMENT_LEVEL)
    
    # Create datasets
    train_dataset = PokemonDataset(train_paths, train_labels, train_transform)
    val_dataset = PokemonDataset(val_paths, val_labels, val_transform)
    
    # CRITICAL: Use WeightedRandomSampler for class balance
    sample_weights = calculate_sample_weights(train_labels)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,  # Use weighted sampler instead of shuffle
        num_workers=0,  # 0 for CPU
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Create model
    model = create_model(num_classes, freeze_backbone=FREEZE_BACKBONE).to(device)
    
    # Loss (no class weights since we're using weighted sampling)
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer - AdamW with weight decay
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )
    
    # Scheduler - ReduceLROnPlateau (reduces LR when validation plateaus)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
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
        val_loss, val_acc, _, _ = validate(
            model, val_loader, criterion, device
        )
        
        # Update history
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(current_lr)
        
        # Print results
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"LR: {current_lr:.6f}")
        
        # Step scheduler
        scheduler.step(val_acc)
        
        # Check if LR changed
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != current_lr:
            print(f"📉 LR reduced: {current_lr:.6f} → {new_lr:.6f}")
        
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
                'idx_to_label': idx_to_label
            }, 'best_pokemon_model.pth')
            print(f"✓ Best model saved! Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"⏳ Patience: {patience_counter}/{PATIENCE}")
            
            if patience_counter >= PATIENCE:
                print(f"\n⚠️  Early stopping at epoch {epoch+1}")
                break
        
        print()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"✓ Best validation accuracy: {best_val_acc:.2f}%")
    print(f"✓ Epochs trained: {len(history['train_acc'])}")
    
    # Plot
    plot_training_history(history)
    
    # Save mappings
    with open('label_mappings.json', 'w') as f:
        json.dump({
            'label_to_idx': label_to_idx,
            'idx_to_label': idx_to_label
        }, f, indent=2)
    
    print(f"✓ Files saved")
    print("\n🎉 Training complete!")

if __name__ == '__main__':
    main()