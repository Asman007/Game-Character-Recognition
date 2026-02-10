#!/usr/bin/env python3
"""
Diagnostic Script - Find out why training is failing
"""

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
from collections import Counter
import json

print("="*70)
print("POKEMON CLASSIFIER DIAGNOSTIC TOOL")
print("="*70)

# 1. Check PyTorch installation
print("\n1. PYTORCH ENVIRONMENT:")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
print(f"   Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

# 2. Check dataset
print("\n2. DATASET ANALYSIS:")
csv_path = 'pokemon/pokemon.csv'
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print(f"   ✓ CSV found: {len(df)} entries")
    print(f"   Columns: {list(df.columns)}")
    
    # Check Type1 distribution
    type_counts = Counter(df['Type1'])
    print(f"\n   Type distribution:")
    for ptype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"      {ptype}: {count}")
    
    # Check for very small classes
    small_classes = [t for t, c in type_counts.items() if c < 20]
    if small_classes:
        print(f"\n   ⚠️  WARNING: {len(small_classes)} classes have <20 samples: {small_classes}")
else:
    print("   ❌ CSV not found!")

# 3. Check images
print("\n3. IMAGE VALIDATION:")
images_dir = 'pokemon/images'
if os.path.exists(images_dir):
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    print(f"   ✓ Images folder found: {len(image_files)} images")
    
    # Check first few images
    print(f"\n   Checking first 5 images:")
    for i, img_file in enumerate(image_files[:5]):
        img_path = os.path.join(images_dir, img_file)
        try:
            img = Image.open(img_path)
            print(f"      {img_file}: {img.size} {img.mode}")
        except Exception as e:
            print(f"      {img_file}: ❌ ERROR - {e}")
    
    # Check for inconsistent sizes
    sizes = []
    modes = []
    for img_file in image_files[:50]:  # Check first 50
        try:
            img = Image.open(os.path.join(images_dir, img_file))
            sizes.append(img.size)
            modes.append(img.mode)
        except:
            pass
    
    unique_sizes = set(sizes)
    unique_modes = set(modes)
    print(f"\n   Image sizes found: {unique_sizes}")
    print(f"   Image modes found: {unique_modes}")
    
    if len(unique_modes) > 1:
        print(f"   ⚠️  WARNING: Multiple image modes detected!")
else:
    print("   ❌ Images folder not found!")

# 4. Test pretrained model loading
print("\n4. PRETRAINED MODEL TEST:")
try:
    # Test with old API
    model_old = models.resnet18(pretrained=True)
    print("   ✓ ResNet18 loaded with pretrained=True (old API)")
    
    # Check if weights are actually loaded
    first_conv_weight = model_old.conv1.weight.data
    print(f"   First conv layer mean: {first_conv_weight.mean().item():.6f}")
    print(f"   First conv layer std: {first_conv_weight.std().item():.6f}")
    
    # These should be non-zero if pretrained weights loaded
    if abs(first_conv_weight.mean().item()) < 0.0001 and abs(first_conv_weight.std().item() - 1.0) < 0.01:
        print("   ⚠️  WARNING: Weights look like random initialization!")
    else:
        print("   ✓ Weights appear to be pretrained")
        
except Exception as e:
    print(f"   ❌ Error loading model: {e}")

# Test with new API
try:
    from torchvision.models import ResNet18_Weights
    model_new = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    print("   ✓ ResNet18 loaded with weights=IMAGENET1K_V1 (new API)")
except Exception as e:
    print(f"   ⚠️  New API not available: {e}")

# 5. Test data loading and preprocessing
print("\n5. DATA PREPROCESSING TEST:")
if os.path.exists(images_dir) and len(image_files) > 0:
    test_img_path = os.path.join(images_dir, image_files[0])
    
    try:
        # Load image
        img = Image.open(test_img_path).convert('RGB')
        print(f"   Original image: {img.size}")
        
        # Apply transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(img)
        print(f"   Transformed tensor shape: {img_tensor.shape}")
        print(f"   Tensor range: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
        print(f"   Tensor mean: {img_tensor.mean():.3f}")
        print(f"   Tensor std: {img_tensor.std():.3f}")
        
        # Test model prediction
        model = models.resnet18(pretrained=True)
        model.eval()
        with torch.no_grad():
            output = model(img_tensor.unsqueeze(0))
        
        print(f"   Model output shape: {output.shape}")
        print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        # Check if output is reasonable
        if output.abs().max() > 100:
            print("   ⚠️  WARNING: Extremely large logits detected!")
        else:
            print("   ✓ Output logits look reasonable")
            
    except Exception as e:
        print(f"   ❌ Error during preprocessing test: {e}")

# 6. Check label encoding
print("\n6. LABEL ENCODING CHECK:")
if os.path.exists('label_mappings.json'):
    with open('label_mappings.json', 'r') as f:
        mappings = json.load(f)
    
    print(f"   ✓ Label mappings found")
    print(f"   Number of classes: {len(mappings['idx_to_label'])}")
    print(f"   Classes: {list(mappings['label_to_idx'].keys())}")
    
    # Check for gaps in indices
    indices = sorted([int(k) for k in mappings['idx_to_label'].keys()])
    expected = list(range(len(indices)))
    if indices != expected:
        print(f"   ⚠️  WARNING: Label indices have gaps! Found: {indices}")
    else:
        print(f"   ✓ Label indices are continuous [0-{len(indices)-1}]")
else:
    print("   ⚠️  No label mappings file found")

# 7. Memory and compute resources
print("\n7. SYSTEM RESOURCES:")
print(f"   CPU cores: {os.cpu_count()}")

# Try to estimate memory usage
try:
    import psutil
    mem = psutil.virtual_memory()
    print(f"   RAM: {mem.total / (1024**3):.1f} GB total, {mem.available / (1024**3):.1f} GB available")
except:
    print("   RAM: Unable to check (psutil not installed)")

print("\n" + "="*70)
print("DIAGNOSTIC SUMMARY")
print("="*70)

# Provide recommendations
print("\nRECOMMENDATIONS:")

issues = []

# Check dataset size
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    if len(df) < 1000:
        issues.append("Dataset is very small (<1000 images)")
    
    type_counts = Counter(df['Type1'])
    small = [t for t, c in type_counts.items() if c < 20]
    if small:
        issues.append(f"{len(small)} classes have <20 samples (may cause instability)")

# Check GPU
if not torch.cuda.is_available():
    issues.append("No GPU available (training will be very slow)")

# Print issues
if issues:
    for i, issue in enumerate(issues, 1):
        print(f"{i}. ⚠️  {issue}")
else:
    print("✓ No major issues detected")

print("\n" + "="*70)