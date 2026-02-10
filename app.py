#!/usr/bin/env python3
"""
Pokemon Type Classifier - Web Interface
Automatically detects and loads the correct model architecture
"""

from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import base64
import json
import os

app = Flask(__name__)

# ========== MODEL DEFINITIONS ==========

class PokemonCNN(nn.Module):
    """Original Custom CNN Architecture"""
    
    def __init__(self, num_classes):
        super(PokemonCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

def create_resnet_model(num_classes):
    """ResNet18 Transfer Learning Model (Fixed Version)"""
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_features, num_classes)
    )
    return model

# ========== LOAD MODEL ==========

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load label mappings
with open('label_mappings.json', 'r') as f:
    mappings = json.load(f)
    idx_to_label = {int(k): v for k, v in mappings['idx_to_label'].items()}

# Load checkpoint
checkpoint = torch.load('best_pokemon_model.pth', map_location=device)
num_classes = len(idx_to_label)

# Auto-detect model architecture
def detect_model_architecture(state_dict):
    """Detect whether the model is custom CNN or ResNet"""
    keys = list(state_dict.keys())
    
    # Check for ResNet keys
    if any('layer1' in key or 'layer2' in key for key in keys):
        return 'resnet18'
    # Check for custom CNN keys
    elif any('conv_layers' in key for key in keys):
        return 'custom_cnn'
    else:
        return 'unknown'

model_type = detect_model_architecture(checkpoint['model_state_dict'])

if model_type == 'resnet18':
    print("✓ Detected ResNet18 model (Transfer Learning)")
    model = create_resnet_model(num_classes).to(device)
    image_size = 224
elif model_type == 'custom_cnn':
    print("✓ Detected Custom CNN model")
    model = PokemonCNN(num_classes).to(device)
    image_size = 128
else:
    raise ValueError("Unknown model architecture in checkpoint!")

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✓ Model loaded successfully")
print(f"✓ Model type: {model_type}")
print(f"✓ Validation accuracy: {checkpoint['val_acc']:.2f}%")
print(f"✓ Number of classes: {num_classes}")
print(f"✓ Image size: {image_size}x{image_size}")

# Image transformation - uses detected image size
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Pokemon type colors
TYPE_COLORS = {
    'Normal': '#A8A878',
    'Fire': '#F08030',
    'Water': '#6890F0',
    'Electric': '#F8D030',
    'Grass': '#78C850',
    'Ice': '#98D8D8',
    'Fighting': '#C03028',
    'Poison': '#A040A0',
    'Ground': '#E0C068',
    'Flying': '#A890F0',
    'Psychic': '#F85888',
    'Bug': '#A8B820',
    'Rock': '#B8A038',
    'Ghost': '#705898',
    'Dragon': '#7038F8',
    'Dark': '#705848',
    'Steel': '#B8B8D0',
    'Fairy': '#EE99AC'
}

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html', 
                         val_accuracy=f"{checkpoint['val_acc']:.2f}",
                         type_colors=TYPE_COLORS)

@app.route('/predict', methods=['POST'])
def predict():
    """Predict Pokemon type from uploaded image"""
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        # Read and process image
        image = Image.open(file.stream).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            probs, indices = torch.topk(probabilities, k=min(5, num_classes))
        
        # Prepare results
        predictions = []
        for prob, idx in zip(probs[0], indices[0]):
            type_name = idx_to_label[idx.item()]
            predictions.append({
                'type': type_name,
                'probability': float(prob.item()) * 100,
                'color': TYPE_COLORS.get(type_name, '#CCCCCC')
            })
        
        return jsonify({'predictions': predictions})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎮 POKEMON TYPE CLASSIFIER - WEB INTERFACE")
    print("="*60)
    print(f"Server starting on http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)