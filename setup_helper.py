#!/usr/bin/env python3
"""
Dataset Setup Helper
Checks if the Pokemon dataset is properly structured
"""

import os
import sys

def find_dataset():
    """Find the pokemon dataset"""
    
    print("🔍 Searching for Pokemon dataset...\n")
    
    # Check current directory
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}\n")
    
    # List all items in current directory
    print("Contents of current directory:")
    items = os.listdir('.')
    for item in sorted(items):
        if os.path.isdir(item):
            print(f"  📁 {item}/")
        else:
            print(f"  📄 {item}")
    
    print("\n" + "="*60)
    
    # Try to find pokemon.csv
    possible_paths = [
        'pokemon',
        'archive',
        'PokemonData',
        'pokemon-images-and-types',
        '.',
        '../pokemon',
        '../archive'
    ]
    
    found_datasets = []
    
    for path in possible_paths:
        csv_path = os.path.join(path, 'pokemon.csv')
        images_path = os.path.join(path, 'images')
        
        if os.path.exists(csv_path):
            has_images = os.path.exists(images_path)
            found_datasets.append({
                'path': path,
                'csv': csv_path,
                'has_images': has_images,
                'images_path': images_path
            })
    
    if not found_datasets:
        print("\n❌ No pokemon.csv found in common locations!")
        print("\n📋 Instructions:")
        print("1. Download the dataset from:")
        print("   https://www.kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types")
        print("\n2. Extract the zip file")
        print("\n3. Make sure you have this structure:")
        print("   pokemon/")
        print("   ├── pokemon.csv")
        print("   └── images/")
        print("       ├── bulbasaur.png")
        print("       ├── ivysaur.png")
        print("       └── ...")
        print("\n4. Either:")
        print("   a) Rename the extracted folder to 'pokemon'")
        print("   b) Run: python train_pokemon_classifier.py <folder_name>")
        return None
    
    print("\n✅ Found dataset(s):")
    for i, dataset in enumerate(found_datasets, 1):
        print(f"\n{i}. Location: {dataset['path']}")
        print(f"   CSV: {'✓' if os.path.exists(dataset['csv']) else '✗'} {dataset['csv']}")
        print(f"   Images folder: {'✓' if dataset['has_images'] else '✗'} {dataset['images_path']}")
        
        if dataset['has_images']:
            num_images = len([f for f in os.listdir(dataset['images_path']) 
                            if f.endswith('.png')])
            print(f"   Number of images: {num_images}")
    
    # Return the best match
    best_dataset = None
    for dataset in found_datasets:
        if dataset['has_images']:
            best_dataset = dataset
            break
    
    if not best_dataset:
        best_dataset = found_datasets[0]
    
    print("\n" + "="*60)
    print(f"\n🎯 Recommended dataset path: {best_dataset['path']}")
    
    if not best_dataset['has_images']:
        print("\n⚠️  Warning: Images folder not found!")
        print(f"   Please ensure there's an 'images/' folder in {best_dataset['path']}")
    
    print("\n📝 To train the model, run:")
    if best_dataset['path'] == '.':
        print("   python train_pokemon_classifier.py .")
    else:
        print(f"   python train_pokemon_classifier.py {best_dataset['path']}")
    
    return best_dataset['path']

if __name__ == '__main__':
    print("="*60)
    print("Pokemon Dataset Setup Helper")
    print("="*60)
    
    dataset_path = find_dataset()
    
    if dataset_path:
        print("\n✨ Setup check complete!")
    else:
        print("\n❌ Setup incomplete - please follow instructions above")
        sys.exit(1)