#!/usr/bin/env python3
"""
Enhanced Model setup script for multiple deepfake detection models
Downloads and validates multiple Hugging Face models
Windows-compatible version
"""

import os
import sys
import io
import requests
import json
from datetime import datetime

# Fix Windows console encoding issues
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Recommended models for deepfake detection
RECOMMENDED_MODELS = [
    'Wvolf/ViT_Deepfake_Detection',
    'dima806/deepfake_vs_real_image_detection',
    'rizvandwiki/gender-classification-2',
    'microsoft/DiT-base',
    'nateraw/vit-base-patch16-224-face-detection'
]

def validate_huggingface_model(model_name):
    """Validate if a Hugging Face model exists and get its info"""
    try:
        # Clean model name
        clean_name = model_name.replace('https://huggingface.co/', '').strip('/')
        
        print(f"[INFO] Validating model: {clean_name}")
        
        # Check if model exists
        response = requests.head(f"https://huggingface.co/{clean_name}", timeout=10)
        
        if response.status_code != 200:
            return False, f"Model not found (HTTP {response.status_code})"
        
        # Get model info
        try:
            info_response = requests.get(f"https://huggingface.co/api/models/{clean_name}", timeout=10)
            if info_response.status_code == 200:
                model_info = info_response.json()
                return True, {
                    'name': clean_name,
                    'downloads': model_info.get('downloads', 0),
                    'likes': model_info.get('likes', 0),
                    'library': model_info.get('library_name', 'transformers'),
                    'pipeline_tag': model_info.get('pipeline_tag', 'unknown'),
                    'tags': model_info.get('tags', [])
                }
            else:
                return True, {'name': clean_name, 'downloads': 0, 'likes': 0}
        except:
            return True, {'name': clean_name, 'downloads': 0, 'likes': 0}
            
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def setup_model(model_name):
    """Download and cache a specific model"""
    try:
        from transformers import (
            AutoModelForImageClassification, 
            AutoImageProcessor,
            ViTForImageClassification, 
            ViTImageProcessor
        )
        import torch
        
        print(f"\n[INFO] Setting up model: {model_name}")
        
        # Validate model first
        is_valid, info = validate_huggingface_model(model_name)
        if not is_valid:
            print(f"[ERROR] {info}")
            return False
        
        clean_name = info['name'] if isinstance(info, dict) else model_name
        
        print(f"[INFO] Downloading processor for {clean_name}...")
        
        # Try Auto classes first
        try:
            processor = AutoImageProcessor.from_pretrained(clean_name)
            model = AutoModelForImageClassification.from_pretrained(clean_name)
            print("[INFO] Successfully loaded with Auto classes")
        except Exception as e:
            print(f"[INFO] Auto classes failed, trying ViT-specific: {e}")
            # Fallback to ViT classes
            try:
                processor = ViTImageProcessor.from_pretrained(clean_name)
                model = ViTForImageClassification.from_pretrained(clean_name)
                print("[INFO] Successfully loaded with ViT classes")
            except Exception as e2:
                print(f"[ERROR] Both loading methods failed: {e2}")
                return False
        
        # Test the model
        print("[INFO] Testing model...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        model.eval()
        
        print(f"[INFO] Model loaded on: {device}")
        print(f"[INFO] Parameters: {model.num_parameters():,}")
        
        if hasattr(model.config, 'id2label'):
            print(f"[INFO] Labels: {model.config.id2label}")
        
        # Test inference
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            outputs = model(dummy_input)
            print(f"[INFO] Output shape: {outputs.logits.shape}")
        
        if isinstance(info, dict):
            print(f"[INFO] Downloads: {info.get('downloads', 'Unknown')}")
            print(f"[INFO] Library: {info.get('library', 'transformers')}")
        
        print(f"[SUCCESS] Model {clean_name} setup completed!")
        return True
        
    except ImportError as e:
        print(f"[ERROR] Missing dependencies: {e}")
        print("[INFO] Please install: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"[ERROR] Setup failed for {model_name}: {e}")
        return False

def setup_multiple_models(model_list=None):
    """Setup multiple models"""
    if model_list is None:
        model_list = RECOMMENDED_MODELS
    
    print("=" * 60)
    print("ENHANCED DEEPFAKE DETECTOR - MULTI-MODEL SETUP")
    print("=" * 60)
    
    successful = []
    failed = []
    
    for i, model_name in enumerate(model_list, 1):
        print(f"\n[{i}/{len(model_list)}] Processing: {model_name}")
        
        if setup_model(model_name):
            successful.append(model_name)
        else:
            failed.append(model_name)
    
    print("\n" + "=" * 60)
    print("SETUP SUMMARY")
    print("=" * 60)
    print(f"✅ Successful: {len(successful)}")
    for model in successful:
        print(f"   - {model}")
    
    if failed:
        print(f"\n❌ Failed: {len(failed)}")
        for model in failed:
            print(f"   - {model}")
    
    print(f"\nCache location: {os.path.expanduser('~/.cache/huggingface/')}")
    
    return len(successful) > 0

def interactive_setup():
    """Interactive setup mode"""
    print("Enhanced Deepfake Detector - Interactive Model Setup")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Setup recommended models")
        print("2. Add custom model")
        print("3. List recommended models")
        print("4. Validate a model")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            setup_multiple_models(RECOMMENDED_MODELS)
            break
        elif choice == '2':
            model_name = input("Enter Hugging Face model name: ").strip()
            if model_name:
                setup_model(model_name)
        elif choice == '3':
            print("\nRecommended models:")
            for i, model in enumerate(RECOMMENDED_MODELS, 1):
                print(f"  {i}. {model}")
        elif choice == '4':
            model_name = input("Enter model name to validate: ").strip()
            if model_name:
                is_valid, info = validate_huggingface_model(model_name)
                if is_valid:
                    print(f"✅ Valid model: {info}")
                else:
                    print(f"❌ Invalid: {info}")
        elif choice == '5':
            break
        else:
            print("Invalid choice. Please try again.")

def main():
    """Main setup function"""
    if len(sys.argv) > 1:
        if sys.argv[1] == '--interactive':
            interactive_setup()
        elif sys.argv[1] == '--model':
            if len(sys.argv) > 2:
                setup_model(sys.argv[2])
            else:
                print("Please specify a model name")
        else:
            print("Usage: python model_setup.py [--interactive] [--model MODEL_NAME]")
    else:
        # Default: setup recommended models
        success = setup_multiple_models()
        if success:
            print("\n[SUCCESS] Multi-model setup completed!")
            print("[INFO] You can now use the enhanced deepfake detector")
            sys.exit(0)
        else:
            print("\n[ERROR] Setup failed!")
            sys.exit(1)

if __name__ == "__main__":
    main()