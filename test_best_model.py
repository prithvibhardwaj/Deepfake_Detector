#!/usr/bin/env python3
"""
Test script for the best Hugging Face deepfake detection model
Tests Wvolf/ViT_Deepfake_Detection integration
Windows-compatible version
"""

import os
import sys
import json
import tempfile
import subprocess
from PIL import Image
import numpy as np
import io

# Fix Windows console encoding issues
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def create_test_image():
    """Create a simple test image"""
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    img.save(temp_file.name)
    temp_file.close()
    
    return temp_file.name

def test_model_download():
    """Test downloading the best Hugging Face model"""
    print("[INFO] Testing best model download...")
    
    try:
        result = subprocess.run([
            sys.executable, 'model_setup.py'
        ], capture_output=True, text=True, timeout=600)  # 10 minute timeout for larger model
        
        print("[INFO] Model download output:")
        print(f"STDOUT: {result.stdout}")
        if result.stderr:
            print(f"STDERR: {result.stderr}")
        
        if result.returncode == 0:
            print("[SUCCESS] Best model download successful!")
            return True
        else:
            print("[ERROR] Best model download failed!")
            print(f"[ERROR] Return code: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("[ERROR] Model download timed out (10 minutes)")
        return False
    except Exception as e:
        print(f"[ERROR] Error testing model download: {e}")
        return False

def test_inference():
    """Test the inference script with the best model"""
    print("[INFO] Creating test image...")
    test_image_path = create_test_image()
    
    print(f"[INFO] Test image created: {test_image_path}")
    
    try:
        print("[INFO] Running inference with best Hugging Face model...")
        result = subprocess.run([
            sys.executable, 'inference.py', test_image_path, 'image'
        ], capture_output=True, text=True, timeout=180)  # 3 minute timeout
        
        print("[INFO] Raw output:")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        print(f"Return code: {result.returncode}")
        
        if result.returncode == 0:
            try:
                # Parse JSON output
                lines = result.stdout.strip().split('\n')
                json_line = None
                
                for line in lines:
                    if line.strip().startswith('{'):
                        json_line = line.strip()
                        break
                
                if json_line:
                    output = json.loads(json_line)
                    print("[SUCCESS] Inference result:")
                    print(f"   Prediction: {output.get('prediction', 'Unknown')}")
                    print(f"   Confidence: {output.get('confidence', 0):.3f}")
                    print(f"   Model: {output.get('model', 'Unknown')}")
                    print(f"   Status: {output.get('status', 'Unknown')}")
                    return True
                else:
                    print("[ERROR] No JSON output found")
                    return False
                    
            except json.JSONDecodeError as e:
                print(f"[ERROR] Could not parse JSON output: {e}")
                return False
        else:
            print("[ERROR] Inference failed")
            return False
            
    except subprocess.TimeoutExpired:
        print("[ERROR] Inference timed out (3 minutes)")
        return False
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
            print("[INFO] Cleaned up test image")

def manual_test():
    """Manual test of model components"""
    print("[INFO] Running manual component test...")
    
    try:
        from transformers import ViTForImageClassification, ViTImageProcessor
        import torch
        
        model_name = 'Wvolf/ViT_Deepfake_Detection'
        
        print(f"[INFO] Testing {model_name} components...")
        
        # Test processor
        processor = ViTImageProcessor.from_pretrained(model_name)
        print("[SUCCESS] Processor loaded successfully")
        
        # Test model
        model = ViTForImageClassification.from_pretrained(model_name)
        print("[SUCCESS] Model loaded successfully")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        model.eval()
        
        print(f"[INFO] Device: {device}")
        print(f"[INFO] Parameters: {model.num_parameters():,}")
        
        # Test inference
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            outputs = model(dummy_input)
            print(f"[SUCCESS] Test inference completed")
            print(f"[INFO] Output shape: {outputs.logits.shape}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Manual test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("TESTING BEST HUGGING FACE DEEPFAKE DETECTION MODEL")
    print("Model: Wvolf/ViT_Deepfake_Detection (98.70% accuracy)")
    print("=" * 60)
    
    # Test 1: Manual component test
    print("\n[TEST 1] Manual component test")
    manual_success = manual_test()
    
    if not manual_success:
        print("[INFO] Trying full download test...")
        
        # Test 2: Model download
        print("\n[TEST 2] Model download test")
        download_success = test_model_download()
        
        if download_success:
            # Test 3: Inference
            print("\n[TEST 3] Inference test")
            inference_success = test_inference()
            
            if inference_success:
                print("\n[SUCCESS] All tests passed! The best model integration is working correctly.")
            else:
                print("\n[WARNING] Model downloaded but inference failed. Check the inference script.")
        else:
            print("\n[ERROR] Model download failed. Check your internet connection.")
    else:
        print("\n[SUCCESS] Manual test passed! Running inference test...")
        
        # Test inference directly
        print("\n[TEST 2] Inference test")
        inference_success = test_inference()
        
        if inference_success:
            print("\n[SUCCESS] All tests passed! The best model integration is working correctly.")
        else:
            print("\n[WARNING] Model works but inference script has issues.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()