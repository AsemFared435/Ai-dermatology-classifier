"""
Test script to verify the AI Dermatology Classifier setup
Run this before launching the Streamlit app to ensure everything works
"""

import sys

def test_imports():
    """Test if all required packages are installed"""
    print("=" * 60)
    print("Testing Package Imports")
    print("=" * 60)
    
    packages = {
        'streamlit': 'Streamlit',
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'plotly': 'Plotly'
    }
    
    all_ok = True
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✓ {name:15} - OK")
        except ImportError:
            print(f"✗ {name:15} - MISSING")
            all_ok = False
    
    return all_ok

def test_model_file():
    """Test if model file exists"""
    print("\n" + "=" * 60)
    print("Testing Model File")
    print("=" * 60)
    
    import os
    
    model_file = 'efficientnet_best.pth'
    
    if os.path.exists(model_file):
        size_mb = os.path.getsize(model_file) / (1024 * 1024)
        print(f"✓ Model file found: {model_file}")
        print(f"  Size: {size_mb:.2f} MB")
        return True
    else:
        print(f"✗ Model file not found: {model_file}")
        print("  Please ensure the model file is in the current directory")
        return False

def test_model_loading():
    """Test if model can be loaded"""
    print("\n" + "=" * 60)
    print("Testing Model Loading")
    print("=" * 60)
    
    try:
        import torch
        from torchvision import models
        import torch.nn as nn
        
        print("Loading model architecture...")
        model = models.efficientnet_b0(pretrained=False)
        
        # Modify classifier
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 3)
        
        print("Loading model weights...")
        state_dict = torch.load('efficientnet_best.pth', map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        
        print("✓ Model loaded successfully!")
        print(f"  Number of classes: 3")
        print(f"  Model type: EfficientNet-B0")
        return True
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False

def test_inference():
    """Test if inference works"""
    print("\n" + "=" * 60)
    print("Testing Model Inference")
    print("=" * 60)
    
    try:
        import torch
        from torchvision import models
        import torch.nn as nn
        from PIL import Image
        import torchvision.transforms as transforms
        import numpy as np
        
        # Load model
        model = models.efficientnet_b0(pretrained=False)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 3)
        state_dict = torch.load('efficientnet_best.pth', map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        
        # Create dummy image
        dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        # Preprocess
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(dummy_img).unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        print("✓ Inference test successful!")
        print(f"  Test prediction class: {predicted.item()}")
        print(f"  Test confidence: {confidence.item() * 100:.2f}%")
        return True
        
    except Exception as e:
        print(f"✗ Error during inference: {e}")
        return False

def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "AI DERMATOLOGY CLASSIFIER TEST SUITE" + " " * 10 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    results = []
    
    # Run tests
    results.append(("Package Imports", test_imports()))
    results.append(("Model File", test_model_file()))
    
    # Only test model loading if file exists
    if results[-1][1]:
        results.append(("Model Loading", test_model_loading()))
        
        # Only test inference if model loads
        if results[-1][1]:
            results.append(("Model Inference", test_inference()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {test_name:20} - {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 All tests passed! You're ready to run the app!")
        print("\nRun the app with:")
        print("  streamlit run dermatology_app.py")
        print("\nOr use the automated script:")
        print("  ./run_app.sh")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above before running the app.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
