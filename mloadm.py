import os
import ssl
import torch
import sys
from pathlib import Path

def fix_ssl_certificates():
    """Fix SSL certificate issues on macOS"""
    try:
        # Method 1: Create unverified context
        ssl._create_default_https_context = ssl._create_unverified_context
        
        # Method 2: For macOS, try to install certificates
        import certifi
        os.environ['SSL_CERT_FILE'] = certifi.where()
        os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
        
        print("✅ SSL certificates configured")
    except Exception as e:
        print(f"⚠️ SSL fix warning: {e}")

def clear_torch_cache():
    """Clear corrupted torch hub cache"""
    try:
        cache_dir = Path.home() / '.cache' / 'torch' / 'hub'
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
            print("✅ Cleared torch hub cache")
            return True
    except Exception as e:
        print(f"⚠️ Cache clear warning: {e}")
    return False

def load_model_direct_pytorch(model_path):
    """Load model directly with PyTorch (fallback method)"""
    try:
        print("🔧 Attempting direct PyTorch model loading...")
        
        # Load the model checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Extract model structure if it's a full YOLOv5 checkpoint
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model = checkpoint['model']
            if hasattr(model, 'float'):
                model = model.float()
        else:
            model = checkpoint
        
        # Set to evaluation mode
        model.eval()
        
        # Add required attributes for YOLOv5 compatibility
        if not hasattr(model, 'names'):
            # Default class names - update these to match your model
            model.names = {0: 'alpha', 1: 'beta', 2: 'muon', 3: 'proton', 4: 'electron'}
        
        print("✅ Model loaded with direct PyTorch")
        return model, 'pytorch'
        
    except Exception as e:
        print(f"❌ Direct PyTorch loading failed: {e}")
        return None, None

def load_model_with_fixes(model_path):
    """Load YOLOv5 model with all compatibility fixes"""
    
    # Step 1: Fix SSL certificates
    fix_ssl_certificates()
    
    # Step 2: Try torch.hub with fixed SSL
    try:
        print("🔧 Trying torch.hub with SSL fix...")
        model = torch.hub.load('ultralytics/yolov5', 'custom', 
                             path=model_path, force_reload=True, trust_repo=True)
        print("✅ Model loaded with torch.hub + SSL fix")
        return model, 'torch_hub'
    except Exception as e:
        print(f"❌ torch.hub with SSL fix failed: {e}")
    
    # Step 3: Clear cache and try again
    if clear_torch_cache():
        try:
            print("🔧 Trying torch.hub after cache clear...")
            model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                 path=model_path, force_reload=True, trust_repo=True)
            print("✅ Model loaded after cache clear")
            return model, 'torch_hub'
        except Exception as e:
            print(f"❌ torch.hub after cache clear failed: {e}")
    
    # Step 4: Try direct PyTorch loading
    model, model_type = load_model_direct_pytorch(model_path)
    if model is not None:
        return model, model_type
    
    # Step 5: Clone YOLOv5 locally and try
    try:
        print("🔧 Attempting to clone YOLOv5 locally...")
        import subprocess
        import git
        
        yolov5_dir = "./yolov5"
        if not os.path.exists(yolov5_dir):
            git.Repo.clone_from("https://github.com/ultralytics/yolov5.git", yolov5_dir)
            print("✅ YOLOv5 cloned successfully")
        
        # Add to Python path
        sys.path.insert(0, yolov5_dir)
        
        # Try loading from local repo
        model = torch.hub.load(yolov5_dir, 'custom', 
                             path=model_path, source='local', trust_repo=True)
        print("✅ Model loaded from local YOLOv5 repo")
        return model, 'local_repo'
        
    except Exception as e:
        print(f"❌ Local YOLOv5 repo failed: {e}")
    
    return None, None

# Test the loading
if __name__ == "__main__":
    model_path = "best.pt"  # Update this to your model path
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        sys.exit(1)
    
    model, model_type = load_model_with_fixes(model_path)
    
    if model is not None:
        print(f"🎉 SUCCESS! Model loaded using method: {model_type}")
        print(f"📊 Model classes: {getattr(model, 'names', 'Unknown')}")
    else:
        print("❌ All loading methods failed")
        print("\n💡 Manual solutions:")
        print("1. Install certificates: /Applications/Python\\ 3.x/Install\\ Certificates.command")
        print("2. pip install certifi")
        print("3. Download YOLOv5 manually: git clone https://github.com/ultralytics/yolov5.git")