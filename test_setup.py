"""
Test script untuk memverifikasi instalasi dan model
Jalankan script ini sebelum menjalankan aplikasi Streamlit
"""

import sys

def test_imports():
    """Test import semua dependencies"""
    print("=" * 50)
    print("Testing Dependencies...")
    print("=" * 50)
    
    required_packages = [
        ("streamlit", "Streamlit"),
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
    ]
    
    failed = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✅ {name} installed")
        except ImportError:
            print(f"❌ {name} NOT installed")
            failed.append(name)
    
    try:
        from indobenchmark import IndoNLGTokenizer
        print("✅ IndobenchmarkToolkit installed")
    except ImportError:
        print("❌ IndobenchmarkToolkit NOT installed")
        failed.append("IndobenchmarkToolkit")
    
    if failed:
        print(f"\n❌ Missing packages: {', '.join(failed)}")
        print("\nRun: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All dependencies installed!")
        return True

def test_cuda():
    """Test CUDA availability"""
    print("\n" + "=" * 50)
    print("Testing CUDA...")
    print("=" * 50)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available")
            print(f"   Device: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("⚠️  CUDA not available - will use CPU")
            print("   (This is fine, but inference will be slower)")
    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")

def test_model_path():
    """Test model path"""
    print("\n" + "=" * 50)
    print("Testing Model Path...")
    print("=" * 50)
    
    import os
    from config import MODEL_PATH
    
    print(f"Model path from config: {MODEL_PATH}")
    
    if os.path.exists(MODEL_PATH):
        print(f"✅ Model directory exists")
        
        # Check for required files
        required_files = ["config.json", "pytorch_model.bin"]
        has_safetensors = os.path.exists(os.path.join(MODEL_PATH, "model.safetensors"))
        has_pytorch = os.path.exists(os.path.join(MODEL_PATH, "pytorch_model.bin"))
        has_config = os.path.exists(os.path.join(MODEL_PATH, "config.json"))
        
        if has_config:
            print("✅ config.json found")
        else:
            print("❌ config.json NOT found")
        
        if has_safetensors or has_pytorch:
            print(f"✅ Model weights found ({'safetensors' if has_safetensors else 'pytorch_model.bin'})")
        else:
            print("❌ Model weights NOT found")
            
        return has_config and (has_safetensors or has_pytorch)
    else:
        print(f"❌ Model directory NOT found: {MODEL_PATH}")
        print("\nTips:")
        print("1. Update MODEL_PATH in config.py")
        print("2. Ensure model is downloaded from Google Drive")
        print("3. Check if path is correct")
        return False

def test_model_loading():
    """Test loading model"""
    print("\n" + "=" * 50)
    print("Testing Model Loading...")
    print("=" * 50)
    
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM
        from indobenchmark import IndoNLGTokenizer
        from config import MODEL_PATH
        import os
        
        if not os.path.exists(MODEL_PATH):
            print("⚠️  Skipping - model path doesn't exist")
            return False
        
        print("Loading tokenizer...")
        tokenizer = IndoNLGTokenizer.from_pretrained("indobenchmark/indobart-v2")
        print("✅ Tokenizer loaded")
        
        print("Loading model (this may take a while)...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)
        model.eval()
        print(f"✅ Model loaded on {device}")
        
        # Test inference
        print("Testing inference...")
        test_text = "Ini adalah teks percobaan untuk menguji model."
        inputs = tokenizer(test_text, return_tensors="pt", max_length=100, truncation=True).to(device)
        
        with torch.no_grad():
            output = model.generate(
                inputs["input_ids"],
                max_new_tokens=50,
                num_beams=2,
            )
        
        result = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"✅ Inference successful")
        print(f"   Test output: {result[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n")
    print("🔍 INDONESIAN TEXT SUMMARIZER - SETUP TEST")
    print("=" * 50)
    print()
    
    # Test 1: Dependencies
    deps_ok = test_imports()
    
    if not deps_ok:
        print("\n❌ Please install dependencies first!")
        sys.exit(1)
    
    # Test 2: CUDA
    test_cuda()
    
    # Test 3: Model Path
    model_path_ok = test_model_path()
    
    # Test 4: Model Loading
    if model_path_ok:
        model_ok = test_model_loading()
    else:
        print("\n⚠️  Skipping model loading test - fix model path first")
        model_ok = False
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    if deps_ok and model_path_ok and model_ok:
        print("✅ All tests passed!")
        print("\n🚀 You can now run the Streamlit app:")
        print("   streamlit run streamlit_app.py")
        print("   or double-click run_app.bat")
    elif deps_ok and model_path_ok:
        print("⚠️  Dependencies and model path OK, but model loading failed")
        print("   Check the error messages above")
    elif deps_ok:
        print("⚠️  Dependencies OK, but model path needs configuration")
        print("   Update MODEL_PATH in config.py")
    else:
        print("❌ Setup incomplete")
        print("   Fix the issues above before running the app")
    
    print()

if __name__ == "__main__":
    main()
