"""
Script untuk upload model ke HuggingFace Hub
Jalankan sekali saja untuk upload model Anda
"""

from huggingface_hub import HfApi, create_repo
import os

# ============================================================
# KONFIGURASI - UBAH INI
# ============================================================
HUGGINGFACE_USERNAME = "your_username"  # Ganti dengan username HF Anda
MODEL_NAME = "indobart-v2-detik-summary"  # Nama model di HF Hub
MODEL_LOCAL_PATH = "./models/indobart-v2-detik-final-20251201-061311"
MAKE_PRIVATE = False  # True jika mau private model

# ============================================================

def upload_model():
    """Upload model ke HuggingFace Hub"""
    
    # Check jika model ada
    if not os.path.exists(MODEL_LOCAL_PATH):
        print(f"❌ Model tidak ditemukan di: {MODEL_LOCAL_PATH}")
        return
    
    # Repo ID format: username/model-name
    repo_id = f"{HUGGINGFACE_USERNAME}/{MODEL_NAME}"
    
    print(f"📤 Uploading model ke: https://huggingface.co/{repo_id}")
    print(f"{'🔒 Private' if MAKE_PRIVATE else '🌍 Public'} repository")
    
    try:
        # Login check
        api = HfApi()
        user = api.whoami()
        print(f"✅ Logged in as: {user['name']}")
        
        # Create repository
        print(f"\n📦 Creating repository...")
        create_repo(
            repo_id=repo_id,
            private=MAKE_PRIVATE,
            exist_ok=True
        )
        print(f"✅ Repository created/exists")
        
        # Upload files
        print(f"\n📤 Uploading files... (ini mungkin butuh beberapa menit)")
        api.upload_folder(
            folder_path=MODEL_LOCAL_PATH,
            repo_id=repo_id,
            repo_type="model",
        )
        
        print(f"\n✅ BERHASIL!")
        print(f"🔗 Model URL: https://huggingface.co/{repo_id}")
        print(f"\n📝 Update config.py dengan:")
        print(f'   MODEL_PATH = "{repo_id}"')
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Pastikan Anda sudah login:")
        print("   huggingface-cli login")

if __name__ == "__main__":
    print("=" * 60)
    print("🤗 Upload Model ke HuggingFace Hub")
    print("=" * 60)
    
    # Validasi konfigurasi
    if HUGGINGFACE_USERNAME == "your_username":
        print("\n❌ ERROR: Ubah dulu HUGGINGFACE_USERNAME di script ini!")
        print("   1. Daftar di https://huggingface.co (gratis)")
        print("   2. Ubah HUGGINGFACE_USERNAME dengan username Anda")
        print("   3. Login: huggingface-cli login")
        exit(1)
    
    upload_model()
