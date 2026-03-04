"""
Script untuk download model dari Google Drive
Untuk user yang ingin auto-download model
"""

import os
import sys

# ============================================================
# KONFIGURASI - UBAH INI SESUAI GOOGLE DRIVE ANDA
# ============================================================

# Cara mendapatkan FOLDER_ID dari Google Drive:
# 1. Upload folder model ke Google Drive
# 2. Klik kanan folder > Share > Anyone with the link (Viewer)
# 3. Copy link: https://drive.google.com/drive/folders/FOLDER_ID?usp=sharing
# 4. FOLDER_ID adalah bagian antara "/folders/" dan "?usp="

GOOGLE_DRIVE_FOLDER_ID = "your_folder_id_here"
MODEL_FOLDER_NAME = "indobart-v2-detik-final-20251201-061311"
OUTPUT_PATH = f"./models/{MODEL_FOLDER_NAME}"

# ============================================================

def download_model():
    """Download model dari Google Drive"""
    
    # Validasi konfigurasi
    if GOOGLE_DRIVE_FOLDER_ID == "your_folder_id_here":
        print("❌ ERROR: Ubah dulu GOOGLE_DRIVE_FOLDER_ID di script ini!")
        print("\n📝 Cara mendapatkan Folder ID:")
        print("   1. Upload model folder ke Google Drive")
        print("   2. Klik kanan folder > Share > Anyone with the link")
        print("   3. Copy link yang muncul")
        print("   4. Link format: https://drive.google.com/drive/folders/FOLDER_ID?usp=sharing")
        print("   5. Copy FOLDER_ID nya (bagian antara /folders/ dan ?usp=)")
        print("\n   Contoh link:")
        print("   https://drive.google.com/drive/folders/1a2b3c4d5e6f7g8h9i0j/view")
        print("   FOLDER_ID = 1a2b3c4d5e6f7g8h9i0j")
        return False
    
    # Check if gdown installed
    try:
        import gdown
    except ImportError:
        print("❌ ERROR: gdown belum terinstall!")
        print("\n💡 Install dulu:")
        print("   pip install gdown")
        return False
    
    # Create models directory
    os.makedirs("./models", exist_ok=True)
    
    print("=" * 60)
    print("📥 Download Model dari Google Drive")
    print("=" * 60)
    print(f"📂 Folder ID: {GOOGLE_DRIVE_FOLDER_ID}")
    print(f"💾 Output: {OUTPUT_PATH}")
    print("\n⏳ Downloading... (ini bisa memakan waktu beberapa menit)")
    
    try:
        # Download folder from Google Drive
        gdown.download_folder(
            id=GOOGLE_DRIVE_FOLDER_ID,
            output=OUTPUT_PATH,
            quiet=False,
            use_cookies=False
        )
        
        # Verify download
        required_files = ["config.json", "model.safetensors", "generation_config.json"]
        missing_files = []
        
        for file in required_files:
            if not os.path.exists(os.path.join(OUTPUT_PATH, file)):
                missing_files.append(file)
        
        if missing_files:
            print(f"\n⚠️  File berikut tidak ditemukan: {', '.join(missing_files)}")
            print("    Model mungkin tidak lengkap!")
            return False
        
        print(f"\n✅ BERHASIL download model!")
        print(f"📂 Model location: {OUTPUT_PATH}")
        print(f"\n✅ Verifikasi file:")
        for file in required_files:
            size = os.path.getsize(os.path.join(OUTPUT_PATH, file))
            print(f"   ✓ {file} ({size:,} bytes)")
        
        print(f"\n🚀 Sekarang Anda bisa jalankan:")
        print(f"   streamlit run streamlit_app.py")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Pastikan folder sudah di-share (Anyone with the link)")
        print("   2. Coba download manual dari Google Drive")
        print("   3. Pastikan koneksi internet stabil")
        print("\n📖 Link sharing:")
        print(f"   https://drive.google.com/drive/folders/{GOOGLE_DRIVE_FOLDER_ID}")
        return False

if __name__ == "__main__":
    success = download_model()
    sys.exit(0 if success else 1)
