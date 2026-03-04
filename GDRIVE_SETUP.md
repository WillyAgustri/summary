# 📤 Panduan Share Model via Google Drive

Panduan lengkap untuk share model melalui Google Drive supaya orang lain bisa download.

## 🎯 Langkah untuk ANDA (Pemilik Model)

### 1. Upload Model ke Google Drive

1. Buka [Google Drive](https://drive.google.com)
2. Buat folder baru (atau masuk ke folder yang ada)
3. Upload folder model Anda: `indobart-v2-detik-final-20251201-061311`
4. Tunggu sampai semua file selesai upload (3 files: ~502 MB)

### 2. Share Folder Model

1. Klik kanan pada folder model
2. Pilih **"Share"** / **"Bagikan"**
3. Klik **"Change to anyone with the link"**
4. Set akses ke **"Viewer"** (supaya orang bisa download tapi tidak edit)
5. **Copy link** yang muncul

Link akan seperti ini:
```
https://drive.google.com/drive/folders/1a2b3c4d5e6f7g8h9i0j?usp=sharing
```

### 3. Dapatkan Folder ID

Dari link di atas, ambil bagian **Folder ID**:
```
https://drive.google.com/drive/folders/1a2b3c4d5e6f7g8h9i0j?usp=sharing
                                         ^^^^^^^^^^^^^^^^^^^^
                                         Ini adalah FOLDER_ID
```

### 4. Update README.md

Edit [README.md](README.md) dan ganti `YOUR_FOLDER_ID` dengan Folder ID Anda:

```markdown
**[Download Model di sini](https://drive.google.com/drive/folders/1a2b3c4d5e6f7g8h9i0j)** 
```

### 5. Update download_model.py (Optional)

Jika ingin user bisa auto-download, edit [download_model.py](download_model.py):

```python
GOOGLE_DRIVE_FOLDER_ID = "1a2b3c4d5e6f7g8h9i0j"  # Ganti dengan ID Anda
```

### 6. Commit & Push

```bash
git add .
git commit -m "Add model download instructions"
git push
```

✅ **Selesai!** Model siap di-share via Google Drive.

---

## 👥 Langkah untuk USER (Yang Download)

### Opsi 1: Download Manual (Mudah)

1. Klik link Google Drive yang ada di README
2. Download semua file di folder model
3. Taruh di folder `models/` di project Anda
4. Jalankan aplikasi

### Opsi 2: Auto-Download (dengan script)

```bash
# 1. Install gdown 
pip install gdown

# 2. Jalankan script download
python download_model.py

# 3. Jalankan aplikasi
streamlit run streamlit_app.py
```

---

## 🔧 Troubleshooting

### "Cannot download - file too large"

Google Drive kadang membatasi download file besar (>100MB) tanpa login.

**Solusi:**
1. Login ke Google Drive
2. Download manual dari browser
3. Atau gunakan script `download_model.py` dengan cookies

### "Folder ID not found"

**Solusi:**
1. Pastikan folder sudah di-share ("Anyone with the link")
2. Check bahwa Folder ID benar
3. Coba akses link di browser untuk test

### "Model files incomplete"

**Solusi:**
1. Pastikan semua files ter-upload di Google Drive:
   - `config.json`
   - `model.safetensors` (~502 MB)
   - `generation_config.json`
2. Re-upload jika ada yang missing

---

## 💡 Tips

### Versi Model

Jika Anda punya beberapa versi model, buat folder terpisah:

```
Google Drive/
└── Indonesian-Summarizer-Models/
    ├── indobart-v2-detik-final-20251201-061311/   (v1)
    ├── indobart-v2-detik-final-20251215-120000/   (v2)
    └── indobart-v2-detik-final-latest/            (latest)
```

### Alternatif Google Drive

Jika Google Drive lambat atau ada masalah, pertimbangkan:
- **HuggingFace Hub** (unlimited, gratis, recommended)
- **Dropbox**
- **OneDrive**
- **Self-hosted** (GitHub Releases - max 2GB)

---

## 📊 Size Reference

```
Model folder total: ~502 MB
├── config.json              (~1 KB)
├── generation_config.json   (~1 KB)
└── model.safetensors        (~502 MB)
```

Upload time estimate:
- 10 Mbps upload: ~7 menit
- 50 Mbps upload: ~1.5 menit
- 100 Mbps upload: ~40 detik

Download time estimate:
- 10 Mbps download: ~7 menit
- 50 Mbps download: ~1.5 menit
- 100 Mbps download: ~40 detik
