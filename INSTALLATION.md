# 🚀 Panduan Instalasi - Indonesian Text Summarizer

Panduan lengkap untuk menginstall dan menjalankan aplikasi Indonesian Text Summarizer di Windows.

## 📋 Prerequisites

### 1. Python
- **Versi**: Python 3.8 atau lebih tinggi
- **Download**: https://www.python.org/downloads/
- **Penting**: Centang "Add Python to PATH" saat instalasi

### 2. Git (Optional)
- Untuk clone repository jika diperlukan
- Download: https://git-scm.com/download/win

### 3. CUDA Toolkit (Optional - untuk GPU)
- Hanya jika Anda memiliki NVIDIA GPU dan ingin menggunakan GPU acceleration
- Download: https://developer.nvidia.com/cuda-downloads
- Pastikan versi CUDA kompatibel dengan PyTorch

## 📦 Instalasi Step-by-Step

### Step 1: Persiapan Project

**Opsi A: Jika sudah ada folder project**
```bash
cd c:\Coding\Google Colabs\novita
```

**Opsi B: Buat folder baru**
```bash
mkdir indonesian-summarizer
cd indonesian-summarizer
```

### Step 2: Install Dependencies

Buka **Command Prompt** atau **PowerShell** di folder project, lalu jalankan:

```bash
pip install -r requirements.txt
```

**Catatan**: Proses ini mungkin memakan waktu 5-15 menit tergantung koneksi internet.

**Jika ada error saat install**, coba:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Untuk CPU only** (jika tidak ada GPU):
```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

**Untuk GPU** (jika ada NVIDIA GPU):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Step 3: Download Model

Model yang sudah di-train perlu didownload dari Google Drive.

#### Via Browser:
1. Buka Google Drive Anda
2. Navigate ke folder model: `/content/drive/MyDrive/Colab Notebooks/Summary009/models/`
3. Download folder model (contoh: `indobart-v2-detik-final-20251201-061311`)
4. Extract di folder `models/` dalam project

#### Via Google Drive Desktop:
1. Install Google Drive Desktop
2. Model akan tersinkronisasi otomatis
3. Catat path lengkap ke folder model

**Struktur folder yang benar:**
```
novita/
├── streamlit_app.py
├── config.py
├── requirements.txt
├── models/
│   └── indobart-v2-detik-final/
│       ├── config.json
│       ├── pytorch_model.bin (atau model.safetensors)
│       └── generation_config.json
```

### Step 4: Konfigurasi

Edit file `config.py` dan sesuaikan `MODEL_PATH`:

```python
# Contoh path lokal
MODEL_PATH = "./models/indobart-v2-detik-final"

# Atau path absolut
MODEL_PATH = "C:/Coding/Google Colabs/novita/models/indobart-v2-detik-final"

# Atau dari Google Drive (jika mounted)
MODEL_PATH = "G:/My Drive/Colab Notebooks/Summary009/models/indobart-v2-detik-final-20251201-061311"
```

**Penting**: Gunakan forward slash `/` atau double backslash `\\` di Windows.

### Step 5: Test Setup

Sebelum menjalankan aplikasi, test dulu instalasi:

```bash
python test_setup.py
```

Script ini akan memeriksa:
- ✅ Semua dependencies terinstall
- ✅ CUDA availability (jika ada GPU)
- ✅ Model path sudah benar
- ✅ Model bisa di-load dengan sukses

**Jika semua test passed**, Anda siap menjalankan aplikasi!

### Step 6: Jalankan Aplikasi

**Opsi A: Via Command Prompt**
```bash
streamlit run streamlit_app.py
```

**Opsi B: Via Batch File (Windows)**
Double-click file `run_app.bat`

**Opsi C: Via PowerShell**
```powershell
streamlit run streamlit_app.py
```

Aplikasi akan otomatis membuka di browser pada `http://localhost:8501`

## 🔧 Troubleshooting

### Error: "No module named 'streamlit'"
```bash
pip install streamlit
```

### Error: "No module named 'indobenchmark'"
```bash
pip install indobenchmark-toolkit
```

### Error: "Torch not compiled with CUDA"
Ini normal jika Anda tidak install CUDA. Aplikasi akan menggunakan CPU.

Untuk enable GPU:
```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Error: "Model not found"
1. Periksa path di `config.py`
2. Pastikan folder model berisi `config.json` dan file model
3. Gunakan absolute path jika relative path tidak bekerja

### Error: "Out of Memory"
1. Kurangi `max_input_length` di sidebar aplikasi
2. Kurangi `batch_size` jika memproses CSV
3. Close aplikasi lain yang menggunakan memory
4. Gunakan CPU jika GPU memory tidak cukup

### Aplikasi lambat / hang
1. **Loading pertama** memang lama (1-3 menit) karena loading model
2. **Inference** lebih cepat setelah model loaded
3. Gunakan **GPU** jika tersedia untuk speed up
4. Kurangi **beam_search** width untuk inference lebih cepat

### Port 8501 sudah digunakan
```bash
streamlit run streamlit_app.py --server.port 8502
```

## 🎯 Quick Test

Setelah aplikasi berjalan:

1. **Tab Single Text**:
   - Paste teks berita Indonesia
   - Click "Summarize"
   - Lihat hasilnya

2. **Tab Batch Processing**:
   - Upload file `example_batch.csv`
   - Click "Process All Texts"
   - Download hasilnya

## 📊 System Requirements Check

Jalankan di Python untuk check system info:
```python
import torch
import sys

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## 🔄 Update Dependencies

Jika ingin update ke versi terbaru:
```bash
pip install --upgrade -r requirements.txt
```

## 💾 Uninstall

Untuk uninstall:
```bash
# Uninstall dependencies
pip uninstall -r requirements.txt -y

# Delete folder
cd ..
rmdir /s novita
```

## 📚 Next Steps

Setelah instalasi berhasil:

1. **Baca [README_STREAMLIT.md](README_STREAMLIT.md)** untuk panduan penggunaan
2. **Eksperimen dengan parameters** di sidebar
3. **Test dengan data Anda sendiri**
4. **Lihat tab "About"** dalam aplikasi untuk info lebih lanjut

## 🤝 Support

Jika masih ada masalah:

1. **Run test script**: `python test_setup.py`
2. **Check error messages** di terminal
3. **Review** panduan troubleshooting di atas
4. **Google** error message spesifik
5. **Check** GitHub Issues (jika applicable)

---

**Selamat menggunakan Indonesian Text Summarizer! 📝✨**
