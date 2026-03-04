# 📝 Indonesian Text Summarizer

Aplikasi web untuk meringkas teks bahasa Indonesia menggunakan model IndoBART-v2 yang telah di-fine-tune.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Model

**File model tidak di-include di repository karena ukurannya besar (~502 MB)**

#### Download dari Google Drive:
1. **[Download Model di sini](https://drive.google.com/drive/folders/YOUR_FOLDER_ID)** 
2. Extract file model
3. Taruh di folder `models/` sehingga strukturnya:
   ```
   summary/
   └── models/
       └── indobart-v2-detik-final-20251201-061311/
           ├── config.json
           ├── generation_config.json
           └── model.safetensors
   ```

#### Atau gunakan script auto-download:
```bash
# Edit download_model.py dengan File/Folder ID Anda
python download_model.py
```

### 3. Jalankan Aplikasi

```bash
streamlit run streamlit_app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`

---

## 📋 Features

### 1️⃣ Single Text Summarization
- Input teks secara manual
- Dapatkan ringkasan instan
- Lihat statistik kompresi

### 2️⃣ Batch Processing
- Upload file CSV dengan kolom 'text'
- Proses banyak teks sekaligus
- Download hasil dalam format CSV

### 3️⃣ Customizable Parameters
- **Number of Sentences**: Jumlah kalimat dalam ringkasan (1-10)
- **Max Output Length**: Panjang maksimum output dalam token (50-200)
- **Max Input Length**: Panjang maksimum input dalam token (400-1024)
- **Beam Search Width**: Lebar beam search untuk kualitas lebih baik (1-8)
- **Temperature**: Mengatur diversitas output (0.1-2.0)

## 📊 CSV Format untuk Batch Processing

File CSV harus memiliki minimal satu kolom bernama `text`:

```csv
text
"Berita pertama yang akan diringkas..."
"Berita kedua yang akan diringkas..."
```

## 🔧 Troubleshooting

### Model tidak bisa di-load
- Pastikan model sudah didownload ke folder `models/`
- Periksa path di `config.py` sesuai dengan nama folder model
- Periksa dependencies terinstall dengan benar

### Out of Memory Error
- Kurangi `max_input_length` di sidebar
- Gunakan CPU jika GPU memory tidak cukup

### Ringkasan tidak berkualitas
- Tingkatkan `num_beams` (beam search width)
- Sesuaikan `temperature` (coba 0.8-1.0)

## 📖 Documentation

Lihat file lainnya untuk informasi lebih detail:
- [INSTALLATION.md](INSTALLATION.md) - Panduan instalasi lengkap
- [README_STREAMLIT.md](README_STREAMLIT.md) - Detail fitur aplikasi

## 📜 License

MIT License

## 👤 Author

Willy Agustri
