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
- Upload file CSV dengan kolom text (atau `Isi Berita`)
- Support format dataset training: `Judul`, `Tanggal`, `text`/`Isi Berita`
- Proses banyak teks sekaligus dengan optional translation
- Download hasil dalam format CSV (include kolom `generated_summary` & `english_summary`)

### 3️⃣ Translation (Indonesian → English)
- Centang checkbox untuk auto-translate hasil ringkasan
- Model ringan Helsinki-NLP/opus-mt-id-en (~300MB)
- Support teks panjang dengan chunking otomatis
- Tab 1: Side-by-side display (Indonesia | English)
- Tab 2: CSV output dengan kolom tambahan `english_summary`

### 4️⃣ URL Extraction
- Input URL artikel berita
- Auto-extract tanggal publikasi dari artikel
- Smart text cleaning (filter watermark, copyright, metadata)
- Langsung ringkas artikel yang di-scrape

### 5️⃣ Customizable Parameters
- **Number of Sentences**: Jumlah kalimat dalam ringkasan (1-10)
- **Max Output Length**: Panjang maksimum output dalam token (50-200)
- **Max Input Length**: Panjang maksimum input dalam token (400-1024)
- **Beam Search Width**: Lebar beam search untuk kualitas lebih baik (1-8)
- **Temperature**: Mengatur diversitas output (0.1-2.0)

## 📊 CSV Format untuk Batch Processing

Aplikasi mendukung 2 format CSV:

### Format 1: Dataset Training (Compatible)
```csv
Judul,Tanggal,text
"Judul Berita 1",01/03/2026,"Isi berita yang akan diringkas..."
"Judul Berita 2",02/03/2026,"Isi berita kedua..."
```

Atau dengan kolom `Isi Berita`:
```csv
Judul,Tanggal,Isi Berita
"Judul Berita 1",01/03/2026,"Isi berita yang akan diringkas..."
```

### Format 2: Format Umum (English headers)
```csv
title,date,text
"News Title 1",01/03/2026,"Text content to be summarized..."
"News Title 2",02/03/2026,"Second news content..."
```

### Format Minimal (hanya text)
```csv
text
"Berita pertama yang akan diringkas..."
"Berita kedua yang akan diringkas..."
```

**Catatan:**
- Kolom **text** atau **Isi Berita** adalah **wajib**
- Kolom **Judul/title** dan **Tanggal/date** adalah **opsional** (akan ditambahkan sebagai header di ringkasan)
- Deteksi kolom **case-insensitive** (`Judul` = `judul` = `JUDUL`)
- File example: 
  - Format umum: [`example_batch.csv`](example_batch.csv)
  - Format dataset: [`example_batch_dataset.csv`](example_batch_dataset.csv)
- Output akan memiliki kolom `generated_summary` (dan `english_summary` jika translate dicentang)

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
