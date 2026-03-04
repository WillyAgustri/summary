# 📝 Indonesian Text Summarizer - Streamlit App

Aplikasi web untuk meringkas teks bahasa Indonesia menggunakan model IndoBART-v2 yang telah di-fine-tune.

## 🚀 Quick Start

### 1. Install Dependencies

Jalankan perintah berikut untuk menginstall semua dependencies:

```bash
pip install -r requirements.txt
```

### 2. Persiapkan Model

Pastikan Anda sudah memiliki model yang telah di-train. Jika model ada di Google Drive:

**Opsi A: Download dari Google Drive**
- Download folder model dari Google Drive ke local
- Letakkan di folder `models/` dalam project ini

**Opsi B: Gunakan model langsung dari Drive (jika mounted)**
- Pastikan path model sudah benar
- Update path di aplikasi Streamlit

### 3. Jalankan Aplikasi

```bash
streamlit run streamlit_app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`

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

## 🎛️ Configuration

### Model Path
Default path: `./models/indobart-v2-detik-final`

Anda bisa mengubah path ini di sidebar aplikasi sesuai lokasi model Anda.

### Generation Parameters

**Recommended Settings untuk Kualitas Tinggi:**
- Beam Search: 4-6
- Temperature: 0.8-1.0
- Number of Sentences: 3-4

**Recommended Settings untuk Kecepatan:**
- Beam Search: 2
- Temperature: 1.0
- Number of Sentences: 2-3

## 📊 CSV Format untuk Batch Processing

File CSV harus memiliki minimal satu kolom bernama `text`:

```csv
text
"Berita pertama yang akan diringkas..."
"Berita kedua yang akan diringkas..."
"Berita ketiga yang akan diringkas..."
```

Anda juga bisa menambahkan kolom lain (misalnya `id`, `title`, dll.), semua kolom akan tetap ada di output dengan tambahan kolom `generated_summary`.

## 🔧 Troubleshooting

### Model tidak bisa di-load
- **Periksa path model**: Pastikan path mengarah ke folder yang berisi `config.json` dan file model
- **Periksa dependencies**: Pastikan semua package terinstall dengan benar
- **Periksa CUDA**: Jika menggunakan GPU, pastikan CUDA terinstall

### Out of Memory Error
- Kurangi `max_input_length` di sidebar
- Kurangi `batch_size` jika memproses banyak teks
- Gunakan CPU jika GPU memory tidak cukup

### Ringkasan tidak berkualitas
- Tingkatkan `num_beams` (beam search width)
- Sesuaikan `temperature` (coba 0.8-1.0)
- Periksa kualitas input text (pastikan teks bersih dan lengkap)

## 💻 System Requirements

**Minimum:**
- RAM: 8GB
- Storage: 5GB (untuk model)
- Python: 3.8+

**Recommended:**
- RAM: 16GB
- GPU: NVIDIA dengan 8GB+ VRAM (untuk inference lebih cepat)
- Storage: 10GB
- Python: 3.9+

## 📖 Usage Examples

### Example 1: Single Text
1. Buka aplikasi
2. Pilih tab "Single Text"
3. Paste artikel berita Indonesia
4. Klik "Summarize"
5. Lihat hasilnya!

### Example 2: Batch Processing
1. Siapkan CSV dengan kolom 'text'
2. Buka tab "Batch Processing"
3. Upload file CSV
4. Klik "Process All Texts"
5. Download hasil CSV

## 🤝 Support

Untuk pertanyaan atau masalah:
1. Periksa dokumentasi di tab "About" dalam aplikasi
2. Review error messages di terminal
3. Pastikan semua dependencies terinstall dengan benar

## 📝 Notes

- Model ini dilatih khusus untuk berita berbahasa Indonesia
- Hasil terbaik dicapai dengan teks berita yang terstruktur dengan baik
- Untuk teks sangat panjang (>2000 kata), proses mungkin memakan waktu lebih lama
- Aplikasi menggunakan caching untuk mempercepat loading model

## 🔄 Updates

**Version 1.0**
- Initial release
- Single text dan batch processing
- Configurable generation parameters
- Progress indicators
- Download results sebagai CSV

---

**Happy Summarizing! 📝✨**
