# Config file untuk Streamlit App
# (Pola berdasarkan copy_dari_09.py)

# Model Configuration
# Model langsung dari HuggingFace Hub
MODEL_PATH = "indobenchmark/indobart-v2"  # Model IndoBART-v2 dari HuggingFace

# Generation Parameters (default values)
DEFAULT_NUM_SENTENCES = 3
DEFAULT_MAX_OUTPUT_LENGTH = 100
DEFAULT_MAX_INPUT_LENGTH = 800
DEFAULT_NUM_BEAMS = 4

# Processing Settings
ENABLE_CHUNKING = True  # Untuk teks panjang
CHUNK_STRIDE = 400  # Overlap antar chunks

# Cache Settings
USE_MODEL_CACHE = True  # Streamlit caching untuk model

# UI Settings
PAGE_TITLE = "Indonesian Text Summarizer"
PAGE_ICON = "📝"
LAYOUT = "wide"  # "wide" or "centered"

# Device Settings (auto-detect jika kosong)
# Options: "cuda", "cpu", or "" for auto-detect
DEVICE = ""  # Kosong = auto-detect

# Note: File config ini sudah tidak digunakan lagi di streamlit_app.py versi terbaru
# Konfigurasi sekarang hardcoded di streamlit_app.py untuk konsistensi dengan copy_dari_09.py
