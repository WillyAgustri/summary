# Config file untuk Streamlit App
# Salin file ini dan sesuaikan dengan setup Anda

# Path ke model fine-tuned
# Contoh: ./models/indobart-v2-detik-final-20251201-061311
MODEL_PATH = "./models/indobart-v2-detik-final"

# Generation Parameters (default values)
DEFAULT_NUM_SENTENCES = 3
DEFAULT_MAX_OUTPUT_LENGTH = 100
DEFAULT_MAX_INPUT_LENGTH = 800
DEFAULT_NUM_BEAMS = 4
DEFAULT_TEMPERATURE = 1.0

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
