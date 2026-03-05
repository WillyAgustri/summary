# -*- coding: utf-8 -*-
"""
Streamlit App - Indonesian Text Summarization
Model: IndoBART-v2 + LoRA Fine-tuning
(Pola berdasarkan copy_dari_09.py)
"""

import streamlit as st
import torch
import types
import pandas as pd
import os
from indobenchmark import IndoNLGTokenizer
from transformers import AutoModelForSeq2SeqLM, GenerationConfig, MarianMTModel, MarianTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from typing import List
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_NAME = "indobenchmark/indobart-v2"  # Base model dari HuggingFace
CHECKPOINT_PATH = "outputs/indobart-v2-detik/checkpoint-800"  # Path ke checkpoint fine-tuned
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-id-en"  # Model terjemahan Indonesia-Inggris
MAX_INPUT_LEN = 800
MAX_OUTPUT_LEN = 100
DEFAULT_NUM_SENTENCES = 3
DEFAULT_NUM_BEAMS = 4
PAGE_TITLE = "Indonesian Text Summarizer + Translator"
PAGE_ICON = "📝"
LAYOUT = "wide"

# ============================================================
# WEB SCRAPING FUNCTIONS
# ============================================================

def extract_text_from_url(url: str) -> tuple:
    """Extract teks artikel dari URL berita (sama seperti copy_dari_09.py)"""
    try:
        # Set headers agar tidak diblock
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Fetch halaman
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # --- Extract Title ---
        title = ""
        title_tag = soup.find('h1') or soup.find('title')
        if title_tag:
            title = title_tag.get_text().strip()
        if not title:
            title = "Tanpa Judul"
        
        # --- Auto-extract Date dari konten ---
        date = None
        # Coba pattern tanggal Indonesia: "5 Desember 2024", "01 Januari 2025"
        match = re.search(r"\d{1,2}\s+\w+\s+\d{4}", soup.get_text())
        if match:
            date = match.group(0)
        
        # --- Extract paragraf artikel ---
        article_text = ""
        
        # Method 1: Cari container umum untuk berita
        container = soup.find('div', class_='single-content') or soup.find('div', class_='entry-content')
        
        # Method 2: Cari tag article
        if not container:
            container = soup.find('article')
        
        # Method 3: Cari div dengan class yang umum untuk konten artikel
        if not container:
            content_divs = soup.find_all('div', class_=re.compile(r'(content|article|story|post-content|entry-content)', re.I))
            if content_divs:
                container = content_divs[0]
        
        # Extract paragraphs
        if container:
            paragraphs = container.find_all('p')
        else:
            # Fallback - ambil semua paragraf dari body
            paragraphs = soup.find_all('p')
        
        # --- Clean & filter paragraphs ---
        clean_paragraphs = []
        for p in paragraphs:
            text = p.get_text(" ", strip=True)
            if text and len(text) > 50:
                # Bersihkan teks sampah umum (watermark, copyright, dll)
                text = re.sub(r"\b(Dibaca|Foto|©|All rights reserved|MTD|WF|DINAS KOMUNIKASI.*)\b.*", "", text)
                text = re.sub(r"\b(Baca juga|Berita terkait|Simak video|ADVERTISEMENT|Halaman selanjutnya)\b.*", "", text, flags=re.IGNORECASE)
                text = text.strip()
                if text:
                    clean_paragraphs.append(text)
        
        article_text = " ".join(clean_paragraphs)
        
        # Normalisasi spasi
        article_text = re.sub(r'\s+', ' ', article_text).strip()
        
        if not article_text:
            return None, "Tidak dapat menemukan konten artikel di URL tersebut."
        
        return (title, date, article_text), None
        
    except requests.exceptions.RequestException as e:
        return None, f"Error fetching URL: {str(e)}"
    except Exception as e:
        return None, f"Error parsing content: {str(e)}"

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_resource
def load_translation_model():
    """Load translation model (Indonesian → English)""";
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        translation_model = MarianMTModel.from_pretrained(TRANSLATION_MODEL).to(device)
        translation_tokenizer = MarianTokenizer.from_pretrained(TRANSLATION_MODEL)
        translation_model.eval()
        return translation_model, translation_tokenizer, device
    except Exception as e:
        st.error(f"Error loading translation model: {e}")
        return None, None, device

@st.cache_resource
def load_model_and_tokenizer():
    """Load model IndoBART-v2 + LoRA fine-tuned dan tokenizer dengan caching"""
    # Auto-detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load tokenizer dari HuggingFace
    tokenizer = IndoNLGTokenizer.from_pretrained(MODEL_NAME)
    
    # Patch fungsi pad untuk kompatibilitas (dari copy_dari_09.py)
    def _compat_pad(self, encoded_inputs, padding=False, max_length=None,
                    pad_to_multiple_of=None, return_attention_mask=None,
                    return_tensors=None, verbose=True, **kwargs):
        return PreTrainedTokenizerBase.pad(
            self, encoded_inputs,
            padding=padding, max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
            return_tensors=return_tensors, verbose=verbose,
        )
    tokenizer.pad = types.MethodType(_compat_pad, tokenizer)
    
    # Load base model + LoRA fine-tuned checkpoint
    try:
        # Load base model dari HuggingFace
        base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
        
        # Setup pad token jika belum ada
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                base_model.resize_token_embeddings(len(tokenizer))
        
        # Load LoRA adapter dari checkpoint (jika ada)
        if os.path.exists(CHECKPOINT_PATH):
            model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
            model = model.merge_and_unload()  # Merge LoRA weights ke base model
            st.sidebar.info(f"✅ Loaded fine-tuned model from {CHECKPOINT_PATH}")
        else:
            st.sidebar.warning(f"⚠️ Checkpoint not found: {CHECKPOINT_PATH}")
            st.sidebar.info("Using base model without fine-tuning")
            model = base_model
        
        model = model.to(device)
        model.eval()
        
        # Set generation config (sama seperti copy_dari_09.py)
        model.generation_config = GenerationConfig(
            do_sample=False,              # nonaktifkan sampling agar lebih deterministik
            num_beams=4,                  # beam search = kualitas ringkasan naik signifikan
            top_p=0.9,
            temperature=0.8,
            top_k=40,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            length_penalty=2.0,           # penalti panjang biar ringkasan lebih padat
            max_new_tokens=128,
            min_new_tokens=25,
            early_stopping=True,
            decoder_start_token_id=(getattr(model.config, "decoder_start_token_id", None)
                                or getattr(tokenizer, "bos_token_id", None)
                                or getattr(tokenizer, "eos_token_id", None)),
        )
        
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error(f"Make sure checkpoint exists at: {CHECKPOINT_PATH}")
        return None, None, None

def chunk_text(text: str, tokenizer, max_input_length: int = 800, stride: int = 400) -> List[str]:
    """Membagi teks panjang jadi beberapa chunk"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=False,
    )
    input_ids = inputs["input_ids"][0]
    total_len = input_ids.size(0)
    chunks = []

    for i in range(0, total_len, stride):
        end = min(i + max_input_length, total_len)
        chunk_ids = input_ids[i:end]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
        if end == total_len:
            break
    return chunks

def summarize_chunk(text: str, model, tokenizer, device="cpu",
                    max_input_length: int = 800, max_output_length: int = 100,
                    num_beams: int = 4) -> str:
    """Ringkas 1 chunk teks (sama seperti copy_dari_09.py)"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_input_length,
        truncation=True,
        padding="max_length",
    ).to(device)

    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            # Parameter di bawah akan override generation_config jika diset
            # Gunakan generation_config default kecuali perlu override
            max_new_tokens=max_output_length,
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def summarize_long_text(
    text: str,
    model,
    tokenizer,
    device="cpu",
    max_input_length: int = 800,
    stride: int = 400,
    max_output_length: int = 100,
    num_sentences: int = 3,
    num_beams: int = 4,
    title: str = None,   # judul opsional
    date: str = None     # tanggal opsional
) -> str:
    """Ringkas teks panjang dengan chunking (sama seperti copy_dari_09.py)"""
    chunks = chunk_text(text, tokenizer, max_input_length=max_input_length, stride=stride)
    summaries = []

    progress_bar = st.progress(0)
    for i, chunk in enumerate(chunks):
        summary = summarize_chunk(
            chunk, model, tokenizer, device=device,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            num_beams=num_beams,
        )
        summaries.append(summary)
        progress_bar.progress((i + 1) / len(chunks))

    final_summary = " ".join(summaries).strip()

    # --------- NORMALISASI KALIMAT ---------
    text = final_summary.replace("\n", " ")
    text = " ".join(text.split())   # hilangkan spasi berlebih

    # --------- PISAH KALIMAT MODEL ---------
    sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 0]

    # --------- RINGKAS JIKA TERLALU PANJANG (Advanced Truncation) ---------
    if num_sentences == 1 and len(sentences) > 0:
        # Ambil hanya klausa pertama dari kalimat pertama untuk ringkasan 1 kalimat
        first = sentences[0]

        # Jika kalimat terlalu panjang, potong di koma pertama
        if "," in first and len(first) > 110:
            first = first.split(",")[0]

        # Ambil maksimum 18–22 kata agar tetap 1 kalimat padat
        words = first.split()
        if len(words) > 22:
            first = " ".join(words[:22])

        final_summary = first
    else:
        # Untuk multi-sentence, ambil N kalimat pertama
        if len(sentences) > num_sentences:
            sentences = sentences[:num_sentences]
        final_summary = ". ".join(sentences).strip()

    # --------- TAMBAH TITIK DI AKHIR ---------
    if not final_summary.endswith("."):
        final_summary += "."

    # --- Susun header (judul + tanggal kalau ada) ---
    header_parts = []
    if title:
        header_parts.append(f"📰 {title}")
    if date:
        header_parts.append(f"📅 {date}")

    header = "\n".join(header_parts)
    if header:
        final_summary = f"{header}\n{final_summary}"

    return final_summary

# ============================================================
# TRANSLATION FUNCTIONS
# ============================================================

def translate_text(text: str, model, tokenizer, device="cpu", max_length: int = 512) -> str:
    """Translate Indonesian text to English"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    
    with torch.no_grad():
        translated = model.generate(**inputs, max_length=max_length)
    
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

def translate_long_text(text: str, model, tokenizer, device="cpu", max_length: int = 512, chunk_size: int = 400) -> str:
    """Translate long text by splitting into chunks"""
    # Split text into sentences to maintain context
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    # Translate each chunk
    translated_chunks = []
    for chunk in chunks:
        translated = translate_text(chunk, model, tokenizer, device, max_length)
        translated_chunks.append(translated)
    
    return " ".join(translated_chunks)

# ============================================================
# MAIN APP
# ============================================================

def main():
    st.title("📝 Indonesian Text Summarizer + Translator")
    st.markdown(f"**Summarization Model:** {MODEL_NAME} (Fine-tuned)")
    st.markdown(f"**Translation Model:** {TRANSLATION_MODEL}")
    
    # Sidebar untuk konfigurasi
    st.sidebar.header("⚙️ Configuration")
    
    # Load model secara otomatis
    if 'model' not in st.session_state:
        with st.spinner("Loading summarization model..."):
            model, tokenizer, device = load_model_and_tokenizer()
            if model is not None:
                st.session_state['model'] = model
                st.session_state['tokenizer'] = tokenizer
                st.session_state['device'] = device
                st.sidebar.success(f"✅ Summarization model loaded on {device}")
            else:
                st.sidebar.error("❌ Failed to load model")
                st.error("⚠️ Gagal memuat model. Silakan refresh halaman.")
                return
    else:
        st.sidebar.success(f"✅ Summarization model ready on {st.session_state['device']}")
    
    # Load translation model (lazy loading)
    if 'translation_model' not in st.session_state:
        st.session_state['translation_model'] = None
        st.session_state['translation_tokenizer'] = None
        st.session_state['translation_loaded'] = False
    
    # Generation parameters
    st.sidebar.header("🎛️ Generation Parameters")
    num_sentences = st.sidebar.slider(
        "Number of Sentences", 
        min_value=1, max_value=10, 
        value=DEFAULT_NUM_SENTENCES,
        help="Jumlah kalimat dalam ringkasan"
    )
    max_output_length = st.sidebar.slider(
        "Max Output Length (tokens)", 
        min_value=50, max_value=200, 
        value=MAX_OUTPUT_LEN,
        help="Panjang maksimum output dalam token"
    )
    max_input_length = st.sidebar.slider(
        "Max Input Length (tokens)", 
        min_value=400, max_value=1024, 
        value=MAX_INPUT_LEN,
        help="Panjang maksimum input dalam token"
    )
    num_beams = st.sidebar.slider(
        "Beam Search Width", 
        min_value=1, max_value=8, 
        value=DEFAULT_NUM_BEAMS,
        help="Beam search: nilai lebih tinggi = kualitas lebih baik tapi lebih lambat"
    )
    
    # Info model
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Info")
    st.sidebar.markdown(f"**Source:** HuggingFace Hub")
    st.sidebar.markdown(f"**Model ID:** `{MODEL_NAME}`")
    st.sidebar.markdown(f"**Device:** `{st.session_state['device']}`")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Single Text", "📊 Batch Processing", "🌐 Translation", "ℹ️ About"])
    
    # Tab 1: Single Text Summarization
    with tab1:
        st.header("Single Text Summarization")
        
        # Pilihan input method
        input_method = st.radio(
            "Pilih metode input:",
            ["📝 Manual Text", "🔗 URL Berita"],
            horizontal=True
        )
        
        text_input = ""
        article_title = ""
        article_date = ""
        
        if input_method == "📝 Manual Text":
            # Optional: Title dan Date
            col1, col2 = st.columns([3, 1])
            with col1:
                article_title = st.text_input(
                    "Judul (opsional):",
                    placeholder="Masukkan judul berita...",
                    key="manual_title"
                )
            with col2:
                article_date = st.text_input(
                    "Tanggal (opsional):",
                    placeholder="01/01/2024",
                    key="manual_date"
                )
            
            # Text input manual
            text_input = st.text_area(
                "Enter Indonesian text to summarize:",
                height=250,
                placeholder="Masukkan teks berita atau artikel dalam Bahasa Indonesia di sini...",
                key="manual_text"
            )
        else:
            # URL input
            url_input = st.text_input(
                "Masukkan URL berita:",
                placeholder="https://www.detik.com/...",
                key="url_input"
            )
            
            if url_input:
                if st.button("📥 Fetch Article", key="fetch_btn"):
                    with st.spinner("Mengambil artikel dari URL..."):
                        result, error = extract_text_from_url(url_input)
                        if error:
                            st.error(f"❌ {error}")
                        else:
                            article_title, extracted_date, text_input = result
                            st.session_state['fetched_text'] = text_input
                            st.session_state['fetched_title'] = article_title
                            st.session_state['fetched_date'] = extracted_date
                            
                            if extracted_date:
                                st.success(f"✅ Artikel berhasil diambil! (Tanggal terdeteksi: {extracted_date})")
                            else:
                                st.success("✅ Artikel berhasil diambil!")
            
            # Display fetched text dan info jika ada
            if 'fetched_text' in st.session_state:
                text_input = st.session_state['fetched_text']
                article_title = st.session_state.get('fetched_title', '')
                article_date = st.session_state.get('fetched_date', '')
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    if article_title:
                        st.markdown(f"**Judul:** {article_title}")
                with col2:
                    if article_date:
                        st.markdown(f"**📅 Tanggal:** {article_date}")
                
                # Optional: override tanggal jika user mau
                if not article_date:
                    article_date = st.text_input(
                        "Tanggal tidak terdeteksi. Masukkan manual (opsional):",
                        placeholder="01 Januari 2024",
                        key="url_date_manual"
                    )
                
                st.text_area(
                    "Teks artikel yang diambil:",
                    value=text_input,
                    height=200,
                    key="display_fetched",
                    disabled=True
                )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            summarize_btn = st.button("🚀 Summarize", type="primary")
        with col2:
            if text_input:
                word_count = len(text_input.split())
                st.caption(f"Input: {word_count} words")
        
        if summarize_btn:
            if not text_input.strip():
                st.warning("⚠️ Silakan masukkan teks atau fetch artikel dari URL terlebih dahulu.")
            else:
                with st.spinner("Generating summary..."):
                    try:
                        summary = summarize_long_text(
                            text_input,
                            st.session_state['model'],
                            st.session_state['tokenizer'],
                            device=st.session_state['device'],
                            max_input_length=max_input_length,
                            max_output_length=max_output_length,
                            num_sentences=num_sentences,
                            num_beams=num_beams,
                            title=article_title if article_title.strip() else None,
                            date=article_date if article_date.strip() else None,
                        )
                        
                        st.success("✅ Summary generated!")
                        
                        st.markdown("### 📋 Ringkasan:")
                        st.info(summary)
                        
                        # Statistics (hitung tanpa header jika ada)
                        summary_text_only = summary
                        if article_title or article_date:
                            # Remove header lines untuk counting
                            lines = summary.split("\\n")
                            summary_text_only = lines[-1] if len(lines) > 0 else summary
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Original Words", len(text_input.split()))
                        with col2:
                            st.metric("Summary Words", len(summary_text_only.split()))
                        with col3:
                            compression = (1 - len(summary_text_only.split()) / len(text_input.split())) * 100
                            st.metric("Compression", f"{compression:.1f}%")
                            
                    except Exception as e:
                        st.error(f"❌ Error during summarization: {e}")
    
    # Tab 2: Batch Processing
    with tab2:
        st.header("Batch Processing")
        st.markdown("Upload file CSV dengan kolom 'text' untuk meringkas banyak teks sekaligus.")
        st.markdown("*Opsional: Tambahkan kolom 'title' dan 'date' untuk header di ringkasan.*")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ File loaded: {len(df)} rows")
                
                # Show preview
                st.markdown("**Preview:**")
                st.dataframe(df.head())
                
                # Check for 'text' column
                if 'text' not in df.columns:
                    st.error("❌ File CSV harus memiliki kolom 'text'")
                else:
                    # Check optional columns
                    has_title = 'title' in df.columns
                    has_date = 'date' in df.columns
                    
                    if has_title or has_date:
                        optional_cols = []
                        if has_title:
                            optional_cols.append("'title'")
                        if has_date:
                            optional_cols.append("'date'")
                        st.info(f"ℹ️ Kolom opsional ditemukan: {', '.join(optional_cols)}")
                    
                    if st.button("🚀 Process All Texts", type="primary"):
                        summaries = []
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for idx, row in df.iterrows():
                            status_text.text(f"Processing {idx+1}/{len(df)}...")
                            text = str(row['text']) if pd.notna(row['text']) else ""
                            
                            # Get optional columns
                            title = str(row['title']) if has_title and pd.notna(row.get('title')) else None
                            date = str(row['date']) if has_date and pd.notna(row.get('date')) else None
                            
                            if text.strip():
                                try:
                                    summary = summarize_long_text(
                                        text,
                                        st.session_state['model'],
                                        st.session_state['tokenizer'],
                                        device=st.session_state['device'],
                                        max_input_length=max_input_length,
                                        max_output_length=max_output_length,
                                        num_sentences=num_sentences,
                                        num_beams=num_beams,
                                        title=title,
                                        date=date,
                                    )
                                    summaries.append(summary)
                                except Exception as e:
                                    summaries.append(f"Error: {str(e)}")
                            else:
                                summaries.append("")
                            
                            progress_bar.progress((idx + 1) / len(df))
                        
                        # Add summaries to dataframe
                        df['generated_summary'] = summaries
                        
                        st.success("✅ All texts processed!")
                        
                        # Display columns yang relevan
                        display_cols = ['text', 'generated_summary']
                        if has_title:
                            display_cols.insert(0, 'title')
                        if has_date:
                            display_cols.insert(1 if has_title else 0, 'date')
                        
                        st.dataframe(df[display_cols])
                        
                        # Download button
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name="summarized_results.csv",
                            mime="text/csv",
                        )
                        
            except Exception as e:
                st.error(f"❌ Error loading file: {e}")
    
    # Tab 3: Translation (Indonesian → English)
    with tab3:
        st.header("🌐 Translation: Indonesian → English")
        st.markdown("Terjemahkan teks atau ringkasan dari Bahasa Indonesia ke Bahasa Inggris")
        
        # Load translation model on demand
        if not st.session_state['translation_loaded']:
            if st.button("📥 Load Translation Model", type="primary"):
                with st.spinner("Loading translation model (Helsinki-NLP/opus-mt-id-en)..."):
                    trans_model, trans_tokenizer, trans_device = load_translation_model()
                    if trans_model is not None:
                        st.session_state['translation_model'] = trans_model
                        st.session_state['translation_tokenizer'] = trans_tokenizer
                        st.session_state['translation_device'] = trans_device
                        st.session_state['translation_loaded'] = True
                        st.success(f"✅ Translation model loaded on {trans_device}")
                        st.rerun()
                    else:
                        st.error("❌ Failed to load translation model")
            
            st.info("ℹ️ Klik tombol di atas untuk memuat model terjemahan (~300MB)")
        else:
            st.success(f"✅ Translation model ready on {st.session_state.get('translation_device', 'cpu')}")
            
            # Translation input
            translation_input = st.text_area(
                "Masukkan teks Bahasa Indonesia:",
                height=200,
                placeholder="Contoh: Pemerintah mengumumkan kebijakan baru untuk meningkatkan perekonomian...",
                key="translation_input"
            )
            
            # Translation options
            col1, col2 = st.columns([1, 5])
            with col1:
                translate_btn = st.button("🔄 Translate", type="primary")
            with col2:
                if translation_input:
                    word_count = len(translation_input.split())
                    st.caption(f"Input: {word_count} words")
            
            if translate_btn:
                if not translation_input.strip():
                    st.warning("⚠️ Silakan masukkan teks yang akan diterjemahkan.")
                else:
                    with st.spinner("Translating..."):
                        try:
                            # Translate text
                            if len(translation_input.split()) > 400:
                                # Long text: use chunking
                                translated = translate_long_text(
                                    translation_input,
                                    st.session_state['translation_model'],
                                    st.session_state['translation_tokenizer'],
                                    device=st.session_state['translation_device']
                                )
                            else:
                                # Short text: direct translation
                                translated = translate_text(
                                    translation_input,
                                    st.session_state['translation_model'],
                                    st.session_state['translation_tokenizer'],
                                    device=st.session_state['translation_device']
                                )
                            
                            st.success("✅ Translation completed!")
                            
                            # Display results side by side
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("### 🇮🇩 Indonesian")
                                st.info(translation_input)
                            with col2:
                                st.markdown("### 🇬🇧 English")
                                st.success(translated)
                            
                            # Statistics
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Indonesian Words", len(translation_input.split()))
                            with col2:
                                st.metric("English Words", len(translated.split()))
                                
                        except Exception as e:
                            st.error(f"❌ Error during translation: {e}")
            
            # Quick tip
            st.markdown("---")
            st.markdown("""
            **💡 Tips:**
            - Model ini ringan (~300MB) dan cukup akurat untuk teks berita
            - Cocok untuk: artikel berita, ringkasan, teks formal
            - Untuk teks panjang (>400 kata), otomatis di-chunk
            - Combine dengan summarization: ringkas dulu, lalu terjemahkan!
            """)
    
    # Tab 4: About
    with tab4:
        st.header("About This App")
        st.markdown("""
        ### 📖 Overview
        Aplikasi ini menggunakan **IndoBART-v2** yang telah di-**fine-tune** dengan teknik 
        **LoRA (Low-Rank Adaptation)** menggunakan dataset ringkasan berita lokal Indonesia 
        (MC, MMC, dan Detik).
        
        ### 🎯 Features
        - **Single Text Summarization**: Meringkas teks individual secara instan
        - **Title & Date Header**: Tambahkan judul dan tanggal opsional di header ringkasan
        - **Batch Processing**: Memproses banyak teks dari file CSV (support kolom title & date)
        - **URL Extraction**: Extract dan ringkas artikel dari URL berita
          - **Auto-detect Date**: Otomatis ekstrak tanggal publikasi dari artikel
          - **Smart Text Cleaning**: Otomatis filter watermark, copyright, dan teks sampah
        - **Translation (Indonesian → English)**: 🆕
          - Model ringan Helsinki-NLP/opus-mt-id-en (~300MB)
          - Support teks panjang dengan chunking otomatis
          - Perfect combo: Ringkas → Terjemahkan
        - **Advanced Truncation**: Otomatis potong kalimat panjang untuk ringkasan 1 kalimat
        - **Customizable Parameters**: Sesuaikan pengaturan generasi untuk berbagai kasus
        - **Long Text Support**: Otomatis chunking untuk teks yang melebihi batas token
        
        ### 🔧 Technical Details
        **Summarization:**
        - **Base Model**: IndoBART-v2 (indobenchmark/indobart-v2)
        - **Fine-tuning**: LoRA/PEFT dengan dataset berita Indonesia
        - **Training Data**: MC, MMC, Detik (ringkasan berita lokal)
        - **Max Input**: 800 tokens (800 token per chunk)
        - **Max Output**: 100 tokens
        - **Generation**: Beam search (4 beams) dengan no_repeat_ngram
        - **Advanced Features**: 
          - Sentence truncation (max 22 words untuk 1 kalimat)
          - Auto date extraction (regex: "\d{1,2}\s+\w+\s+\d{4}")
          - Text cleaning (filter: watermark, copyright, metadata)
        
        **Translation:**
        - **Model**: Helsinki-NLP/opus-mt-id-en (MarianMT)
        - **Direction**: Indonesian → English only
        - **Size**: ~300MB (lightweight)
        - **Max tokens**: 512 per chunk
        - **Auto-chunking**: Support teks >400 kata
        
        - **Device**: Auto-detect CUDA/CPU
        
        ### 📊 Generation Parameters
        - **Num Beams**: Beam search width - nilai lebih tinggi = kualitas lebih baik (default: 4)
        - **Max Output Length**: Panjang maksimum output dalam token (default: 100)
        - **Max Input Length**: Panjang maksimum input, teks lebih panjang akan di-chunk (default: 800)
        - **Num Sentences**: Jumlah kalimat dalam ringkasan akhir (default: 3)
          - **1 kalimat**: Otomatis truncate di koma jika > 110 char, max 22 kata
          - **Multi-kalimat**: Ambil N kalimat pertama dari hasil model
        
        ### 💡 Usage Tips
        - Model ini telah di-fine-tune khusus untuk ringkasan berita Indonesia
        - **Title & Date**: Masukkan judul dan tanggal untuk header ringkasan yang informatif
        - **URL Mode**: Tanggal otomatis terdeteksi dari artikel (format: "5 Desember 2024")
        - **Text Cleaning**: Scraper otomatis remove watermark (MTD, WF), copyright, dan metadata
        - **1 Sentence Mode**: Cocok untuk headline/lead, otomatis dipotong jadi ringkasan ultra-padat
        - **Batch CSV**: File CSV bisa punya kolom 'text', 'title' (opsional), 'date' (opsional)
        - Untuk hasil terbaik, gunakan kalimat lengkap dan teks terstruktur dengan baik
        - Sesuaikan jumlah kalimat berdasarkan panjang ringkasan yang diinginkan
        - Nilai beam search lebih tinggi menghasilkan kualitas lebih baik namun lebih lambat
        - Model ini bekerja paling baik dengan teks berita dalam Bahasa Indonesia
        
        ### 📚 Model Information
        **Base Model**: IndoBART-v2 adalah model sequence-to-sequence berbasis BART yang di-pretrain 
        pada korpus Bahasa Indonesia.
        
        **Fine-tuning**: Model ini telah di-fine-tune menggunakan:
        - **Teknik**: LoRA (Low-Rank Adaptation) dengan PEFT
        - **Dataset**: Gabungan dataset MC, MMC, dan Detik (berita Indonesia)
        - **Parameters**: r=16, lora_alpha=32, lora_dropout=0.05
        - **Target Modules**: q_proj, k_proj, v_proj, o_proj, fc1, fc2
        - **Training**: 5 epochs dengan learning rate 5e-5
        
        **Paper**: [IndoNLG: Benchmark and Resources for Evaluating Indonesian Natural Language Generation](https://aclanthology.org/2021.emnlp-main.699/)
        
        ### 🤝 Support
        Untuk pertanyaan atau masalah, silakan merujuk pada dokumentasi model atau hubungi tim pengembang.
        """)
        
        st.markdown("---")
        st.markdown(f"**Base Model:** `{MODEL_NAME}`")
        st.markdown(f"**Checkpoint:** `{CHECKPOINT_PATH}`")
        if 'device' in st.session_state:
            st.markdown(f"**Device:** `{st.session_state['device']}`")
        st.markdown("**Framework:** PyTorch + Transformers + PEFT + Streamlit")

# ============================================================
# RUN APP
# ============================================================

if __name__ == "__main__":
    main()
