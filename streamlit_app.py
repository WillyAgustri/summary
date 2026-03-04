# -*- coding: utf-8 -*-
"""
Streamlit App - Indonesian Text Summarization
Model: IndoBART-v2
(Pola berdasarkan copy_dari_09.py)
"""

import streamlit as st
import torch
import types
import pandas as pd
from indobenchmark import IndoNLGTokenizer
from transformers import AutoModelForSeq2SeqLM, GenerationConfig
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import List
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_NAME = "indobenchmark/indobart-v2"  # Model langsung dari HuggingFace
MAX_INPUT_LEN = 800
MAX_OUTPUT_LEN = 100
DEFAULT_NUM_SENTENCES = 3
DEFAULT_NUM_BEAMS = 4
PAGE_TITLE = "Indonesian Text Summarizer"
PAGE_ICON = "📝"
LAYOUT = "wide"

# ============================================================
# WEB SCRAPING FUNCTIONS
# ============================================================

def extract_text_from_url(url: str) -> tuple:
    """Extract teks artikel dari URL berita"""
    try:
        # Set headers agar tidak diblock
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Fetch halaman
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Coba extract title
        title = ""
        title_tag = soup.find('h1') or soup.find('title')
        if title_tag:
            title = title_tag.get_text().strip()
        
        # Extract paragraf artikel
        # Coba beberapa selector umum untuk situs berita Indonesia
        article_text = ""
        
        # Method 1: Cari tag article
        article = soup.find('article')
        if article:
            paragraphs = article.find_all('p')
            article_text = ' '.join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 50])
        
        # Method 2: Cari div dengan class yang umum untuk konten artikel
        if not article_text:
            content_divs = soup.find_all('div', class_=re.compile(r'(content|article|story|post-content|entry-content)', re.I))
            for div in content_divs:
                paragraphs = div.find_all('p')
                text = ' '.join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 50])
                if len(text) > len(article_text):
                    article_text = text
        
        # Method 3: Fallback - ambil semua paragraf dari body
        if not article_text:
            paragraphs = soup.find_all('p')
            article_text = ' '.join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 50])
        
        # Bersihkan teks
        article_text = re.sub(r'\s+', ' ', article_text).strip()
        
        if not article_text:
            return None, "Tidak dapat menemukan konten artikel di URL tersebut."
        
        return (title, article_text), None
        
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
def load_model_and_tokenizer():
    """Load model IndoBART-v2 dan tokenizer dengan caching"""
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
    
    # Load model dari HuggingFace
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
        model.eval()
        
        # Setup pad token jika belum ada
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                model.resize_token_embeddings(len(tokenizer))
        
        # Set generation config (mirip copy_dari_09.py)
        model.generation_config = GenerationConfig(
            do_sample=False,              # deterministic untuk konsistensi
            num_beams=4,                  # beam search untuk kualitas
            top_p=0.9,
            temperature=0.8,
            top_k=40,
            no_repeat_ngram_size=3,       # hindari pengulangan
            repetition_penalty=1.2,
            length_penalty=1.0,           # penalti panjang
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
    """Ringkas 1 chunk teks (mirip copy_dari_09.py)"""
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
            num_beams=num_beams,
            max_new_tokens=max_output_length,  # gunakan max_new_tokens bukan max_length
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
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
) -> str:
    """Ringkas teks panjang dengan chunking (mirip copy_dari_09.py)"""
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

    # Gabungkan semua summary dan ambil N kalimat pertama
    final_summary = " ".join(summaries).strip()
    
    # Normalisasi teks
    final_summary = final_summary.replace("\n", " ")
    final_summary = " ".join(final_summary.split())
    
    # Split berdasarkan titik
    sentences = [s.strip() for s in final_summary.split(".") if len(s.strip()) > 0]
    
    # Ambil N kalimat pertama
    if len(sentences) > num_sentences:
        sentences = sentences[:num_sentences]
    
    final_summary = ". ".join(sentences).strip()
    if not final_summary.endswith("."):
        final_summary += "."
    
    return final_summary

# ============================================================
# MAIN APP
# ============================================================

def main():
    st.title("📝 Indonesian Text Summarizer")
    st.markdown(f"**Model:** {MODEL_NAME}")
    
    # Sidebar untuk konfigurasi
    st.sidebar.header("⚙️ Configuration")
    
    # Load model secara otomatis
    if 'model' not in st.session_state:
        with st.spinner("Loading model from HuggingFace..."):
            model, tokenizer, device = load_model_and_tokenizer()
            if model is not None:
                st.session_state['model'] = model
                st.session_state['tokenizer'] = tokenizer
                st.session_state['device'] = device
                st.sidebar.success(f"✅ Model loaded on {device}")
            else:
                st.sidebar.error("❌ Failed to load model")
                st.error("⚠️ Gagal memuat model. Silakan refresh halaman.")
                return
    else:
        st.sidebar.success(f"✅ Model ready on {st.session_state['device']}")
    
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
    tab1, tab2, tab3 = st.tabs(["📄 Single Text", "📊 Batch Processing", "ℹ️ About"])
    
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
        
        if input_method == "📝 Manual Text":
            # Text input manual
            text_input = st.text_area(
                "Enter Indonesian text to summarize:",
                height=300,
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
                            article_title, text_input = result
                            st.session_state['fetched_text'] = text_input
                            st.session_state['fetched_title'] = article_title
                            st.success("✅ Artikel berhasil diambil!")
            
            # Display fetched text jika ada
            if 'fetched_text' in st.session_state:
                text_input = st.session_state['fetched_text']
                article_title = st.session_state.get('fetched_title', '')
                
                if article_title:
                    st.markdown(f"**Judul:** {article_title}")
                
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
                        )
                        
                        st.success("✅ Summary generated!")
                        
                        # Tampilkan judul jika ada
                        if article_title:
                            st.markdown(f"### 📰 {article_title}")
                        
                        st.markdown("### 📋 Ringkasan:")
                        st.info(summary)
                        
                        # Statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Original Words", len(text_input.split()))
                        with col2:
                            st.metric("Summary Words", len(summary.split()))
                        with col3:
                            compression = (1 - len(summary.split()) / len(text_input.split())) * 100
                            st.metric("Compression", f"{compression:.1f}%")
                            
                    except Exception as e:
                        st.error(f"❌ Error during summarization: {e}")
    
    # Tab 2: Batch Processing
    with tab2:
        st.header("Batch Processing")
        st.markdown("Upload file CSV dengan kolom 'text' untuk meringkas banyak teks sekaligus.")
        
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
                    if st.button("🚀 Process All Texts", type="primary"):
                        summaries = []
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for idx, row in df.iterrows():
                            status_text.text(f"Processing {idx+1}/{len(df)}...")
                            text = str(row['text']) if pd.notna(row['text']) else ""
                            
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
                        st.dataframe(df[['text', 'generated_summary']])
                        
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
    
    # Tab 3: About
    with tab3:
        st.header("About This App")
        st.markdown("""
        ### 📖 Overview
        Aplikasi ini menggunakan **IndoBART-v2**, model BART yang di-pretrain khusus untuk 
        Bahasa Indonesia. Model dimuat langsung dari HuggingFace Hub tanpa perlu fine-tuning 
        tambahan.
        
        ### 🎯 Features
        - **Single Text Summarization**: Meringkas teks individual secara instan
        - **Batch Processing**: Memproses banyak teks dari file CSV
        - **Customizable Parameters**: Sesuaikan pengaturan generasi untuk berbagai kasus
        - **Long Text Support**: Otomatis chunking untuk teks yang melebihi batas token
        
        ### 🔧 Technical Details
        - **Model**: IndoBART-v2 (indobenchmark/indobart-v2)
        - **Source**: HuggingFace Hub
        - **Max Input**: Sampai 1024 tokens (configurable)
        - **Generation**: Beam search dengan parameter yang dapat dikonfigurasi
        - **Device**: Auto-detect CUDA/CPU
        
        ### 📊 Generation Parameters
        - **Num Beams**: Beam search width - nilai lebih tinggi = kualitas lebih baik
        - **Max Output Length**: Panjang maksimum output dalam token
        - **Max Input Length**: Panjang maksimum input, teks lebih panjang akan di-chunk
        - **Num Sentences**: Jumlah kalimat dalam ringkasan akhir
        
        ### 💡 Usage Tips
        - Untuk hasil terbaik, gunakan kalimat lengkap dan teks terstruktur dengan baik
        - Sesuaikan jumlah kalimat berdasarkan panjang ringkasan yang diinginkan
        - Nilai beam search lebih tinggi menghasilkan kualitas lebih baik namun lebih lambat
        - Model ini bekerja paling baik dengan teks berita dalam Bahasa Indonesia
        
        ### 📚 Model Information
        IndoBART-v2 adalah model sequence-to-sequence berbasis BART yang di-pretrain 
        pada korpus Bahasa Indonesia. Model ini cocok untuk berbagai task NLG termasuk 
        summarization, paraphrasing, dan generation.
        
        **Paper**: [IndoNLG: Benchmark and Resources for Evaluating Indonesian Natural Language Generation](https://aclanthology.org/2021.emnlp-main.699/)
        
        ### 🤝 Support
        Untuk pertanyaan atau masalah, silakan merujuk pada dokumentasi model atau hubungi tim pengembang.
        """)
        
        st.markdown("---")
        st.markdown(f"**Model:** `{MODEL_NAME}`")
        if 'device' in st.session_state:
            st.markdown(f"**Device:** `{st.session_state['device']}`")
        st.markdown("**Framework:** PyTorch + Transformers + Streamlit")

# ============================================================
# RUN APP
# ============================================================

if __name__ == "__main__":
    main()
