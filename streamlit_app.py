# -*- coding: utf-8 -*-
"""
Streamlit App - Indonesian Text Summarization
Model: IndoBART-v2 + LoRA Fine-tuning
(Pattern based on copy_dari_09.py)
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
MODEL_NAME = "indobenchmark/indobart-v2"  # Base model from HuggingFace
CHECKPOINT_PATH = "outputs/indobart-v2-detik/checkpoint-800"  # Path to fine-tuned checkpoint
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-id-en"  # Indonesian-English translation model
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
    """Extract article text from news URL (same as copy_dari_09.py)"""
    try:
        # Set headers to avoid blocking
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Fetch page
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
            title = "No Title"
        
        # --- Auto-extract Date from content ---
        date = None
        # Try Indonesian date pattern: "5 Desember 2024", "01 Januari 2025"
        match = re.search(r"\d{1,2}\s+\w+\s+\d{4}", soup.get_text())
        if match:
            date = match.group(0)
        
        # --- Extract article paragraphs ---
        article_text = ""
        
        # Method 1: Find common container for news
        container = soup.find('div', class_='single-content') or soup.find('div', class_='entry-content')
        
        # Method 2: Find article tag
        if not container:
            container = soup.find('article')
        
        # Method 3: Find div with common class for article content
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
                # Clean common junk text (watermark, copyright, etc.)
                text = re.sub(r"\b(Dibaca|Foto|©|All rights reserved|MTD|WF|DINAS KOMUNIKASI.*)\b.*", "", text)
                text = re.sub(r"\b(Baca juga|Berita terkait|Simak video|ADVERTISEMENT|Halaman selanjutnya)\b.*", "", text, flags=re.IGNORECASE)
                text = text.strip()
                if text:
                    clean_paragraphs.append(text)
        
        article_text = " ".join(clean_paragraphs)
        
        # Normalize spaces
        article_text = re.sub(r'\s+', ' ', article_text).strip()
        
        if not article_text:
            return None, "Cannot find article content at the URL."
        
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
# CUSTOM CSS FOR MODERN UI
# ============================================================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main App Styling - Clean Light Theme */
    .stApp {
        background: #f8fafc;
    }
    
    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }
    
    /* Typography - Better Readability */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #1e293b !important;
        font-weight: 600 !important;
    }
    
    p, div, label, span, li {
        color: #475569 !important;
        line-height: 1.7 !important;
    }
    
    /* Title styling - Bootstrap inspired */
    h1 {
        color: #0f172a !important;
        font-weight: 700 !important;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem !important;
        letter-spacing: -0.02em !important;
    }
    
    /* Tabs styling - Clean Tailwind style */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #f1f5f9;
        padding: 4px;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: transparent;
        border-radius: 8px;
        color: #64748b;
        font-weight: 500;
        font-size: 0.95rem;
        padding: 0 20px;
        border: none;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e2e8f0;
        color: #334155;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1), 0 1px 2px rgba(0, 0, 0, 0.06) !important;
        font-weight: 600 !important;
    }
    
    /* Input fields - Clean and readable */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border-radius: 8px !important;
        border: 1.5px solid #cbd5e1 !important;
        padding: 10px 14px !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease !important;
        background-color: #ffffff !important;
        color: #1e293b !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
        outline: none !important;
    }
    
    /* Input labels */
    .stTextInput > label, .stTextArea > label {
        color: #334155 !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Buttons - Bootstrap primary style */
    .stButton > button {
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        border: none !important;
        transition: all 0.2s ease !important;
        height: 44px !important;
    }
    
    .stButton > button[kind="primary"] {
        background-color: #3b82f6 !important;
        color: white !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #2563eb !important;
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.25) !important;
        transform: translateY(-1px) !important;
    }
    
    .stButton > button[kind="secondary"] {
        background-color: #e2e8f0 !important;
        color: #334155 !important;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background-color: #cbd5e1 !important;
    }
    
    /* Alert boxes - Clear and visible */
    .stAlert {
        border-radius: 10px !important;
        border-left: 4px solid !important;
        padding: 12px 16px !important;
        background-color: #ffffff !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Success alert */
    div[data-baseweb="notification"][kind="success"] {
        background-color: #f0fdf4 !important;
        border-left-color: #22c55e !important;
    }
    
    /* Info alert */
    div[data-baseweb="notification"][kind="info"] {
        background-color: #eff6ff !important;
        border-left-color: #3b82f6 !important;
    }
    
    /* Warning alert */
    div[data-baseweb="notification"][kind="warning"] {
        background-color: #fffbeb !important;
        border-left-color: #f59e0b !important;
    }
    
    /* Error alert */
    div[data-baseweb="notification"][kind="error"] {
        background-color: #fef2f2 !important;
        border-left-color: #ef4444 !important;
    }
    
    /* Metrics - Clean numbers */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #0f172a !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-weight: 500 !important;
        color: #64748b !important;
        font-size: 0.875rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    
    /* Checkbox - Cleaner style */
    .stCheckbox {
        padding: 8px 0;
    }
    
    .stCheckbox > label {
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        color: #334155 !important;
    }
    
    /* Radio buttons */
    .stRadio > label {
        font-weight: 600 !important;
        font-size: 1rem !important;
        color: #1e293b !important;
        margin-bottom: 12px !important;
    }
    
    .stRadio [role="radiogroup"] label {
        background-color: #ffffff !important;
        border: 1.5px solid #e2e8f0 !important;
        border-radius: 8px !important;
        padding: 10px 16px !important;
        margin-right: 8px !important;
        transition: all 0.2s ease !important;
    }
    
    .stRadio [role="radiogroup"] label:hover {
        border-color: #cbd5e1 !important;
        background-color: #f8fafc !important;
    }
    
    /* Slider - Blue theme */
    .stSlider [role="slider"] {
        background-color: #3b82f6 !important;
    }
    
    .stSlider [data-baseweb="slider"] > div > div {
        background-color: #cbd5e1 !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        border-radius: 10px !important;
        border: 2px dashed #cbd5e1 !important;
        padding: 32px 24px !important;
        background-color: #ffffff !important;
        transition: all 0.2s ease !important;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #3b82f6 !important;
        background-color: #f8fafc !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #3b82f6 !important;
        border-radius: 10px !important;
    }
    
    /* Sidebar - Clean white */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #0f172a !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown div {
        color: #475569 !important;
    }
    
    /* Dataframe - Clean table */
    .stDataFrame {
        border-radius: 10px !important;
        overflow: hidden !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
        border: 1px solid #e2e8f0 !important;
    }
    
    /* Custom card styling - Bootstrap card inspired */
    .card {
        background: white;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        margin: 16px 0;
        border: 1px solid #e2e8f0;
        transition: all 0.2s ease;
    }
    
    .card:hover {
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .card-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 12px;
    }
    
    .card-text {
        color: #475569;
        line-height: 1.6;
        margin: 0;
    }
    
    /* Badge/Tag style */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 500;
        margin-right: 8px;
    }
    
    .badge-blue {
        background-color: #dbeafe;
        color: #1e40af;
    }
    
    .badge-green {
        background-color: #dcfce7;
        color: #166534;
    }
    
    .badge-purple {
        background-color: #f3e8ff;
        color: #6b21a8;
    }
    
    /* Section divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background-color: #e2e8f0;
    }
    
    /* Info box with icon */
    .info-box {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border-left: 4px solid #3b82f6;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 16px 0;
    }
    
    .info-box p {
        margin: 0;
        color: #1e40af !important;
        font-weight: 500;
    }
    
    /* Success box */
    .success-box {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-left: 4px solid #22c55e;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 16px 0;
    }
    
    .success-box p {
        margin: 0;
        color: #166534 !important;
        font-weight: 500;
    }
    
    /* Result card with border */
    .result-card {
        background: white;
        border: 2px solid #e2e8f0;
        border-radius: 10px;
        padding: 20px;
        margin: 12px 0;
        transition: all 0.2s ease;
    }
    
    .result-card:hover {
        border-color: #cbd5e1;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Caption/helper text */
    .helper-text {
        color: #64748b !important;
        font-size: 0.875rem !important;
        margin-top: 4px !important;
    }
    
    /* Section header with underline */
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #0f172a;
        padding-bottom: 12px;
        margin-bottom: 20px;
        border-bottom: 2px solid #e2e8f0;
    }
    
    /* Improved spacing */
    .stMarkdown {
        margin-bottom: 1rem;
    }
    
    /* Link styling */
    a {
        color: #3b82f6 !important;
        text-decoration: none !important;
        font-weight: 500 !important;
        transition: color 0.2s ease !important;
    }
    
    a:hover {
        color: #2563eb !important;
        text-decoration: underline !important;
    }
    
    /* Code blocks */
    code {
        background-color: #f1f5f9 !important;
        color: #e11d48 !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
        font-size: 0.9em !important;
        font-family: 'Courier New', monospace !important;
    }
</style>
""", unsafe_allow_html=True)

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
    """Load IndoBART-v2 + LoRA fine-tuned model and tokenizer with caching"""
    # Auto-detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load tokenizer from HuggingFace
    tokenizer = IndoNLGTokenizer.from_pretrained(MODEL_NAME)
    
    # Patch pad function for compatibility (from copy_dari_09.py)
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
        # Load base model from HuggingFace
        base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
        
        # Setup pad token if not available
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                base_model.resize_token_embeddings(len(tokenizer))
        
        # Load LoRA adapter from checkpoint (if available)
        if os.path.exists(CHECKPOINT_PATH):
            model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
            model = model.merge_and_unload()  # Merge LoRA weights to base model
            st.sidebar.info(f"✅ Loaded fine-tuned model from {CHECKPOINT_PATH}")
        else:
            st.sidebar.warning(f"⚠️ Checkpoint not found: {CHECKPOINT_PATH}")
            st.sidebar.info("Using base model without fine-tuning")
            model = base_model
        
        model = model.to(device)
        model.eval()
        
        # Set generation config (same as copy_dari_09.py)
        model.generation_config = GenerationConfig(
            do_sample=False,              # disable sampling for more deterministic results
            num_beams=4,                  # beam search = significantly better summary quality
            top_p=0.9,
            temperature=0.8,
            top_k=40,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            length_penalty=2.0,           # length penalty for more compact summary
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
    """Split long text into multiple chunks"""
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
    """Summarize 1 text chunk (same as copy_dari_09.py)"""
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
            # Parameters below will override generation_config if set
            # Use default generation_config unless override is needed
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
) -> str:
    """Summarize long text with chunking (abstractive summary)"""
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

    # --------- SENTENCE NORMALIZATION ---------
    text = final_summary.replace("\n", " ")
    text = " ".join(text.split())   # remove extra spaces

    # --------- SPLIT SENTENCES ---------
    sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 0]

    # --------- TRUNCATE IF TOO LONG (Advanced Truncation) ---------
    if num_sentences == 1 and len(sentences) > 0:
        # Take only the first clause from the first sentence for 1-sentence summary
        first = sentences[0]

        # If sentence is too long, cut at first comma
        if "," in first and len(first) > 110:
            first = first.split(",")[0]

        # Take maximum 18-22 words to keep 1 compact sentence
        words = first.split()
        if len(words) > 22:
            first = " ".join(words[:22])

        final_summary = first
    else:
        # For multi-sentence, take first N sentences
        if len(sentences) > num_sentences:
            sentences = sentences[:num_sentences]
        final_summary = ". ".join(sentences).strip()

    # --------- ADD PERIOD AT THE END ---------
    if not final_summary.endswith("."):
        final_summary += "."

    return final_summary

def extractive_summary(text: str, num_sentences: int = 3) -> str:
    """
    Extractive summarization: Select key sentences from the original text.
    This is lead-like summarization (copy-paste sentences from the article).
    """
    # Normalize text
    text = text.replace("\n", " ")
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]  # Filter very short sentences
    
    if len(sentences) == 0:
        return "No valid sentences found."
    
    # For extractive summary, take the first N sentences (lead-based approach)
    # This is typical for news articles where key information is at the beginning
    selected_sentences = sentences[:min(num_sentences, len(sentences))]
    
    # Join sentences
    result = " ".join(selected_sentences)
    
    # Ensure it ends with period
    if not result.endswith(('.', '!', '?')):
        result += "."
    
    return result

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
    st.markdown("<h1 style='text-align: center; color: #0f172a;'>📝 Indonesian Text Summarizer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748b; font-size: 1.1rem; margin-bottom: 1.5rem;'>AI-Powered Summarization with IndoBART-v2 & Translation</p>", unsafe_allow_html=True)
    
    # Feature highlight banner
    st.markdown("""
    <div class="info-box" style="text-align: center; margin-bottom: 2rem;">
        <p style="margin: 0; font-size: 0.95rem;">
            <span class="badge badge-blue">🎯 Abstractive & Extractive</span>
            <span class="badge badge-green">🌐 Translation</span>
            <span class="badge badge-purple">📊 Batch Processing</span>
            <span class="badge badge-blue">🔗 URL Extraction</span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model info cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="card" style="text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 8px;">🤖</div>
            <div class="card-title" style="font-size: 1rem; margin-bottom: 4px;">Model</div>
            <p class="card-text" style="font-size: 0.9rem; margin: 0;">IndoBART-v2</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card" style="text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 8px;">⚡</div>
            <div class="card-title" style="font-size: 1rem; margin-bottom: 4px;">Fine-tuned</div>
            <p class="card-text" style="font-size: 0.9rem; margin: 0;">LoRA/PEFT</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="card" style="text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 8px;">🌐</div>
            <div class="card-title" style="font-size: 1rem; margin-bottom: 4px;">Translation</div>
            <p class="card-text" style="font-size: 0.9rem; margin: 0;">Helsinki-NLP</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Sidebar for configuration
    st.sidebar.markdown("<h2 style='text-align: center; margin-bottom: 1.5rem; color: #0f172a;'>⚙️ Settings</h2>", unsafe_allow_html=True)
    
    # Load model automatically
    if 'model' not in st.session_state:
        with st.spinner("Loading summarization model..."):
            model, tokenizer, device = load_model_and_tokenizer()
            if model is not None:
                st.session_state['model'] = model
                st.session_state['tokenizer'] = tokenizer
                st.session_state['device'] = device
                st.sidebar.markdown(f"<div class='success-box' style='text-align: center;'>✅ Model loaded on {device}</div>", unsafe_allow_html=True)
            else:
                st.sidebar.error("❌ Failed to load model")
                st.error("⚠️ Failed to load model. Please refresh the page.")
                return
    else:
        st.sidebar.markdown(f"<div class='success-box' style='text-align: center;'>✅ Model ready on {st.session_state['device']}</div>", unsafe_allow_html=True)
    
    # Load translation model (lazy loading)
    if 'translation_model' not in st.session_state:
        st.session_state['translation_model'] = None
        st.session_state['translation_tokenizer'] = None
        st.session_state['translation_loaded'] = False
    
    # Generation parameters
    st.sidebar.markdown("<h3 style='margin-top: 2rem; margin-bottom: 1rem; color: #0f172a;'> 🎛️ Generation Parameters</h3>", unsafe_allow_html=True)
    num_sentences = st.sidebar.slider(
        "Number of Sentences", 
        min_value=1, max_value=10, 
        value=DEFAULT_NUM_SENTENCES,
        help="Number of sentences in the summary"
    )
    max_output_length = st.sidebar.slider(
        "Max Output Length (tokens)", 
        min_value=50, max_value=200, 
        value=MAX_OUTPUT_LEN,
        help="Maximum output length in tokens"
    )
    max_input_length = st.sidebar.slider(
        "Max Input Length (tokens)", 
        min_value=400, max_value=1024, 
        value=MAX_INPUT_LEN,
        help="Maximum input length in tokens"
    )
    num_beams = st.sidebar.slider(
        "Beam Search Width", 
        min_value=1, max_value=8, 
        value=DEFAULT_NUM_BEAMS,
        help="Beam search: higher value = better quality but slower"
    )
    
    # Info model
    st.sidebar.markdown("---")
    st.sidebar.markdown("<h3 style='margin-bottom: 1rem; color: #0f172a;'>📊 Model Info</h3>", unsafe_allow_html=True)
    st.sidebar.markdown(f"**Source:** HuggingFace Hub")
    st.sidebar.markdown(f"**Model ID:** `{MODEL_NAME}`")
    st.sidebar.markdown(f"**Device:** `{st.session_state['device']}`")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Single Text", "📊 Batch Processing", "🌐 Translation", "ℹ️ About"])
    
    # Tab 1: Single Text Summarization
    with tab1:
        st.markdown("<div class='section-title'>📄 Single Text Summarization</div>", unsafe_allow_html=True)
        
        # Pilihan input method
        input_method = st.radio(
            "Choose input method:",
            ["📝 Manual Text", "🔗 News URL"],
            horizontal=True
        )
        
        text_input = ""
        article_title = ""
        article_date = ""
        article_url = ""
        
        if input_method == "📝 Manual Text":
            # Title, Date and URL inputs
            article_title = st.text_input(
                "Judul Berita:",
                placeholder="Masukkan judul berita...",
                key="manual_title"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                article_date = st.text_input(
                    "Tanggal Berita:",
                    placeholder="01/01/2024",
                    key="manual_date"
                )
            with col2:
                article_url = st.text_input(
                    "URL Berita:",
                    placeholder="https://www.detik.com/...",
                    key="manual_url"
                )
            
            # Text input manual
            text_input = st.text_area(
                "Tempelkan teks berita disini:",
                height=250,
                placeholder="Tempelkan teks berita disini...",
                key="manual_text"
            )
        else:
            # URL input
            url_input = st.text_input(
                "Enter news URL:",
                placeholder="https://www.detik.com/...",
                key="url_input"
            )
            
            if url_input:
                article_url = url_input  # Store the URL
                if st.button("📥 Fetch Article", key="fetch_btn"):
                    with st.spinner("Fetching article from URL..."):
                        result, error = extract_text_from_url(url_input)
                        if error:
                            st.error(f"❌ {error}")
                        else:
                            article_title, extracted_date, text_input = result
                            st.session_state['fetched_text'] = text_input
                            st.session_state['fetched_title'] = article_title
                            st.session_state['fetched_date'] = extracted_date
                            st.session_state['fetched_url'] = url_input  # Store URL
                            
                            if extracted_date:
                                st.success(f"✅ Article successfully fetched! (Date detected: {extracted_date})")
                            else:
                                st.success("✅ Article successfully fetched!")
            
            # Display fetched text dan info jika ada
            if 'fetched_text' in st.session_state:
                text_input = st.session_state['fetched_text']
                article_title = st.session_state.get('fetched_title', '')
                article_date = st.session_state.get('fetched_date', '')
                article_url = st.session_state.get('fetched_url', url_input)  # Get stored URL
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    if article_title:
                        st.markdown(f"**Title:** {article_title}")
                with col2:
                    if article_date:
                        st.markdown(f"**📅 Date:** {article_date}")
                
                # Optional: override tanggal jika user mau
                if not article_date:
                    article_date = st.text_input(
                        "Date not detected. Enter manually (optional):",
                        placeholder="01 January 2024",
                        key="url_date_manual"
                    )
                
                st.text_area(
                    "Fetched article text:",
                    value=text_input,
                    height=200,
                    key="display_fetched",
                    disabled=True
                )
        
        # Translation option
        translate_summary = st.checkbox(
            "🌐 Translate summaries to English",
            value=False,
            help="Automatically translate all summaries to English (requires translation model)"
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
                st.warning("⚠️ Please enter text or fetch article from URL first.")
            else:
                # Check if translation is needed and model not loaded
                if translate_summary and not st.session_state['translation_loaded']:
                    with st.spinner("Loading translation model..."):
                        trans_model, trans_tokenizer, trans_device = load_translation_model()
                        if trans_model is not None:
                            st.session_state['translation_model'] = trans_model
                            st.session_state['translation_tokenizer'] = trans_tokenizer
                            st.session_state['translation_device'] = trans_device
                            st.session_state['translation_loaded'] = True
                        else:
                            st.error("❌ Failed to load translation model")
                            translate_summary = False
                
                with st.spinner("Generating summaries..."):
                    try:
                        # Generate Abstractive 1 sentence
                        summary_1s = summarize_long_text(
                            text_input,
                            st.session_state['model'],
                            st.session_state['tokenizer'],
                            device=st.session_state['device'],
                            max_input_length=max_input_length,
                            max_output_length=max_output_length,
                            num_sentences=1,
                            num_beams=num_beams,
                        )
                        
                        # Generate Abstractive 3 sentences
                        summary_3s = summarize_long_text(
                            text_input,
                            st.session_state['model'],
                            st.session_state['tokenizer'],
                            device=st.session_state['device'],
                            max_input_length=max_input_length,
                            max_output_length=max_output_length,
                            num_sentences=3,
                            num_beams=num_beams,
                        )
                        
                        # Generate Extractive summary
                        summary_extractive = extractive_summary(text_input, num_sentences=3)
                        
                        st.success("✅ Summaries generated!")
                        
                        # Translate if option is checked
                        if translate_summary and st.session_state['translation_loaded']:
                            with st.spinner("Translating summaries to English..."):
                                try:
                                    translated_1s = translate_text(
                                        summary_1s,
                                        st.session_state['translation_model'],
                                        st.session_state['translation_tokenizer'],
                                        device=st.session_state['translation_device']
                                    )
                                    
                                    translated_3s = translate_text(
                                        summary_3s,
                                        st.session_state['translation_model'],
                                        st.session_state['translation_tokenizer'],
                                        device=st.session_state['translation_device']
                                    )
                                    
                                    translated_ext = translate_text(
                                        summary_extractive,
                                        st.session_state['translation_model'],
                                        st.session_state['translation_tokenizer'],
                                        device=st.session_state['translation_device']
                                    )
                                except Exception as e:
                                    st.error(f"❌ Translation error: {e}")
                                    translate_summary = False
                        
                        # Display metadata
                        st.markdown("<div class='section-title'>📰 Informasi Berita</div>", unsafe_allow_html=True)
                        
                        info_html = "<div class='card'>"
                        if article_title:
                            info_html += f"<p><strong>📌 Judul:</strong> {article_title}</p>"
                        if article_date:
                            info_html += f"<p><strong>📅 Tanggal:</strong> {article_date}</p>"
                        if article_url:
                            info_html += f"<p><strong>🔗 URL:</strong> <a href='{article_url}' target='_blank' style='color: #3b82f6;'>{article_url}</a></p>"
                        info_html += "</div>"
                        st.markdown(info_html, unsafe_allow_html=True)
                        
                        st.markdown("<div class='section-title'>📋 Hasil Ringkasan</div>", unsafe_allow_html=True)
                        
                        if translate_summary and st.session_state['translation_loaded']:
                            # Display side by side (Indonesian | English)
                            st.markdown("<h4 style='color: #0f172a; margin-top: 1.5rem; font-size: 1.1rem;'>1️⃣ Ringkasan Abstraktif 1 Kalimat</h4>", unsafe_allow_html=True)
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("<div class='result-card' style='border-left: 3px solid #3b82f6;'><p style='margin: 0; font-weight: 600; color: #3b82f6; margin-bottom: 8px; font-size: 0.85rem;'>🇮🇩 INDONESIAN</p><p style='margin: 0; color: #1e293b;'>" + summary_1s + "</p></div>", unsafe_allow_html=True)
                            with col2:
                                st.markdown("<div class='result-card' style='border-left: 3px solid #22c55e;'><p style='margin: 0; font-weight: 600; color: #22c55e; margin-bottom: 8px; font-size: 0.85rem;'>🇬🇧 ENGLISH</p><p style='margin: 0; color: #1e293b;'>" + translated_1s + "</p></div>", unsafe_allow_html=True)
                            
                            st.markdown("<h4 style='color: #0f172a; margin-top: 1.5rem; font-size: 1.1rem;'>3️⃣ Ringkasan Abstraktif 3 Kalimat</h4>", unsafe_allow_html=True)
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("<div class='result-card' style='border-left: 3px solid #3b82f6;'><p style='margin: 0; font-weight: 600; color: #3b82f6; margin-bottom: 8px; font-size: 0.85rem;'>🇮🇩 INDONESIAN</p><p style='margin: 0; color: #1e293b;'>" + summary_3s + "</p></div>", unsafe_allow_html=True)
                            with col2:
                                st.markdown("<div class='result-card' style='border-left: 3px solid #22c55e;'><p style='margin: 0; font-weight: 600; color: #22c55e; margin-bottom: 8px; font-size: 0.85rem;'>🇬🇧 ENGLISH</p><p style='margin: 0; color: #1e293b;'>" + translated_3s + "</p></div>", unsafe_allow_html=True)
                            
                            st.markdown("<h4 style='color: #0f172a; margin-top: 1.5rem; font-size: 1.1rem;'>📝 Ringkasan Ekstraktif</h4>", unsafe_allow_html=True)
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("<div class='result-card' style='border-left: 3px solid #3b82f6;'><p style='margin: 0; font-weight: 600; color: #3b82f6; margin-bottom: 8px; font-size: 0.85rem;'>🇮🇩 INDONESIAN</p><p style='margin: 0; color: #1e293b;'>" + summary_extractive + "</p></div>", unsafe_allow_html=True)
                            with col2:
                                st.markdown("<div class='result-card' style='border-left: 3px solid #22c55e;'><p style='margin: 0; font-weight: 600; color: #22c55e; margin-bottom: 8px; font-size: 0.85rem;'>🇬🇧 ENGLISH</p><p style='margin: 0; color: #1e293b;'>" + translated_ext + "</p></div>", unsafe_allow_html=True)
                        else:
                            # Display Indonesian only
                            st.markdown("<h4 style='color: #0f172a; margin-top: 1.5rem; font-size: 1.1rem;'>1️⃣ Ringkasan Abstraktif 1 Kalimat</h4>", unsafe_allow_html=True)
                            st.markdown("<div class='result-card'><p style='margin: 0; color: #1e293b;'>" + summary_1s + "</p></div>", unsafe_allow_html=True)
                            
                            st.markdown("<h4 style='color: #0f172a; margin-top: 1.5rem; font-size: 1.1rem;'>3️⃣ Ringkasan Abstraktif 3 Kalimat</h4>", unsafe_allow_html=True)
                            st.markdown("<div class='result-card'><p style='margin: 0; color: #1e293b;'>" + summary_3s + "</p></div>", unsafe_allow_html=True)
                            
                            st.markdown("<h4 style='color: #0f172a; margin-top: 1.5rem; font-size: 1.1rem;'>📝 Ringkasan Ekstraktif</h4>", unsafe_allow_html=True)
                            st.markdown("<div class='result-card'><p style='margin: 0; color: #1e293b;'>" + summary_extractive + "</p></div>", unsafe_allow_html=True)
                        
                        # Statistics
                        st.markdown("<div class='section-title'>📊 Statistics</div>", unsafe_allow_html=True)
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Original Words", len(text_input.split()))
                        with col2:
                            st.metric("Abstractive 1S", f"{len(summary_1s.split())} words")
                        with col3:
                            st.metric("Abstractive 3S", f"{len(summary_3s.split())} words")
                        with col4:
                            st.metric("Extractive", f"{len(summary_extractive.split())} words")
                            
                    except Exception as e:
                        st.error(f"❌ Error during summarization: {e}")
    
    # Tab 2: Batch Processing
    with tab2:
        st.markdown("<div class='section-title'>📊 Batch Processing</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-box'>
            <h4 style='color: #0f172a; margin-bottom: 16px; font-size: 1.1rem;'>📤 Upload CSV File</h4>
            <p style='color: #475569; margin-bottom: 16px; line-height: 1.6;'>
                Upload a CSV file to summarize multiple texts at once. 
                The system will automatically detect column names.
            </p>
            
            <h5 style='color: #0f172a; margin-bottom: 12px; font-size: 1rem;'>📋 Supported Column Formats:</h5>
            <div style='background: #ffffff; padding: 16px; border-radius: 8px; border-left: 3px solid #3b82f6;'>
                <table style='width: 100%; border-collapse: collapse;'>
                    <tr style='border-bottom: 1px solid #e2e8f0;'>
                        <td style='padding: 10px; color: #0f172a; font-weight: 600;'>Column Type</td>
                        <td style='padding: 10px; color: #0f172a; font-weight: 600;'>Accepted Names</td>
                        <td style='padding: 10px; color: #0f172a; font-weight: 600;'>Status</td>
                    </tr>
                    <tr style='border-bottom: 1px solid #f1f5f9;'>
                        <td style='padding: 10px; color: #475569;'><strong>Text</strong></td>
                        <td style='padding: 10px;'><code style='background: #f1f5f9; padding: 2px 6px; border-radius: 4px; color: #0f172a;'>text</code>, <code style='background: #f1f5f9; padding: 2px 6px; border-radius: 4px; color: #0f172a;'>Isi Berita</code></td>
                        <td style='padding: 10px;'><span style='background: #fee2e2; color: #dc2626; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600;'>REQUIRED</span></td>
                    </tr>
                    <tr style='border-bottom: 1px solid #f1f5f9;'>
                        <td style='padding: 10px; color: #475569;'><strong>Title</strong></td>
                        <td style='padding: 10px;'><code style='background: #f1f5f9; padding: 2px 6px; border-radius: 4px; color: #0f172a;'>title</code>, <code style='background: #f1f5f9; padding: 2px 6px; border-radius: 4px; color: #0f172a;'>Judul</code></td>
                        <td style='padding: 10px;'><span style='background: #e0e7ff; color: #4f46e5; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600;'>OPTIONAL</span></td>
                    </tr>
                    <tr style='border-bottom: 1px solid #f1f5f9;'>
                        <td style='padding: 10px; color: #475569;'><strong>Date</strong></td>
                        <td style='padding: 10px;'><code style='background: #f1f5f9; padding: 2px 6px; border-radius: 4px; color: #0f172a;'>date</code>, <code style='background: #f1f5f9; padding: 2px 6px; border-radius: 4px; color: #0f172a;'>Tanggal</code></td>
                        <td style='padding: 10px;'><span style='background: #e0e7ff; color: #4f46e5; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600;'>OPTIONAL</span></td>
                    </tr>
                    <tr>
                        <td style='padding: 10px; color: #475569;'><strong>URL</strong></td>
                        <td style='padding: 10px;'><code style='background: #f1f5f9; padding: 2px 6px; border-radius: 4px; color: #0f172a;'>url</code>, <code style='background: #f1f5f9; padding: 2px 6px; border-radius: 4px; color: #0f172a;'>URL</code></td>
                        <td style='padding: 10px;'><span style='background: #e0e7ff; color: #4f46e5; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600;'>OPTIONAL</span></td>
                    </tr>
                </table>
            </div>
            
            <p style='color: #64748b; margin-top: 12px; font-size: 0.9rem; margin-bottom: 0;'>
                💡 <em>Column names are case-insensitive. The system will automatically detect and map them.</em>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ File loaded: {len(df)} rows")
                
                # Show preview
                st.markdown("**Preview:**")
                st.dataframe(df.head())
                
                # Flexible column detection (case-insensitive)
                def find_column(df, names):
                    """Find column by multiple possible names (case-insensitive)"""
                    df_lower = {col.lower(): col for col in df.columns}
                    for name in names:
                        if name.lower() in df_lower:
                            return df_lower[name.lower()]
                    return None
                
                # Detect columns
                text_col = find_column(df, ['text', 'Isi Berita', 'isi berita', 'isi_berita'])
                title_col = find_column(df, ['title', 'Judul', 'judul'])
                date_col = find_column(df, ['date', 'Tanggal', 'tanggal'])
                url_col = find_column(df, ['url', 'URL', 'link', 'Link'])
                
                if text_col is None:
                    st.error("❌ CSV file must have a 'text' or 'Isi Berita' column")
                else:
                    st.info(f"📝 Text column detected: **{text_col}**")
                    
                    # Check optional columns
                    has_title = title_col is not None
                    has_date = date_col is not None
                    has_url = url_col is not None
                    
                    if has_title or has_date or has_url:
                        optional_cols = []
                        if has_title:
                            optional_cols.append(f"'{title_col}'")
                        if has_date:
                            optional_cols.append(f"'{date_col}'")
                        if has_url:
                            optional_cols.append(f"'{url_col}'")
                        st.info(f"ℹ️ Optional columns found: {', '.join(optional_cols)}")
                    
                    # Translation option for batch
                    translate_batch = st.checkbox(
                        "🌐 Translate summaries to English",
                        value=False,
                        help="Automatically translate all summaries to English",
                        key="translate_batch"
                    )
                    
                    if st.button("🚀 Process All Texts", type="primary"):
                        # Load translation model if needed
                        if translate_batch and not st.session_state['translation_loaded']:
                            with st.spinner("Loading translation model..."):
                                trans_model, trans_tokenizer, trans_device = load_translation_model()
                                if trans_model is not None:
                                    st.session_state['translation_model'] = trans_model
                                    st.session_state['translation_tokenizer'] = trans_tokenizer
                                    st.session_state['translation_device'] = trans_device
                                    st.session_state['translation_loaded'] = True
                                else:
                                    st.error("❌ Failed to load translation model")
                                    translate_batch = False
                        
                        summaries_1s = []
                        summaries_3s = []
                        summaries_extractive = []
                        translated_1s = [] if translate_batch else None
                        translated_3s = [] if translate_batch else None
                        translated_ext = [] if translate_batch else None
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for idx, row in df.iterrows():
                            status_text.text(f"Processing {idx+1}/{len(df)}...")
                            text = str(row[text_col]) if pd.notna(row[text_col]) else ""
                            
                            if text.strip():
                                try:
                                    # Generate Abstractive 1 sentence
                                    summary_1s = summarize_long_text(
                                        text,
                                        st.session_state['model'],
                                        st.session_state['tokenizer'],
                                        device=st.session_state['device'],
                                        max_input_length=max_input_length,
                                        max_output_length=max_output_length,
                                        num_sentences=1,
                                        num_beams=num_beams,
                                    )
                                    summaries_1s.append(summary_1s)
                                    
                                    # Generate Abstractive 3 sentences
                                    summary_3s = summarize_long_text(
                                        text,
                                        st.session_state['model'],
                                        st.session_state['tokenizer'],
                                        device=st.session_state['device'],
                                        max_input_length=max_input_length,
                                        max_output_length=max_output_length,
                                        num_sentences=3,
                                        num_beams=num_beams,
                                    )
                                    summaries_3s.append(summary_3s)
                                    
                                    # Generate Extractive summary
                                    summary_ext = extractive_summary(text, num_sentences=3)
                                    summaries_extractive.append(summary_ext)
                                    
                                    # Translate if option is checked
                                    if translate_batch and st.session_state['translation_loaded']:
                                        try:
                                            trans_1s = translate_text(
                                                summary_1s,
                                                st.session_state['translation_model'],
                                                st.session_state['translation_tokenizer'],
                                                device=st.session_state['translation_device']
                                            )
                                            translated_1s.append(trans_1s)
                                            
                                            trans_3s = translate_text(
                                                summary_3s,
                                                st.session_state['translation_model'],
                                                st.session_state['translation_tokenizer'],
                                                device=st.session_state['translation_device']
                                            )
                                            translated_3s.append(trans_3s)
                                            
                                            trans_ext = translate_text(
                                                summary_ext,
                                                st.session_state['translation_model'],
                                                st.session_state['translation_tokenizer'],
                                                device=st.session_state['translation_device']
                                            )
                                            translated_ext.append(trans_ext)
                                        except Exception as e:
                                            translated_1s.append(f"Translation error: {str(e)}")
                                            translated_3s.append(f"Translation error: {str(e)}")
                                            translated_ext.append(f"Translation error: {str(e)}")
                                    
                                except Exception as e:
                                    summaries_1s.append(f"Error: {str(e)}")
                                    summaries_3s.append(f"Error: {str(e)}")
                                    summaries_extractive.append(f"Error: {str(e)}")
                                    if translate_batch:
                                        translated_1s.append("")
                                        translated_3s.append("")
                                        translated_ext.append("")
                            else:
                                summaries_1s.append("")
                                summaries_3s.append("")
                                summaries_extractive.append("")
                                if translate_batch:
                                    translated_1s.append("")
                                    translated_3s.append("")
                                    translated_ext.append("")
                            
                            progress_bar.progress((idx + 1) / len(df))
                        
                        # Add summaries to dataframe
                        df['ringkasan_abstraktif_1_kalimat'] = summaries_1s
                        df['ringkasan_abstraktif_3_kalimat'] = summaries_3s
                        df['ringkasan_ekstraktif'] = summaries_extractive
                        if translate_batch:
                            df['english_abstraktif_1_kalimat'] = translated_1s
                            df['english_abstraktif_3_kalimat'] = translated_3s
                            df['english_ekstraktif'] = translated_ext
                        
                        st.success("✅ All texts processed!")
                        
                        # Display columns yang relevan
                        display_cols = []
                        if has_title:
                            display_cols.append(title_col)
                        if has_date:
                            display_cols.append(date_col)
                        if has_url:
                            display_cols.append(url_col)
                        display_cols.extend([text_col, 'ringkasan_abstraktif_1_kalimat', 
                                           'ringkasan_abstraktif_3_kalimat', 'ringkasan_ekstraktif'])
                        if translate_batch:
                            display_cols.extend(['english_abstraktif_1_kalimat', 
                                               'english_abstraktif_3_kalimat', 'english_ekstraktif'])
                        
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
        st.markdown("<div class='section-title'>🌐 Translation: Indonesian → English</div>", unsafe_allow_html=True)
        st.markdown("<div class='info-box'><p style='margin: 0;'>Translate text or summary from Indonesian to English using Helsinki-NLP model</p></div>", unsafe_allow_html=True)
        
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
            
            st.info("ℹ️ Click the button above to load the translation model (~300MB)")
        else:
            st.success(f"✅ Translation model ready on {st.session_state.get('translation_device', 'cpu')}")
            
            # Translation input
            translation_input = st.text_area(
                "Enter Indonesian text:",
                height=200,
                placeholder="Example: Pemerintah mengumumkan kebijakan baru untuk meningkatkan perekonomian...",
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
                    st.warning("⚠️ Please enter text to translate.")
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
                            st.markdown("<div class='section-title'>📋 Translation Results</div>", unsafe_allow_html=True)
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("<h4 style='color: #0f172a; font-size: 1rem; margin-bottom: 8px;'>🇮🇩 Indonesian</h4>", unsafe_allow_html=True)
                                st.markdown(f"<div class='result-card' style='border-left: 3px solid #3b82f6;'><p style='margin: 0; color: #1e293b;'>{translation_input}</p></div>", unsafe_allow_html=True)
                            with col2:
                                st.markdown("<h4 style='color: #0f172a; font-size: 1rem; margin-bottom: 8px;'>🇬🇧 English</h4>", unsafe_allow_html=True)
                                st.markdown(f"<div class='result-card' style='border-left: 3px solid #22c55e;'><p style='margin: 0; color: #1e293b;'>{translated}</p></div>", unsafe_allow_html=True)
                            
                            # Statistics
                            st.markdown("<div class='section-title'>📊 Statistics</div>", unsafe_allow_html=True)
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Indonesian Words", len(translation_input.split()))
                            with col2:
                                st.metric("English Words", len(translated.split()))
                                
                        except Exception as e:
                            st.error(f"❌ Error during translation: {e}")
            
            # Quick tip
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div class='info-box'>
            <p style='margin: 0 0 8px 0; font-weight: 600; color: #1e40af;'>💡 Tips:</p>
            <ul style='margin: 0; padding-left: 20px;'>
                <li>This model is lightweight (~300MB) and accurate for news text</li>
                <li>Best for: news articles, summaries, formal text</li>
                <li>For long text (>400 words), automatically chunked</li>
                <li>Combine with summarization: summarize first, then translate!</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Tab 4: About
    with tab4:
        st.markdown("<div class='section-title'>ℹ️ About This App</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='card'>
        <h3 style='color: #0f172a; margin-bottom: 16px; font-size: 1.25rem;'>📖 Overview</h3>
        <p style='color: #475569;'>This application uses <strong>IndoBART-v2</strong> that has been <strong>fine-tuned</strong> with the 
        <strong>LoRA (Low-Rank Adaptation)</strong> technique using Indonesian local news summary datasets 
        (MC, MMC, and Detik).</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='card'>
        <h3 style='color: #0f172a; margin-bottom: 16px; font-size: 1.25rem;'>🎯 Features</h3>
        <ul style='margin-left: 20px; line-height: 1.8; color: #475569;'>
            <li><strong>Single Text Summarization</strong>: Summarize individual text instantly</li>
            <li><strong>Multiple Summary Types</strong>: Abstractive (1 & 3 sentences) and Extractive summaries</li>
            <li><strong>Batch Processing</strong>: Process multiple texts from CSV file</li>
            <li><strong>URL Extraction</strong>: Extract and summarize articles from news URLs with auto-date detection</li>
            <li><strong>Translation</strong>: Translate summaries to English with Helsinki-NLP model</li>
            <li><strong>Smart Text Cleaning</strong>: Automatically filter watermarks, copyrights, and junk text</li>
            <li><strong>Customizable Parameters</strong>: Adjust generation settings for various cases</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class='card'>
            <h3 style='color: #0f172a; margin-bottom: 16px; font-size: 1.1rem;'>🔧 Summarization</h3>
            <p style='margin-bottom: 8px; color: #475569;'><strong>Base Model:</strong> IndoBART-v2</p>
            <p style='margin-bottom: 8px; color: #475569;'><strong>Fine-tuning:</strong> LoRA/PEFT</p>
            <p style='margin-bottom: 8px; color: #475569;'><strong>Training Data:</strong> MC, MMC, Detik</p>
            <p style='margin-bottom: 8px; color: #475569;'><strong>Max Input:</strong> 800 tokens</p>
            <p style='margin-bottom: 8px; color: #475569;'><strong>Max Output:</strong> 100 tokens</p>
            <p style='margin-bottom: 0; color: #475569;'><strong>Generation:</strong> Beam search (4 beams)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='card'>
            <h3 style='color: #0f172a; margin-bottom: 16px; font-size: 1.1rem;'>🌐 Translation</h3>
            <p style='margin-bottom: 8px; color: #475569;'><strong>Model:</strong> Helsinki-NLP/opus-mt-id-en</p>
            <p style='margin-bottom: 8px; color: #475569;'><strong>Direction:</strong> Indonesian → English only</p>
            <p style='margin-bottom: 8px; color: #475569;'><strong>Size:</strong> ~300MB (lightweight)</p>
            <p style='margin-bottom: 8px; color: #475569;'><strong>Max tokens:</strong> 512 per chunk</p>
            <p style='margin-bottom: 0; color: #475569;'><strong>Auto-chunking:</strong> Support >400 words</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='card'>
        <h3 style='color: #0f172a; margin-bottom: 16px; font-size: 1.25rem;'>💡 Usage Tips</h3>
        <ul style='margin-left: 20px; line-height: 1.8; color: #475569; margin-bottom: 0;'>
            <li>Model fine-tuned specifically for Indonesian news summaries</li>
            <li><strong>URL Mode:</strong> Date automatically detected (format: "5 Desember 2024")</li>
            <li><strong>Text Cleaning:</strong> Auto-removes watermarks, copyrights, and metadata</li>
            <li><strong>Abstractive Summary:</strong> AI generates new sentences based on content</li>
            <li><strong>Extractive Summary:</strong> Copy-paste key sentences from original text (lead-based)</li>
            <li><strong>Translation:</strong> Check translation option to get English versions</li>
            <li><strong>Batch CSV:</strong> Supports multiple column formats for flexibility</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='card'>
        <h3 style='color: #0f172a; margin-bottom: 16px; font-size: 1.25rem;'>📚 Model Information</h3>
        <p style='margin-bottom: 12px; color: #475569;'><strong>Base Model:</strong> IndoBART-v2 is a BART-based sequence-to-sequence model pretrained 
        on Indonesian language corpus.</p>
        
        <p style='margin-bottom: 12px; color: #475569;'><strong>Fine-tuning Details:</strong></p>
        <ul style='margin-left: 20px; line-height: 1.8; color: #475569;'>
            <li><strong>Technique:</strong> LoRA (Low-Rank Adaptation) with PEFT</li>
            <li><strong>Dataset:</strong> Combined MC, MMC, and Detik datasets (Indonesian news)</li>
            <li><strong>Parameters:</strong> r=16, lora_alpha=32, lora_dropout=0.05</li>
            <li><strong>Target Modules:</strong> q_proj, k_proj, v_proj, o_proj, fc1, fc2</li>
            <li><strong>Training:</strong> 5 epochs with learning rate 5e-5</li>
        </ul>
        
        <p style='margin-top: 12px; color: #475569;'><strong>Paper:</strong> <a href='https://aclanthology.org/2021.emnlp-main.699/' target='_blank' style='color: #3b82f6;'>IndoNLG: Benchmark and Resources for Evaluating Indonesian Natural Language Generation</a></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Footer
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class='card' style='text-align: center;'>
            <p style='margin: 0; color: #64748b;'><strong>Base Model</strong></p>
            <p style='margin: 4px 0 0 0; color: #0f172a;'><code>{MODEL_NAME}</code></p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class='card' style='text-align: center;'>
            <p style='margin: 0; color: #64748b;'><strong>Checkpoint</strong></p>
            <p style='margin: 4px 0 0 0; color: #0f172a; font-size: 0.85rem;'><code>{CHECKPOINT_PATH}</code></p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            if 'device' in st.session_state:
                st.markdown(f"""
                <div class='card' style='text-align: center;'>
                <p style='margin: 0; color: #64748b;'><strong>Device</strong></p>
                <p style='margin: 4px 0 0 0; color: #0f172a;'><code>{st.session_state['device']}</code></p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<p style='text-align: center; color: #9ca3af; margin-top: 2rem;'>Built with PyTorch + Transformers + PEFT + Streamlit</p>", unsafe_allow_html=True)

# ============================================================
# RUN APP
# ============================================================

if __name__ == "__main__":
    main()
