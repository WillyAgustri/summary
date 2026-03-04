# -*- coding: utf-8 -*-
"""
Streamlit App - Indonesian Text Summarization
Model: IndoBART-v2
"""

import streamlit as st
import torch
import types
import pandas as pd
from indobenchmark import IndoNLGTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import List
import os
import re

# Import config
try:
    from config import (
        MODEL_PATH, DEFAULT_NUM_SENTENCES, DEFAULT_MAX_OUTPUT_LENGTH,
        DEFAULT_MAX_INPUT_LENGTH, DEFAULT_NUM_BEAMS, DEFAULT_TEMPERATURE,
        PAGE_TITLE, PAGE_ICON, LAYOUT, DEVICE as CONFIG_DEVICE
    )
except ImportError:
    # Default values if config.py not found
    MODEL_PATH = "./models/indobart-v2-detik-final"
    DEFAULT_NUM_SENTENCES = 3
    DEFAULT_MAX_OUTPUT_LENGTH = 100
    DEFAULT_MAX_INPUT_LENGTH = 800
    DEFAULT_NUM_BEAMS = 4
    DEFAULT_TEMPERATURE = 1.0
    PAGE_TITLE = "Indonesian Text Summarizer"
    PAGE_ICON = "📝"
    LAYOUT = "wide"
    CONFIG_DEVICE = ""

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
def load_model_and_tokenizer(model_path: str):
    """Load model dan tokenizer dengan caching"""
    # Use config device if specified, otherwise auto-detect
    if CONFIG_DEVICE:
        device = CONFIG_DEVICE
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load tokenizer
    tokenizer = IndoNLGTokenizer.from_pretrained("indobenchmark/indobart-v2")
    
    # Patch fungsi pad
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
    
    # Load model
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
        model.eval()
        
        # Set default generation config
        model.config.max_new_tokens = 100
        model.config.early_stopping = True
        model.config.no_repeat_ngram_size = 3
        model.config.length_penalty = 1.0
        
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
                    num_beams: int = 4, temperature: float = 1.0) -> str:
    """Ringkas 1 chunk teks"""
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
            max_new_tokens=max_output_length,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
            temperature=temperature,
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
    temperature: float = 1.0,
) -> str:
    """Ringkas teks panjang dengan chunking"""
    chunks = chunk_text(text, tokenizer, max_input_length=max_input_length, stride=stride)
    summaries = []

    progress_bar = st.progress(0)
    for i, chunk in enumerate(chunks):
        summary = summarize_chunk(
            chunk, model, tokenizer, device=device,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            num_beams=num_beams,
            temperature=temperature
        )
        summaries.append(summary)
        progress_bar.progress((i + 1) / len(chunks))

    final_summary = " ".join(summaries).strip()
    sentences = final_summary.split(". ")
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
    st.markdown("**Model:** IndoBART-v2 Fine-tuned for Indonesian News Summarization")
    
    # Sidebar untuk konfigurasi
    st.sidebar.header("⚙️ Configuration")
    MODEL_PATH
    # Model path
    default_model_path = st.sidebar.text_input(
        "Model Path",
        value="./models/indobart-v2-detik-final",
        help="Path ke model fine-tuned IndoBART"
    )
    
    # Load model button
    if 'model' not in st.session_state or st.sidebar.button("🔄 Load/Reload Model"):
        with st.spinner("Loading model..."):
            model, tokenizer, device = load_model_and_tokenizer(default_model_path)
            if model is not None:
                st.session_state['model'] = model
                st.session_state['tokenizer'] = tokenizer
                st.session_state['device'] = device
                st.sidebar.success(f"✅ Model loaded on {device}")
            else:
                st.sidebar.error("❌ Failed to load model")
                return
    
    # Check if model is loaded
    if 'model' not in st.session_state:
        st.warning("⚠️ Please configure and load the model first using the sidebar.")
        st.info("💡 **Setup Instructions:**\n1. Enter the path to your trained model\n2. Click 'Load/Reload Model'\n3. Wait for the model to load")
        return
    
    # Generation parameters
    st.sidebar.header("🎛️ Generation Parameters")
    num_sentences = st.sidebar.slider("Number of Sentences", min_value=1, max_value=10, value=DEFAULT_NUM_SENTENCES)
    max_output_length = st.sidebar.slider("Max Output Length (tokens)", min_value=50, max_value=200, value=DEFAULT_MAX_OUTPUT_LENGTH)
    max_input_length = st.sidebar.slider("Max Input Length (tokens)", min_value=400, max_value=1024, value=DEFAULT_MAX_INPUT_LENGTH)
    num_beams = st.sidebar.slider("Beam Search Width", min_value=1, max_value=8, value=DEFAULT_NUM_BEAMS)
    temperature = st.sidebar.slider("Temperature", min_value=0.1, max_value=2.0, value=DEFAULT_TEMPERATURE, step=0.1)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📄 Single Text", "📊 Batch Processing", "ℹ️ About"])
    
    # Tab 1: Single Text Summarization
    with tab1:
        st.header("Single Text Summarization")
        
        # Text input
        text_input = st.text_area(
            "Enter Indonesian text to summarize:",
            height=300,
            placeholder="Paste your Indonesian news article or text here..."
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
                st.warning("⚠️ Please enter some text to summarize.")
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
                            temperature=temperature,
                        )
                        
                        st.success("✅ Summary generated!")
                        st.markdown("### 📋 Summary:")
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
        st.markdown("Upload a CSV file with a 'text' column to summarize multiple texts at once.")
        
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
                    st.error("❌ CSV file must contain a 'text' column")
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
                                        temperature=temperature,
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
        This application uses **IndoBART-v2**, a BART model fine-tuned for Indonesian text summarization.
        The model has been specifically trained on Indonesian news articles to generate concise and 
        informative summaries.
        
        ### 🎯 Features
        - **Single Text Summarization**: Summarize individual texts instantly
        - **Batch Processing**: Process multiple texts from CSV files
        - **Customizable Parameters**: Adjust generation settings for different use cases
        - **Long Text Support**: Automatic chunking for texts exceeding token limits
        
        ### 🔧 Technical Details
        - **Model**: IndoBART-v2 (indobenchmark/indobart-v2)
        - **Fine-tuning**: LoRA (Low-Rank Adaptation) for efficient training
        - **Max Input**: Up to 1024 tokens (configurable)
        - **Generation**: Beam search with configurable parameters
        
        ### 📊 Model Performance
        The model has been evaluated on Indonesian news summarization tasks with the following metrics:
        - ROUGE-1, ROUGE-2, ROUGE-L scores
        - Training on combined MC, MMC, and Detik datasets
        
        ### 💡 Usage Tips
        - For best results, use complete sentences and well-structured text
        - Adjust the number of sentences based on desired summary length
        - Higher beam search values generally produce better quality but slower generation
        - Temperature affects diversity: lower = more focused, higher = more diverse
        
        ### 🤝 Support
        For questions or issues, please refer to the model documentation or contact the development team.
        """)
        
        st.markdown("---")
        st.markdown("**Model Path:** `" + default_model_path + "`")
        if 'device' in st.session_state:
            st.markdown(f"**Device:** `{st.session_state['device']}`")

# ============================================================
# RUN APP
# ============================================================

if __name__ == "__main__":
    main()
