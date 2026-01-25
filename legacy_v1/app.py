# app.py
import streamlit as st
import pandas as pd
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True
)
logger = logging.getLogger("MediGemma_App")

def init_session_state():
    """
    Centralized State Management.
    Ensures all required session keys exist with safe default values.
    Prevents KeyErrors on page refresh (F5).
    """
    # Define your app's "Schema" here
    defaults = {
        "messages": [{"role": "assistant", "content": "System Ready..."}],
        "md_mode_active": False,  # Default to Chat Mode
        "preview_df": None,       # Default to No Data
        "vision_enabled": True    # Default setting
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    logger.info("Session State initialized successfully")

init_session_state()

from llama_index.core import VectorStoreIndex, Document, Settings

# IMPORTS FROM MODULES
from utils import init_models
from dashboard_ui import render_dashboard
from chat_ui import render_chat_interface
from data_manager import load_and_process_data

# --- CONFIG ---
st.set_page_config(page_title="Medi-Gemma Clinical CDSS", page_icon="🩺", layout="wide")
init_models()

# DATA LOADER
@st.cache_resource(show_spinner=False)
def cached_data_loader(enc_file, pat_file=None, file_signature=None):
    """
    Streamlit wrapper. Handles caching.
    """
    # Rewind the file pointers to the beginning if they were read before
    if hasattr(enc_file, 'seek'): enc_file.seek(0)
    if pat_file and hasattr(pat_file, 'seek'): pat_file.seek(0)

    return load_and_process_data(enc_file, pat_file)

# --- SIDEBAR ---
with st.sidebar:
    st.title("Medi-Gemma")
    st.divider()
    
    # Dashboard Toggle
    if 'md_mode_active' not in st.session_state: st.session_state['md_mode_active'] = False
    md_mode = st.toggle("👨‍⚕️ Medical Director Dashboard", value=st.session_state['md_mode_active'])
    st.session_state['md_mode_active'] = md_mode
    
    st.divider()
    st.write("### Configuration")
    enc_upload = st.file_uploader("Encounters CSV", type=["csv"])
    pat_upload = st.file_uploader("Patients CSV", type=["csv"])
    
    enable_vision = st.checkbox("Enable LLaVA-Medical-Director (Active)", value=True)
    
    if st.button("🔄 Reset System"):
        st.cache_resource.clear()
        st.rerun()

# --- MAIN LOGIC ---
index = None
df = None
if enc_upload:
    # 1. Create Signature (for Caching)
    # We use name + size + ID to ensure uniqueness without reading the whole file yet
    file_sig = f"{enc_upload.name}_{enc_upload.size}_{enc_upload.file_id}"

    if pat_upload: 
        file_sig += f"_{pat_upload.name}_{pat_upload.size}_{pat_upload.file_id}"
        
    with st.spinner("Processing Data..."):
        # 2. PASS THE FILE OBJECT DIRECTLY (No Disk Write!)
        # We pass the uploaded object itself. Pandas reads it from memory.
        index, df = cached_data_loader(enc_upload, pat_upload, file_signature=file_sig)
        logger.info("Data processed successfully")

elif os.path.exists("encounters_synthetic_wound.csv"):
    pat_path = "patients_synthetic_wound.csv" if os.path.exists("patients_synthetic_wound.csv") else None
    with st.spinner("Loading Local Data..."):
        index, df = cached_data_loader("encounters_synthetic_wound.csv", pat_path, file_signature="local_v1")
        logger.info("Local data loaded successfully")

# Ensure session state is updated every time (even on Cache Hit)
if df is not None:
    st.session_state['preview_df'] = df
    logger.info("Session state updated successfully")

# ROUTING
# 1. CACHE SAFETY CHECK
# If we have an index (Cache Hit) but no dataframe in session state (because the cached function didn't run),
# we must force a reload or disable dashboard mode to prevent a crash.
if index and st.session_state['preview_df'] is None:
    # Option A: Just disable dashboard mode safely
    st.session_state['md_mode_active'] = False
    # Option B: (Advanced) You could try to reload the CSV here, but turning off dashboard is safer for now.

# 2. RENDER INTERFACE
if index and st.session_state['preview_df'] is not None:
    if st.session_state['md_mode_active']:
        render_dashboard(st.session_state['preview_df'])
    else:
        render_chat_interface(index, enable_vision)
else:
    # If index is None OR preview_df is None (and we fell through), show upload prompt
    if not index:
        st.warning("⚠️ Please upload patient data.")
    elif st.session_state['preview_df'] is None:
        st.info("🔄 Session refreshed. Please upload data or click 'Reset System' to restore the dashboard.")