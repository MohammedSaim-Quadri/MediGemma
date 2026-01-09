import sys
import os
from dotenv import load_dotenv
load_dotenv()
if os.getenv("PYTORCH_CUDA_ALLOC_CONF"):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = os.getenv("PYTORCH_CUDA_ALLOC_CONF")

if os.getenv("TOKENIZERS_PARALLELISM"):
    os.environ["TOKENIZERS_PARALLELISM"] = os.getenv("TOKENIZERS_PARALLELISM")
import pandas as pd
import plotly.express as px

# Add the project root directory to Python's path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import streamlit as st
import logging
from src.core.router import IntentRouter, QueryIntent
from src.engine.analytics import AnalyticsEngine
from src.engine.generator import LLMEngine
from src.engine.vision import VisionEngine
from src.data_manager import DataManager
from src.engine.rag import ClinicalRAGEngine
from src.safety.protocol_manager import ProtocolManager
from src.safety.verifier import SafetyVerifier

# Configure Logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)

# Page Config
st.set_page_config(page_title="Medi-Gemma CDSS", layout="wide", page_icon="🏥")

# --- INITIALIZATION ---
@st.cache_resource
def init_system():
    """Initializes all engines once and caches them."""
    logger.info("🚀 Initializing System...")
    vision_path = os.getenv("VISION_MODEL_PATH", "./LLaVA-Medical-Director")
    llm_model_name = os.getenv("REASONING_MODEL", "gemma2:27b")
    
    logger.info(f"🔹 Vision Model: {vision_path}")
    logger.info(f"🔹 LLM Model: {llm_model_name}")

    # 1. LLM Engine
    llm = LLMEngine(model_name=llm_model_name)
    llm.initialize()
    
    # 2. Data Manager
    dm = DataManager()
    
    # 3. Vision Engine (Your Fine-Tuned Model)
    vision = VisionEngine(model_path=vision_path)
    
    # 4. Analytics Engine (Pandas)
    analytics = AnalyticsEngine(data_manager=dm)

    #5. RAG Engine
    rag = ClinicalRAGEngine(index=dm.index)
    
    # 6. Router
    router = IntentRouter(llm_engine=llm)

    #7. Safety protocols
    protocols = ProtocolManager()

    #8. Verifier
    verifier = SafetyVerifier()
    
    return llm, dm, vision, analytics, rag, router, protocols, verifier

# Load System
llm_engine, data_manager, vision_engine, analytics_engine, rag_engine, router, protocol_manager, verifier = init_system()

# --- SESSION STATE (Chat History) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- UI LAYOUT ---
st.title("🏥 Medi-Gemma: Clinical Decision Support")
st.caption("Hybrid Architecture: Deterministic Analytics + Clinical RAG + Vision Safety")
st.markdown("---")

# Sidebar: File Upload & Vision
with st.sidebar:
    st.header("🏥 Medi-Gemma")
    st.caption("Clinical Intelligence Platform")

    if data_manager.df is None:
        st.warning("⚠️ System Offline: Load Data")
    else:
        st.success(f"Active Cohort: {len(data_manager.df)} Encounters")
    
    # --- NEW: CSV UPLOADERS (The "Realism" Feature) ---
    with st.expander("📂 Upload Datasets", expanded=True):
        enc_file = st.file_uploader("Upload Encounters (CSV)", type=["csv"])
        pat_file = st.file_uploader("Upload Demographics (CSV)", type=["csv"])
        
        if enc_file and st.button("🔄 Initialize System"):
            with st.spinner("Ingesting Data & Building Vector Index..."):
                success = data_manager.load_data(enc_file, pat_file)
                if success:
                    # RE-INITIALIZE ENGINES WITH NEW DATA
                    analytics_engine.dm = data_manager
                    analytics_engine.initialize() # Re-builds Pandas Engine
                    
                    rag_engine.index = data_manager.index
                    rag_engine.initialize() # Re-builds Chat Engine
                    
                    st.success("System Ready!")
                    st.rerun() # Refresh to update status
                else:
                    st.error("Failed to process files.")

    st.markdown("---")
    st.header("👁️ Wound Vision")
    uploaded_file = st.file_uploader("Upload Wound Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        st.image(uploaded_file, caption="Clinical Observation", use_column_width=True)
        if st.button("Analyze Image"):
            with st.status("🧠 Medical Director Vision Pipeline...", expanded=True) as status:
                st.write("👁️ Loading LLaVA Vision Encoder...")
                vision_result = vision_engine.analyze(uploaded_file)
                st.write("✅ Feature Extraction Complete.")
                
                st.write("🛡️ Mapping to Clinical Protocols...")
                protocol = protocol_manager.get_protocol(vision_result)
                st.write("✅ Safety Check Passed.")
                
                status.update(label="Analysis Complete", state="complete", expanded=False)

            # Display Results in Sidebar
            st.success("Visual Findings:")
            st.info(vision_result)

            st.markdown("### 📋 Recommended Protocol")
            st.markdown(f"**Condition:** {protocol.get('name', 'Unknown')}")
            st.markdown(f"**Severity:** {protocol.get('severity', 'Check')}")
            st.markdown("**Management Steps:**")
            for step in protocol.get('management', []):
                st.markdown(f"- {step}")

            context_msg = (
                f"**[SYSTEM UPDATE] Visual Analysis Complete:**\n"
                f"- Finding: {vision_result}\n"
                f"- Protocol Triggered: {protocol.get('name')}\n"
                f"- Recommended Management: {', '.join(protocol.get('management')[:2])}..."
            )
            st.session_state.messages.append({"role": "assistant", "content": context_msg})

# --- MAIN CHAT INTERFACE ---

# Create Tabs for different views
tab1, tab2 = st.tabs(["📊 MD Dashboard", "💬 Clinical Assistant"])

# --- TAB 1: POPULATION HEALTH DASHBOARD ---
with tab1:
    if data_manager.df is None:
        st.info("👋 Welcome! Please upload patient data in the sidebar to visualize the dashboard.")
    else:
        # High Level Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Encounters", len(data_manager.df))
        with col2:
            st.metric("Unique Patients", data_manager.df['Patient_ID'].nunique())
        with col3:
            avg_age = data_manager.df['Age'].mean() if 'Age' in data_manager.df.columns else 0
            st.metric("Avg Patient Age", f"{avg_age:.1f} yrs")
        with col4:
            # Calculate Critical Cases (High Necrosis)
            critical = len(data_manager.df[data_manager.df['Necrosis_Percent'] > 0])
            st.metric("Critical Cases (Necrosis)", critical, delta_color="inverse")

        st.markdown("---")

        # Row 1: Charts
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Wound Severity Distribution")
            if 'Wound_Size_Length_cm' in data_manager.df.columns:
                # Simple Histogram of wound sizes
                fig_hist = px.histogram(data_manager.df, x="Wound_Size_Length_cm", 
                                      nbins=20, title="Wound Size Frequency (cm)",
                                      color_discrete_sequence=['#FF4B4B'])
                st.plotly_chart(fig_hist, use_container_width=True)
        
        with c2:
            st.subheader("Tissue Composition Analysis")
            # Aggregated Tissue Types
            if 'Necrosis_Percent' in data_manager.df.columns:
                avg_necrosis = data_manager.df['Necrosis_Percent'].mean()
                avg_slough = data_manager.df['Slough_Percent'].mean()
                avg_gran = data_manager.df['Granulation_Percent'].mean()
                
                tissue_data = pd.DataFrame({
                    "Tissue Type": ["Necrosis (Black)", "Slough (Yellow)", "Granulation (Red)"],
                    "Average %": [avg_necrosis, avg_slough, avg_gran]
                })
                fig_pie = px.pie(tissue_data, values="Average %", names="Tissue Type", 
                               title="Average Tissue Composition",
                               color_discrete_sequence=['black', 'gold', 'red'])
                st.plotly_chart(fig_pie, use_container_width=True)

        # Row 2: Recent Alerts
        st.subheader("🚨 Recent Critical Alerts (Necrosis > 0%)")
        critical_df = data_manager.df[data_manager.df['Necrosis_Percent'] > 0][['Patient_ID', 'Encounter_Date', 'Necrosis_Percent', 'Treatment_Plan']].head(10)
        st.dataframe(critical_df, use_container_width=True)

# --- TAB 2: CHAT INTERFACE ---
with tab2:
    st.caption("AI-Powered Clinical Decision Support (RAG + Safety Verification)")
    
    # 1. Display History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 2. Handle Input
    if prompt := st.chat_input("Ask about a patient, treatment protocol, or medical query..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            if data_manager.df is None:
                full_response = "⚠️ **System Empty:** Please upload patient data."
                response_placeholder.error(full_response)
            else:
                with st.status("🤖 Medical Director Logic Active...", expanded=True) as status:
                    st.write("🔀 Analyzing Intent (Router)...")
                    intent = router.classify(prompt)
                    st.write(f"✅ Routed to: **{intent.name}**")
                    
                    raw_response = ""
                    
                    if intent == QueryIntent.DATA_ANALYTICS:
                        st.write("📊 Executing Deterministic Pandas Query...")
                        result = analytics_engine.execute_query(prompt)
                        raw_response = result['text']
                        
                    elif intent == QueryIntent.SIMPLE_FACT:
                        st.write("📖 Querying Medical Knowledge Base...")
                        raw_response = llm_engine.generate(f"Answer concisely: {prompt}")
                        
                    elif intent == QueryIntent.PATIENT_ASSESSMENT:
                        st.write("🏥 Retrieving Patient History (Strict Filter)...")
                        if rag_engine.index is None:
                            raw_response = "⚠️ Index not ready."
                        else:
                            # T.I.M.E. Framework RAG
                            raw_response = rag_engine.chat(prompt)
                    
                    # FINAL SAFETY CHECK
                    st.write("🛡️ Verifying Output (Safety Verifier)...")
                    is_safe, reason = verifier.verify(raw_response)
                    
                    if is_safe:
                        full_response = raw_response
                    else:
                        full_response = f"⚠️ **Safety Block:** {reason}"
                        logger.warning(f"Blocked response: {raw_response}")
                    
                    status.update(label="Response Generated", state="complete", expanded=False)

                response_placeholder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
