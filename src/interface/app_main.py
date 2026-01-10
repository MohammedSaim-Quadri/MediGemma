import streamlit as st
import os
import sys
# Add the project root directory to Python's path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import logging
from src.data_manager import DataManager
from src.engine.analytics import AnalyticsEngine
from src.engine.rag import ClinicalRAGEngine
from src.engine.vision import VisionEngine
from src.engine.generator import LLMEngine
# NEW: Import the simple function
from src.core.router import classify_query, QueryIntent
from src.safety.protocol_manager import ProtocolManager
from src.safety.verifier import SafetyVerifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. INITIALIZATION ---
@st.cache_resource
def init_system():
    logger.info("🚀 Initializing System...")
    dm = DataManager()
    
    llm = LLMEngine()
    llm.initialize()
    vision = VisionEngine() 
    analytics = AnalyticsEngine(dm)
    rag = ClinicalRAGEngine(index=dm.index)
    
    proto = ProtocolManager()
    ver = SafetyVerifier()
    
    return llm, dm, vision, analytics, rag, proto, ver

llm_engine, data_manager, vision_engine, analytics_engine, rag_engine, protocol_manager, verifier = init_system()

# --- 2. SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Medical Director System Online. Upload data or patient images to begin."}
    ]
if "last_analyzed_file" not in st.session_state:
    st.session_state.last_analyzed_file = None

# --- 3. SIDEBAR (Config) ---
with st.sidebar:
    st.header("⚙️ Settings")
    with st.expander("📂 Clinical Data", expanded=False):
        enc_file = st.file_uploader("Encounters CSV", type=["csv"])
        pat_file = st.file_uploader("Demographics CSV", type=["csv"])
        if enc_file and st.button("🔄 Ingest Data"):
            with st.spinner("Processing..."):
                if data_manager.load_data(enc_file, pat_file):
                    analytics_engine.dm = data_manager
                    analytics_engine.initialize()
                    rag_engine.index = data_manager.index
                    rag_engine.initialize()
                    st.success("Data Ingested!")
                    st.rerun()

    if st.button("🧹 Clear Conversation"):
        st.session_state.messages = [{"role": "assistant", "content": "Memory cleared."}]
        st.session_state.last_analyzed_file = None
        st.rerun()

# --- 4. MAIN INTERFACE ---
st.title("🏥 Medi-Gemma Director")

# SWAPPED TABS: Chat is now Tab 1
tab1, tab2 = st.tabs(["💬 Clinical Assistant", "📊 MD Dashboard"])

# =========================================================
# TAB 1: CHAT & VISION (The Main Workflow)
# =========================================================
with tab1:
    st.caption("AI-Powered Clinical Decision Support (RAG + Safety Verification)")

    # --- A. AUTO-VISION HANDLER ---
    # We place the uploader here so it feels like part of the chat flow
    uploaded_file = st.file_uploader("Upload Clinical Image (Auto-Analysis)", type=["jpg", "png", "jpeg"], key="chat_uploader")

    # CHECK: Is there a file? AND Is it new?
    if uploaded_file and uploaded_file.name != st.session_state.last_analyzed_file:
        
        st.image(uploaded_file, use_column_width=True, caption="Analyzing Specimen...")
        
        # AUTO-TRIGGER (No Button)
        with st.status("🧠 Running Clinical Vision Pipeline...", expanded=True) as status:
            st.write("🔌 Orchestrating VRAM (Evicting Chat Engine)...")
            st.write("👁️ LLaVA Vision Encoder: Scanning Tissue...")
            
            # Run Analysis
            vision_result = vision_engine.analyze(uploaded_file)
            
            st.write("🛡️ Protocol Manager: Mapping Guidelines...")
            protocol = protocol_manager.get_protocol(vision_result)
            
            status.update(label="✅ Analysis Complete", state="complete", expanded=False)
        
        # Formulate Context Message
        context_msg = (
            f"**[SYSTEM UPDATE] Visual Analysis Log:**\n"
            f"**Target:** {uploaded_file.name}\n"
            f"**Visual Findings:** {vision_result}\n"
            f"**Protocol Identified:** {protocol.get('name')}\n"
            f"**Recommended Actions:** {', '.join(protocol.get('management')[:3])}..."
        )
        
        # Inject & Save State
        st.session_state.messages.append({"role": "assistant", "content": context_msg})
        st.session_state.last_analyzed_file = uploaded_file.name 
        st.rerun() # Force refresh

    # --- B. CHAT INTERFACE ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about a patient, treatment protocol, or medical query..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            
            # CHECK: Is data loaded?
            if data_manager.df is None and not st.session_state.last_analyzed_file:
                full_response = "⚠️ **System Empty:** Please upload patient data or an image."
                response_placeholder.error(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            else:
                with st.status("🤖 Medical Director Logic Active...", expanded=True) as status:
                    # 1. DETERMINISTIC ROUTING
                    intent = classify_query(prompt, st.session_state.messages)
                    st.write(f"🔀 Intent Detected: **{intent.upper()}**")
                    
                    raw_response = ""
                    source_lbl = ""
                    
                    # 2. EXECUTION
                    if intent == QueryIntent.DATA:
                        st.write("📊 Executing Pandas Analytics...")
                        source_lbl = "Analytics Engine"
                        if data_manager.df is not None:
                            res = analytics_engine.execute_query(prompt)
                            raw_response = res['text']
                        else:
                            raw_response = "⚠️ Analytics Unavailable: No CSV loaded."
                        
                    else: # QueryIntent.CLINICAL
                        st.write("🏥 Consulting Clinical Engine (RAG/LLM)...")
                        source_lbl = "Clinical Engine"
                        # We simply pass the prompt. The ChatEngine (RAG) uses the history (including [SYSTEM UPDATE])
                        # so no manual injection is needed anymore.
                        if rag_engine.index:
                            raw_response = rag_engine.chat(prompt, conversation_history=st.session_state.messages)
                        else:
                            # Fallback if no CSVs but we have an image or just generic questions
                            raw_response = llm_engine.generate(prompt)
                    
                    # 3. SAFETY VERIFICATION
                    st.write("🛡️ Verifying Output...")
                    is_safe, reason = verifier.verify(raw_response)
                    
                    if is_safe:
                        full_response = raw_response
                    else:
                        full_response = f"⚠️ **Safety Block:** {reason}"
                        logger.warning(f"Blocked response: {raw_response}")
                    
                    status.update(label="Response Generated", state="complete", expanded=False)

                # Final Output
                response_placeholder.markdown(full_response)
                st.caption(f"ℹ️ Source: {source_lbl}")
            
            # Save to History
            st.session_state.messages.append({"role": "assistant", "content": full_response})

# =========================================================
# TAB 2: MD DASHBOARD (Cohort Overview)
# =========================================================
with tab2:
    st.header("📊 Cohort Analytics Dashboard")
    
    if data_manager.df is None:
        st.info("Upload 'encounters.csv' to view the dashboard.")
    else:
        # Quick Stats Row
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Encounters", len(data_manager.df))
        col2.metric("Unique Patients", data_manager.df['Patient_ID'].nunique() if 'Patient_ID' in data_manager.df else "N/A")
        
        # Calculate Critical Cases if columns exist
        if 'Necrosis_Percent' in data_manager.df.columns:
            critical = len(data_manager.df[data_manager.df['Necrosis_Percent'] > 0])
            col3.metric("Critical Cases", critical)
        else:
            col3.metric("Protocols Active", "Standard")

        st.markdown("---")
        
        # Data Preview
        st.subheader("📋 Patient Encounters")
        st.dataframe(data_manager.df, use_container_width=True)
        
        # Visual Analytics (If plot supported)
        st.subheader("📈 Population Health")
        if 'description' in data_manager.df.columns:
             # Simple distribution of common terms
             st.bar_chart(data_manager.df['description'].value_counts().head(10))