import streamlit as st
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
import re
import time
from datetime import datetime
import sys
# Add the project root directory to Python's path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import logging
from src.engine.engine_core import (
    ClinicalRAGEngine,
    LLMEngine,
    VisionEngine,
    AnalyticsEngine,
    DataManager,
    generate_priority_report
)
from src.core.router import classify_query, QueryIntent
from src.safety.protocol_manager import ProtocolManager
from src.safety.verifier import SafetyVerifier
from src.core.orchestrator import ClinicalOrchestrator
from src.engine.test_models import (
    master_evict_with_retry,
    cleanup_python_models,
    analyze_with_gemma3,
    analyze_with_medgemma,
    analyze_with_hulumed,
    load_medgemma_27b,
    analyze_with_medgemma_4b,
    load_hulumed,
    unload_model_safely,
    register_model,
    get_registered_model,
    clear_registry,
    set_model_loading,
    get_model_loading,
    build_inference_config,
)
from src.engine.load_timer import (
    get_estimated_load_time, record_load_time,
    MODEL_KEY_HULUMED, MODEL_KEY_MEDGEMMA_27B, MODEL_KEY_MEDGEMMA_4B
)
from src.interface.progress_timer import run_timed_progress
from src.interface.copy_button import render_copy_button

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.makedirs("thesis_data", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# --- Deployment baseline configs (from config/deployment_baselines.yaml) ---
BASELINE_CONFIGS = {
    "gemma3": build_inference_config("gemma3", profile_name="default", prompt_template="clinician_v1"),
    "medgemma_27b": build_inference_config("medgemma_27b", profile_name="default", prompt_template="clinician_v1"),
    "medgemma_4b": build_inference_config("medgemma_4b", profile_name="tuned", prompt_template="clinician_v3_mg4b"),
    "hulumed": build_inference_config("hulumed", profile_name="thinking", prompt_template="structured_output"),
}

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

    orchestrator = ClinicalOrchestrator(analytics, rag, llm, dm)
    
    return llm, dm, vision, analytics, rag, proto, ver, orchestrator

llm_engine, data_manager, vision_engine, analytics_engine, rag_engine, protocol_manager, verifier, orchestrator = init_system()

if 'medgemma_loaded' not in st.session_state:
    st.session_state['medgemma_loaded'] = False
if 'hulumed_loaded' not in st.session_state:
    st.session_state['hulumed_loaded'] = False
if 'medgemma4b_loaded' not in st.session_state:
    st.session_state['medgemma4b_loaded'] = False

if 'vision_engine' not in st.session_state:
    st.session_state['vision_engine'] = vision_engine

if 'vision_history' not in st.session_state:
    st.session_state['vision_history'] = []

# --- Sync global model registry flags into this session ---
# Model objects live ONLY in the global registry (never in session_state)
# to avoid cross-session orphan references that block VRAM cleanup.
_reg_name = get_registered_model()[0]  # Only get name; do NOT hold model/processor refs
if _reg_name == "hulumed" and not st.session_state.get('hulumed_loaded'):
    st.session_state['hulumed_loaded'] = True
    st.session_state['medgemma_loaded'] = False
    st.session_state['medgemma4b_loaded'] = False
    logger.info("📋 Synced Hulu-Med flag from global registry into new session")
elif _reg_name == "medgemma" and not st.session_state.get('medgemma_loaded'):
    st.session_state['medgemma_loaded'] = True
    st.session_state['hulumed_loaded'] = False
    st.session_state['medgemma4b_loaded'] = False
    logger.info("📋 Synced MedGemma 27B flag from global registry into new session")
elif _reg_name == "mg4b" and not st.session_state.get('medgemma4b_loaded'):
    st.session_state['medgemma4b_loaded'] = True
    st.session_state['medgemma_loaded'] = False
    st.session_state['hulumed_loaded'] = False
    logger.info("📋 Synced MedGemma 4B flag from global registry into new session")

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
                    # 1. Run the Rules Engine
                    triage_report = generate_priority_report(data_manager.df)
                    
                    # 2. Create a Map (Patient ID -> Severity)
                    # Example: {'10770': 'Critical', '10771': 'Stable'}
                    status_map = {entry['Patient ID']: entry['Severity'] for entry in triage_report}
                    
                    # 3. Bake it into the DataFrame as a REAL column
                    # Now the 'Status' column contains "Critical", "Urgent", or "Stable"
                    data_manager.df['Status'] = data_manager.df['Patient_ID'].map(status_map).fillna('Stable')
                    logger.info(f"✅ Enriched DataFrame with Triage Status. Critical Count: {len([x for x in triage_report if x['Severity'] == 'Critical'])}")
                    
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

    # VRAM Management
    with st.expander("🧠 VRAM Management", expanded=False):
        if st.button("🔥 Emergency: Clear ALL Models"):
            with st.spinner("Clearing all models from VRAM..."):
                set_model_loading(None)  # Reset any stuck loading flag
                cleanup_python_models()
                success = master_evict_with_retry(required_free_gb=20.0, max_retries=10)
                if success:
                    st.success("✅ All models evicted successfully")
                    st.rerun()
                else:
                    st.error("❌ Could not fully clear VRAM. Check logs.")

# --- 4. MAIN INTERFACE ---
st.title("🏥 Medi-Gemma Director")

# SWAPPED TABS: Chat is now Tab 1
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "💬 Clinical Assistant",
    "📊 MD Dashboard",
    "Hulu-Med",
    "MedGemma 27B",
    "Gemma 3 Multimodal",
    "MedGemma-1.5 4B",
    "Benchmark",
    ])

# =========================================================
# TAB 1: CHAT & VISION (The Main Workflow)
# =========================================================
with tab1:
    st.caption("AI-Powered Clinical Decision Support (RAG + Safety Verification)")

    linked_pat_id = st.text_input("🔗 Link to Patient ID (Optional):", 
                                  placeholder="e.g., 10770", 
                                  help="Enter ID here to force the AI to use this patient's history.")
    
    # Store in session state for the Chat Engine to see later
    if linked_pat_id:
        st.session_state['linked_patient_id'] = linked_pat_id

    # --- A. AUTO-VISION HANDLER ---
    # We place the uploader here so it feels like part of the chat flow
    uploaded_file = st.file_uploader("Upload Clinical Image (Auto-Analysis)", type=["jpg", "png", "jpeg"], key="chat_uploader")

    # CHECK: Is there a file? AND Is it new?
    if uploaded_file:
        st.image(uploaded_file, width=350, caption="Analyzing Specimen...")
        
        if uploaded_file.name != st.session_state.last_analyzed_file:
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
            link_msg = f"**Linked Patient:** {linked_pat_id}" if linked_pat_id else "**Linked Patient:** None"
            context_msg = (
                f"**[SYSTEM UPDATE] Visual Analysis Log:**\n"
                f"**Target:** {uploaded_file.name} | {link_msg}\n"
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
            render_copy_button(msg["content"])

    if prompt := st.chat_input("Ask about a patient, treatment protocol, or medical query..."):
        final_prompt = prompt
        
        if 'linked_patient_id' in st.session_state and st.session_state['linked_patient_id']:
            pid = st.session_state['linked_patient_id']
            # Only prepend if user didn't type it themselves
            if pid not in prompt:
                final_prompt = f"Regarding Patient {pid}: {prompt}"

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            render_copy_button(prompt)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()

            # CHECK: Is data loaded?
            if data_manager.df is None and not st.session_state.last_analyzed_file:
                full_response = "⚠️ **System Empty:** Please upload patient data or an image."
                response_placeholder.error(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            else:
                # --- START NEW ORCHESTRATOR BLOCK ---
                with st.status("🤖 Medical Director Logic Active...", expanded=True) as status:
                    try:
                        # 1. THE DECOUPLED CALL (One line does it all)
                        raw_response, source_lbl, response_obj, intent = orchestrator.process_query(
                            final_prompt, 
                            st.session_state.messages
                        )
                        
                        st.write(f"🔀 Intent: **{str(intent).upper()}**")
                        st.write(f"⚙️ Engine: {source_lbl}")
                        
                        status.update(label="Response Generated", state="complete", expanded=False)
                        
                    except Exception as e:
                        # Graceful Error Handling
                        st.error(f"System Error: {str(e)}")
                        logger.error(f"Orchestrator Failed: {e}", exc_info=True)
                        raw_response = "⚠️ I encountered an internal error processing your request."
                        source_lbl = "System Error"
                        response_obj = None
                        intent = "ERROR"
                # --- END NEW ORCHESTRATOR BLOCK ---
                    
                # 3. SAFETY VERIFICATION
                st.write("🛡️ Verifying Output...")
                is_safe, reason = verifier.verify(raw_response)
                    
                if is_safe:
                    full_response = raw_response
                        
                else:
                    full_response = f"⚠️ **Safety Block:** {reason}"
                    logger.warning(f"Blocked response: {raw_response}")

                try:
                    # 1. Get the directory where app_main.py lives (src/interface)
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                        
                    # 2. Go up two levels to get project root
                    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
                        
                    # 3. Build the absolute path to thesis_data
                    log_dir = os.path.join(project_root, "audit_logs")
                        
                    # Ensure dir exists (safe to run every time)
                    os.makedirs(log_dir, exist_ok=True)
                        
                    # 4. Define the full file path
                    log_file = os.path.join(log_dir, "interactions.jsonl")
                    log_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "query": prompt,
                        "response": full_response,
                        "raw_response": raw_response,  # Original before safety check
                        "intent": intent if isinstance(intent, str) else str(intent),
                        "source": source_lbl,
                        "is_safe": is_safe,
                        "safety_reason": reason if not is_safe else None,
                        "has_image": bool(uploaded_file),
                        "patient_id_mentioned": bool(re.search(r'\b\d{4,6}\b', prompt))
                    }
                        
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                except Exception as e:
                    logger.error(f"Failed to log interaction: {e}")
                    
                status.update(label="Response Generated", state="complete", expanded=False)

                # Final Output
                response_placeholder.markdown(full_response)
                render_copy_button(full_response)
                st.caption(f"ℹ️ Source: {source_lbl}")
                # EXPLAINABILITY
                if response_obj and hasattr(response_obj, 'source_nodes') and response_obj.source_nodes:
                    with st.expander("🔍 Evidence Used (Explainability)", expanded=False):
                        st.caption("The AI consulted these patient records:")
                        for i, node in enumerate(response_obj.source_nodes):
                            # Try to parse metadata safely
                            meta_id = node.metadata.get('patient_id', 'Unknown')
                            meta_date = node.metadata.get('date', 'Unknown')
                            st.markdown(f"**📄 Record {i+1}:** Patient {meta_id} ({meta_date})")
                            st.text(f"...{node.text[:150]}...")
                            st.markdown("---")
            
            # Save to History
            st.session_state.messages.append({"role": "assistant", "content": full_response})

# =========================================================
# TAB 2: MD DASHBOARD (Cohort Overview & Triage)
# =========================================================
with tab2:
    st.header("📊 Medical Director Triage Console")
    
    if data_manager.df is None:
        st.info("⚠️ Upload 'encounters.csv' to generate the Triage Report.")
    else:
        # --- 1. RANKING LOGIC (The Core Requirement) ---
        # Run the rules engine to sort patients by Severity (Critical -> Urgent -> Stable)
        triage_data = generate_priority_report(data_manager.df)
        
        # Calculate Counts
        total_pat = len(data_manager.df['Patient_ID'].unique())
        critical = len([x for x in triage_data if x['Severity'] == 'Critical'])
        urgent = len([x for x in triage_data if x['Severity'] == 'Urgent'])
        stable = total_pat - critical - urgent
        
        # --- 2. STATUS BAR (Metrics) ---
        st.markdown("### 🏥 Unit Status")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Census", total_pat, "Patients")
        c2.metric("🚨 Critical Action", critical, "Requires Review", delta_color="inverse")
        c3.metric("⚠️ Urgent Watchlist", urgent, "Monitor", delta_color="off")
        c4.metric("✅ Stable", stable, "Clear")
        
        st.divider()

        # --- 3. PRIORITIZED PATIENT LIST (Ranking) ---
        st.subheader("⚡ Prioritized Patient Action List")
        
        if not triage_data:
            st.success("✅ No critical anomalies detected. Cohort is stable.")
        
        for patient in triage_data:
            # -- UI Styling based on Severity --
            if patient['Severity'] == 'Critical':
                icon = "🔴"
                expander_open = True # Auto-open critical cases
            elif patient['Severity'] == 'Urgent':
                icon = "🟠"
                expander_open = False
            else:
                icon = "🟢"
                expander_open = False
            
            # -- Clean up Text for Display --
            p_age = str(patient['Age']).replace("nan", "?")
            p_sex = str(patient['Sex']).replace("nan", "?")
            
            # -- The Card Header --
            label = f"{icon} **Patient {patient['Patient ID']}** | Status: {patient['Severity']} | Last Visit: {patient['Date']}"
            
            # -- The Card Body --
            with st.expander(label, expanded=expander_open):
                col_a, col_b = st.columns([1, 2])
                
                with col_a:
                    st.markdown("#### 👤 Demographics")
                    st.write(f"**Age/Sex:** {p_age} / {p_sex}")
                    st.write(f"**Wound Size:** {patient['Wound Size (cm)']} cm")
                    st.info(f"**History:** {patient['Comorbidities']}")
                
                with col_b:
                    st.markdown("#### ⚠️ Clinical Alerts")
                    if patient['Alerts']:
                        for alert in patient['Alerts']:
                            # FIX: Standard If/Else to prevent 'DeltaGenerator' print errors
                            if "Critical" in patient['Severity']:
                                st.error(alert)
                            else:
                                st.warning(alert)
                    else:
                        st.write("No active alerts.")
                        
                    st.markdown("**📝 Latest Clinical Note:**")
                    st.caption(f"_{patient['Latest Note']}_")

        st.divider()

        # --- 4. RAW DATA INSPECTOR (Kept from old version) ---
        with st.expander("📋 View Raw Patient Data Table", expanded=False):
            st.dataframe(data_manager.df, width=1000)

# ============================================================================
# TAB 3: MedGemma 1.5 4B 
# ============================================================================
with tab6:
    st.header("🔬 MedGemma 1.5 4B-IT")
    st.caption("Lightweight version of Google's medical imaging specialist. Optimized for efficiency.")

    # 1. Loading Logic and Interface Gating
    if st.session_state.get('medgemma4b_loaded', False):
        st.success("✅ MedGemma 4B is Ready.")
        
        # Initialize chat history for this specific model
        if 'mg4b_history' not in st.session_state:
            st.session_state['mg4b_history'] = []
            
        mg4b_file = st.file_uploader("Upload Wound Image", type=['png', 'jpg', 'jpeg'], key="mg4b_uploader")
        
        if mg4b_file:
            # Reset history if a new file is uploaded
            if st.session_state.get('last_mg4b_file') != mg4b_file.name:
                st.session_state['mg4b_history'] = []
                st.session_state['last_mg4b_file'] = mg4b_file.name

            temp_path = os.path.join("uploads", f"temp_mg4b_{mg4b_file.name}")
            with open(temp_path, "wb") as f:
                f.write(mg4b_file.getbuffer())
            
            st.image(mg4b_file, caption="Input Image", width=350)
            
            # Display history
            for msg in st.session_state['mg4b_history']:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
                    render_copy_button(msg["content"])

            # Chat input
            user_input = st.chat_input("Ask MedGemma 4B about the wound...")
            if user_input:
                st.session_state['mg4b_history'].append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.write(user_input)
                    render_copy_button(user_input)

                with st.chat_message("assistant"):
                    with st.spinner("MedGemma 4B is analyzing..."):
                        try:
                            _, _proc, _mdl = get_registered_model()
                            response = analyze_with_medgemma_4b(
                                temp_path,
                                user_input,
                                _proc,
                                _mdl,
                                config=BASELINE_CONFIGS["medgemma_4b"],
                            )
                            st.write(response)
                            render_copy_button(response)
                            st.session_state['mg4b_history'].append({"role": "assistant", "content": response})
                        except Exception as e:
                            st.error(f"Analysis Failed: {str(e)}")
        else:
            st.info("📤 Upload a wound image to begin analysis with MedGemma 4B.")
    
    elif get_model_loading() == 'mg4b':
        st.info("⏳ MedGemma 4B is loading, please wait...")

    else:
        # 2. Not Loaded State
        st.warning("⚠️ MedGemma 4B is not loaded.")

        # Check for VRAM conflicts
        if st.session_state.get('medgemma_loaded', False) or st.session_state.get('hulumed_loaded', False):
            st.info("💡 A larger model is currently in VRAM. Loading this will clear the GPU.")

        if st.button("🚀 Load MedGemma 4B", key="tab3_load_btn"):
            set_model_loading('mg4b')
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Step 1/2: Evicting other models (Nuclear Cleanup)...")
            from src.engine.test_models import load_medgemma_4b

            success = master_evict_with_retry(required_free_gb=6.0, max_retries=10)

            if not success:
                set_model_loading(None)
                st.error("❌ Critical: Could not clear VRAM properly.")
                progress_bar.empty()
                status_text.empty()
            else:
                estimated = get_estimated_load_time(MODEL_KEY_MEDGEMMA_4B)
                with run_timed_progress(progress_bar, status_text, estimated,
                                        label_prefix="Step 2/2: Loading MedGemma 4B",
                                        start_pct=40) as tracker:
                    processor, model = load_medgemma_4b()

                if processor and model:
                    register_model("mg4b", processor, model)  # Register IMMEDIATELY
                    record_load_time(MODEL_KEY_MEDGEMMA_4B, tracker.actual_elapsed)
                    progress_bar.progress(100)
                    st.session_state['medgemma4b_loaded'] = True
                    st.session_state['medgemma_loaded'] = False
                    st.session_state['hulumed_loaded'] = False

                    status_text.text("✅ MedGemma 4B loaded successfully.")
                    time.sleep(1)
                    st.rerun()
                else:
                    set_model_loading(None)
                    progress_bar.empty()
                    status_text.empty()
                    st.error("❌ Load failed. Check terminal for OOM details.")


# ============================================================================
# TAB 4: Gemma 3
# ============================================================================
with tab5:
    st.header("🧬 Gemma 3 27B Multimodal")
    st.caption("Native multimodal model from Google. General-purpose, not medical-specialized.")
    
    # Initialize chat history
    if 'gemma3_history' not in st.session_state:
        st.session_state['gemma3_history'] = []
    if 'gemma3_conversation' not in st.session_state:
        st.session_state['gemma3_conversation'] = []
    
    # Image upload
    gemma3_file = st.file_uploader(
        "Upload Wound Image", 
        type=['png', 'jpg', 'jpeg'], 
        key="gemma3_uploader",
    )
    
    # Reset history if file changes
    if gemma3_file and 'last_gemma3_file' not in st.session_state:
        st.session_state['last_gemma3_file'] = gemma3_file.name
    if gemma3_file and st.session_state.get('last_gemma3_file') != gemma3_file.name:
        st.session_state['gemma3_history'] = []
        st.session_state['gemma3_conversation'] = []
        st.session_state['last_gemma3_file'] = gemma3_file.name
    
    if gemma3_file:
        # Save temp file
        temp_path = os.path.join("uploads", f"temp_gemma3_{gemma3_file.name}")
        with open(temp_path, "wb") as f:
            f.write(gemma3_file.getbuffer())
        
        st.image(gemma3_file, caption="Input Image", width=350)
        
        # Display chat history
        for msg in st.session_state['gemma3_history']:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                render_copy_button(msg["content"])

        # Chat input
        user_input = st.chat_input("Ask Gemma 3 about the wound...")
        
        if user_input:
            if st.session_state.get('medgemma_loaded') or st.session_state.get('hulumed_loaded') or st.session_state.get('medgemma4b_loaded'):
                with st.status("🚨 VRAM Full: Evicting Python Models for Gemma 3...", expanded=True) as status:
                    from src.engine.test_models import cleanup_python_models
                    set_model_loading(None)  # Clear any loading flag
                    cleanup_python_models() # Forces MedGemma/HuluMed out of VRAM

                    # Reset all flags
                    st.session_state['medgemma_loaded'] = False
                    st.session_state['hulumed_loaded'] = False
                    st.session_state['medgemma4b_loaded'] = False

                    status.update(label="✅ GPU Ready for Ollama", state="complete")
            # Add user message
            st.session_state['gemma3_history'].append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)
                render_copy_button(user_input)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Gemma 3 is analyzing..."):
                    try:
                        response = analyze_with_gemma3(
                            temp_path,
                            user_input,
                            st.session_state['gemma3_conversation'],
                            config=BASELINE_CONFIGS["gemma3"],
                        )

                        st.write(response)
                        render_copy_button(response)
                        
                        # Update conversation for context
                        st.session_state['gemma3_conversation'].append({
                            "role": "user",
                            "content": user_input
                        })
                        st.session_state['gemma3_conversation'].append({
                            "role": "assistant",
                            "content": response
                        })
                        
                        # Save to history
                        st.session_state['gemma3_history'].append({
                            "role": "assistant",
                            "content": response
                        })
                        
                    except Exception as e:
                        st.error(f"Analysis Failed: {str(e)}")
                        st.info("Make sure you've installed Gemma 3: `ollama pull gemma3:27b`")
    else:
        st.info("📤 Upload a wound image to begin analysis with Gemma 3.")

# ============================================================================
# TAB 5: MedGemma 27B
# ============================================================================
with tab4:
    st.header("🏥 MedGemma 27B Multimodal")
    st.caption("Google's medical imaging specialist.")
    
    if st.session_state.get('medgemma_loaded', False):
        st.success("✅ MedGemma 27B is Ready.")
        # --- MedGemma Main Interface ---
        if 'medgemma_history' not in st.session_state:
            st.session_state['medgemma_history'] = []
        
        medgemma_file = st.file_uploader("Upload Wound Image", type=['png', 'jpg', 'jpeg'], key="medgemma_uploader")
        
        if medgemma_file:
            if st.session_state.get('last_medgemma_file') != medgemma_file.name:
                st.session_state['medgemma_history'] = []
                st.session_state['last_medgemma_file'] = medgemma_file.name

            temp_path = os.path.join("uploads", f"temp_medgemma_{medgemma_file.name}")
            with open(temp_path, "wb") as f:
                f.write(medgemma_file.getbuffer())
            
            st.image(medgemma_file, caption="Input Image", width=350)
            
            for msg in st.session_state['medgemma_history']:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
                    render_copy_button(msg["content"])

            user_input = st.chat_input("Ask MedGemma about the wound...")
            if user_input:
                st.session_state['medgemma_history'].append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.write(user_input)
                    render_copy_button(user_input)

                with st.chat_message("assistant"):
                    with st.spinner("MedGemma is analyzing..."):
                        try:
                            _, _proc, _mdl = get_registered_model()
                            response = analyze_with_medgemma(
                                temp_path,
                                user_input,
                                _proc,
                                _mdl,
                                config=BASELINE_CONFIGS["medgemma_27b"],
                            )
                            st.write(response)
                            render_copy_button(response)
                            st.session_state['medgemma_history'].append({"role": "assistant", "content": response})
                        except Exception as e:
                            st.error(f"Analysis Failed: {str(e)}")
        else:
            st.info("📤 Upload a wound image to begin analysis with MedGemma.")
    
    elif get_model_loading() == 'medgemma':
        st.info("⏳ MedGemma 27B is loading, please wait...")

    else:
        st.warning("⚠️ MedGemma is not loaded.")

        if st.session_state.get('hulumed_loaded', False):
            st.info("💡 Another Model is currently in VRAM. Loading MedGemma will evict it.")

        if st.button("🚀 Load MedGemma", key="tab5_load_btn"):
            set_model_loading('medgemma')
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Step 1/2: Clearing VRAM...")
            success = master_evict_with_retry(required_free_gb=18.0, max_retries=10)

            if not success:
                set_model_loading(None)
                st.error("❌ Could not free enough VRAM")
                progress_bar.empty()
                status_text.empty()
            else:
                estimated = get_estimated_load_time(MODEL_KEY_MEDGEMMA_27B)
                with run_timed_progress(progress_bar, status_text, estimated,
                                        label_prefix="Step 2/2: Loading MedGemma 27B",
                                        start_pct=40) as tracker:
                    processor, model = load_medgemma_27b()

                if processor and model:
                    register_model("medgemma", processor, model)  # Register IMMEDIATELY
                    record_load_time(MODEL_KEY_MEDGEMMA_27B, tracker.actual_elapsed)
                    progress_bar.progress(100)
                    st.session_state['medgemma_loaded'] = True
                    st.session_state['hulumed_loaded'] = False
                    st.session_state['medgemma4b_loaded'] = False
                    status_text.text("✅ MedGemma 27B loaded successfully.")
                    time.sleep(1)
                    st.rerun()
                else:
                    set_model_loading(None)
                    progress_bar.empty()
                    status_text.empty()
                    st.error("❌ Failed. Check terminal logs for details.")

# ============================================================================
# TAB 6: Hulu-Med 32B
# ============================================================================
with tab3:
    st.header("🧬 Hulu-Med 32B")
    st.caption("Qwen2.5-32B-based medical VLM")
    st.info("💡 Hulu-Med 32B is ~18GB at 4-bit.")

    # Check if Hulu-Med is already loaded
    if st.session_state.get('hulumed_loaded', False):
        st.success("✅ Hulu-Med 32B is Ready.")
        
        # --- Hulu-Med Main Interface ---
        if 'hulumed_history' not in st.session_state:
            st.session_state['hulumed_history'] = []

        hulumed_file = st.file_uploader("Upload Wound Image", type=['png', 'jpg', 'jpeg'], key="hulumed_uploader")

        if hulumed_file:
            if st.session_state.get('last_hulumed_file') != hulumed_file.name:
                st.session_state['hulumed_history'] = []
                st.session_state['last_hulumed_file'] = hulumed_file.name

            temp_path = os.path.abspath(os.path.join("uploads", f"temp_hulumed_{hulumed_file.name}"))
            with open(temp_path, "wb") as f:
                f.write(hulumed_file.getbuffer())

            st.image(hulumed_file, caption="Input Image", width=350)

            for msg in st.session_state['hulumed_history']:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
                    render_copy_button(msg["content"])

            user_input = st.chat_input("Ask Hulu-Med about the wound...")
            if user_input:
                st.session_state['hulumed_history'].append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.write(user_input)
                    render_copy_button(user_input)

                with st.chat_message("assistant"):
                    with st.spinner("Hulu-Med is analyzing..."):
                        try:
                            _, _proc, _mdl = get_registered_model()
                            response = analyze_with_hulumed(
                                temp_path,
                                user_input,
                                _proc,
                                _mdl,
                                config=BASELINE_CONFIGS["hulumed"],
                            )
                            st.write(response)
                            render_copy_button(response)
                            st.session_state['hulumed_history'].append({"role": "assistant", "content": response})
                        except Exception as e:
                            st.error(f"Analysis Failed: {str(e)}")
        else:
            st.info("📤 Upload a wound image to begin analysis with Hulu-Med.")
    
    elif get_model_loading() == 'hulumed':
        # Loading in progress (rerun during load) — show spinner only
        st.info("⏳ Hulu-Med is loading, please wait...")

    else:
        # Model not loaded - show load button
        st.warning("⚠️ Hulu-Med is not loaded.")

        # Show info if another model is currently loaded
        if st.session_state.get('medgemma_loaded', False):
            st.info("💡 Another is currently in VRAM. Loading Hulu-Med will evict it.")

        if st.button("🚀 Load Hulu-Med", key="load_hulu_btn"):
            set_model_loading('hulumed')
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Step 1/2: Evicting other models from VRAM...")
            success = master_evict_with_retry(required_free_gb=20.0, max_retries=10)

            if not success:
                set_model_loading(None)
                st.error("❌ Could not free enough VRAM")
                progress_bar.empty()
                status_text.empty()
            else:
                estimated = get_estimated_load_time(MODEL_KEY_HULUMED)
                with run_timed_progress(progress_bar, status_text, estimated,
                                        label_prefix="Step 2/2: Loading Hulu-Med",
                                        start_pct=30) as tracker:
                    processor, model = load_hulumed()

                if processor and model:
                    register_model("hulumed", processor, model)  # Register IMMEDIATELY
                    record_load_time(MODEL_KEY_HULUMED, tracker.actual_elapsed)
                    progress_bar.progress(100)
                    st.session_state['hulumed_loaded'] = True
                    st.session_state['medgemma_loaded'] = False
                    st.session_state['medgemma4b_loaded'] = False
                    status_text.text("✅ Hulu-Med 32B loaded successfully.")
                    time.sleep(1)
                    st.rerun()
                else:
                    set_model_loading(None)
                    progress_bar.empty()
                    status_text.empty()
                    st.error("❌ Failed to load Hulu-Med.")

# ============================================================================
# TAB 7: Benchmark (Placeholder)
# ============================================================================
with tab7:
    st.header("Benchmark")
    st.info(
        "**Batch Testing (Coming Soon)**\n\n"
        "This tab will support batch evaluation of vision models across different "
        "parameter profiles and prompt templates.\n\n"
        "For now, use the CLI benchmark script:\n"
        "```bash\n"
        "python scripts/run_benchmark.py --model medgemma_27b --dry-run\n"
        "```"
    )
