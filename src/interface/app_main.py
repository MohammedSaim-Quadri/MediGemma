import streamlit as st
import os
import json
import re
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.makedirs("thesis_data", exist_ok=True)

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

if 'vision_engine' not in st.session_state:
    st.session_state['vision_engine'] = vision_engine

if 'vision_history' not in st.session_state:
    st.session_state['vision_history'] = []

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

# --- 4. MAIN INTERFACE ---
st.title("🏥 Medi-Gemma Director")

# SWAPPED TABS: Chat is now Tab 1
tab1, tab2, tab3 = st.tabs(["💬 Clinical Assistant", "📊 MD Dashboard", "LLaVA Test"])

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

# ==========================================
# TAB 3: VISION DIAGNOSTICS (LLaVA Lab)
# ==========================================
with tab3:
    st.header("👁️ LLaVA-Med Vision Lab")
    st.caption("Direct access to the Vision-Language Model. Focused on Image Analysis (No Patient Context).")
    
    # Initialize Vision Chat History
    if 'vision_history' not in st.session_state:
        st.session_state['vision_history'] = []

    # 1. Image Upload
    vision_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="vision_uploader")
    
    # Reset history if file changes
    if vision_file and 'last_vision_file' not in st.session_state:
            st.session_state['last_vision_file'] = vision_file.name
    if vision_file and st.session_state.get('last_vision_file') != vision_file.name:
            st.session_state['vision_history'] = []
            st.session_state['last_vision_file'] = vision_file.name

    if vision_file:
        # Save temp file
        temp_path = f"temp_vision_{vision_file.name}"
        with open(temp_path, "wb") as f:
            f.write(vision_file.getbuffer())
        
        st.image(vision_file, caption="Input Image", width=350)
        
        # 2. Controls
        if st.button("🚀 Initialize Vision Engine", key="init_vision_btn"):
            with st.spinner("Loading Vision Model (This evicts Gemma)..."):
                st.session_state['vision_engine'].load_model()
            st.success("Vision Engine Ready.")

        # 3. Chat Display
        for msg in st.session_state['vision_history']:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # 4. Chat Input
        user_input = st.chat_input("Ask LLaVA about the image...")
        
        if user_input:
            if not st.session_state['vision_engine'].loaded:
                st.error("Please click 'Initialize Vision Engine' first.")
            else:
                # A. Show User Input immediately
                st.session_state['vision_history'].append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.write(user_input)
                
                # B. Generate Response
                with st.chat_message("assistant"):
                    with st.spinner("LLaVA is analyzing..."):
                        try:
                            # --- ROBUST CONTEXT CONSTRUCTION ---
                            context_str = ""
                            if len(st.session_state['vision_history']) > 1:
                                context_str = "Context from previous turns:\n"
                                # Take last 3 turns, excluding the current one we just added
                                for m in st.session_state['vision_history'][:-1][-3:]:
                                    label = "User asked" if m['role'] == 'user' else "You answered"
                                    context_str += f"- {label}: {m['content']}\n"
                                context_str += "\n"

                            medical_instruction = (
                                "ACT AS A CLINICAL SCIENTIST. You are analyzing a wound image for a medical report.\n"
                                "You MUST provide specific estimates. Do not use vague terms like 'extensive'.\n"
                                "Fill out this form strictly:\n\n"
                                "1. COMPOSITION (Must add up to 100%): [e.g., '10%' Eschar, 40% Slough, '50%' Granulation']\n"
                                "2. DIMENSIONS: [Estimate LxW in cm. If no ruler, use body landmarks to guess. e.g., 'approx 4x3 cm']\n"
                                "3. DEPTH: [Superficial / Partial / Full Thickness]\n"
                                "4. INFECTION: [Yes/No - Look for pus, redness, swelling]\n"
                                "5. PERIWOUND: [Healthy / Macerated / Inflamed]\n"
                                f"USER QUESTION: {user_input}"
                            )
                            
                            # The final prompt sent to LLaVA
                            combined_prompt = f"{context_str}{medical_instruction}"
                            
                            # Run Analysis
                            response = st.session_state['vision_engine'].analyze(temp_path, prompt=combined_prompt)
                            
                            st.write(response)
                            
                            # Save to History
                            st.session_state['vision_history'].append({"role": "assistant", "content": response})
                            
                        except Exception as e:
                            st.error(f"Analysis Failed: {str(e)}")