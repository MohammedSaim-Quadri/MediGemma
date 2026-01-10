import streamlit as st
import os
import json
from datetime import datetime
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
from clinical_rules import generate_priority_report

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
                with st.status("🤖 Medical Director Logic Active...", expanded=True) as status:
                    # 1. DETERMINISTIC ROUTING
                    intent = classify_query(final_prompt, st.session_state.messages)
                    st.write(f"🔀 Intent Detected: **{intent.upper()}**")
                    
                    raw_response = ""
                    response_obj = None
                    source_lbl = ""
                    
                    # 2. EXECUTION
                    if intent == QueryIntent.DATA:
                        st.write("📊 Executing Pandas Analytics...")
                        source_lbl = "Analytics Engine"
                        if data_manager.df is not None:
                            res = analytics_engine.execute_query(final_prompt)
                            raw_response = res['text']
                        else:
                            raw_response = "⚠️ Analytics Unavailable: No CSV loaded."
                        
                    else: # QueryIntent.CLINICAL
                        st.write("🏥 Consulting Clinical Engine (RAG/LLM)...")
                        source_lbl = "Clinical Engine"
                        # We simply pass the prompt. The ChatEngine (RAG) uses the history (including [SYSTEM UPDATE])
                        # so no manual injection is needed anymore.
                        if rag_engine.index:
                            response_obj = rag_engine.chat(final_prompt, st.session_state.messages)
                            # Handle both object and string returns
                            if isinstance(response_obj, str):
                                raw_response = response_obj
                            else:
                                raw_response = str(response_obj)
                        else:
                            # 1. Create a clean list of dictionaries from history
                            messages_for_llm = [
                                {"role": m["role"], "content": m["content"]} 
                                for m in st.session_state.messages
                            ]
                            # 2. Add the current user prompt
                            messages_for_llm.append({"role": "user", "content": prompt})
                            
                            # 3. Get response (ChatResponse object)
                            resp = llm_engine.chat(messages_for_llm)
                            
                            # 4. Extract text safely
                            raw_response = resp.message.content
                            response_obj = None
                    
                    # 3. SAFETY VERIFICATION
                    st.write("🛡️ Verifying Output...")
                    is_safe, reason = verifier.verify(raw_response)
                    
                    if is_safe:
                        full_response = raw_response
                        
                    else:
                        full_response = f"⚠️ **Safety Block:** {reason}"
                        logger.warning(f"Blocked response: {raw_response}")

                    try:
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
                        
                        log_file = "thesis_data/interactions.jsonl"
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
            st.dataframe(data_manager.df, use_container_width=True)