# chat_ui.py
import streamlit as st
import time
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.core import Settings, get_response_synthesizer

# Imports from your other modules
from utils import extract_patient_ids, analyze_image
from clinical_rules import get_holistic_answer

def render_chat_interface(index, enable_vision):
    """
    Renders the main Chat Interface (Consult + Data Inspector tabs).
    """
    st.title("🏥 Medi-Gemma Clinical CDSS")
    tab1, tab2 = st.tabs(["💬 Consult Interface", "📊 Data Inspector"])

    with tab1:
        # 1. Initialize History
        if "messages" not in st.session_state:
            st.session_state.messages = [{
                "role": "assistant", 
                "content": "System Ready. Upload data or ask clinical questions.",
                "logs": []
            }]

        # 2. Render Chat History
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("image_desc"):
                    with st.expander("📷 Image Analysis"):
                        st.write(msg["image_desc"])
                if msg.get("logs"):
                    with st.expander("🧠 Explainability Console", expanded=False):
                        for log in msg["logs"]:
                            st.text(log)

        # 3. Image Uploader
        with st.expander("📷 Upload Wound Image (Optional)", expanded=False):
            st.caption("Multimodal Triage: Upload an image and link it to a Patient ID to combine Vision + History.")
            
            # New Input Field for Linking
            patient_id_for_image = st.text_input("Link to Patient ID (e.g., 10770):", key="img_patient_id")
            
            img_upload = st.file_uploader("Drag & drop wound image here", type=["jpg", "png", "jpeg"], key="vision_uploader")
            
            if img_upload:
                st.image(img_upload, caption="Ready for Analysis", width=200)
                if not patient_id_for_image:
                    st.warning("⚠️ Enter Patient ID above to enable Full Clinical Context")

        # 4. Chat Input & Processing
        if prompt := st.chat_input("Ask a question..."):
            
            # Add User Message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                if img_upload:
                    st.image(img_upload, width=200)

            # Generate Assistant Response
            if index:
                with st.chat_message("assistant"):
                    reasoning_expander = st.expander("🧠 Explainability Console (Live)", expanded=True)
                    log_placeholder = reasoning_expander.empty()
                    #answer_placeholder = st.empty()
                    logs = []

                    def add_log(message):
                        logs.append(message)
                        log_placeholder.text("\n".join(logs))
                        time.sleep(0.1)

                    add_log(f"[{time.strftime('%H:%M:%S')}] Query Received: '{prompt}'")
                    
                    # Variables for History/Debug
                    response_text = ""
                    response_obj = None 
                    patient_context_str = ""
                    vision_context = ""
                    vision_display_str = "" # For the UI expander
                    vision_data = None

                    # VISION ANALYSIS
                    if img_upload:
                        if enable_vision:
                            add_log(f"[{time.strftime('%H:%M:%S')}] 👁️ VISION: Activating LLaVA...")
                            vision_data = analyze_image(img_upload)

                            # ROBUST ERROR CHECKING
                            if vision_data and "error" not in vision_data:
                                # SUCCESS PATH
                                add_log(f"🧐 RAW MODEL SAID: '{vision_data.get('raw_output', 'N/A')}'")
                                
                                diagnosis = vision_data.get('diagnosis', 'Unknown')
                                protocol = vision_data.get('protocol', 'No protocol found')
                                
                                vision_context = (
                                    f"[CLINICAL IMAGE ANALYSIS]\n"
                                    f"DIAGNOSIS: {diagnosis}\n"
                                    f"APPROVED PROTOCOL:\n{protocol}"
                                )
                                
                                vision_display_str = f"**🔍 AI Diagnosis:** {diagnosis}\n\n{protocol}"
                                add_log("✅ VISION: Diagnosis & Protocol Retrieved.")
                                
                            else:
                                # ERROR PATH
                                error_msg = vision_data.get("error", "Unknown error") if vision_data else "Analysis failed"
                                recovery = vision_data.get("recovery", "unknown") if vision_data else "unknown"
                                
                                add_log(f"❌ VISION ERROR: {error_msg}")
                                
                                # Set flag so downstream code knows vision failed
                                vision_data = None
                                vision_context = f"\n\n[SYSTEM NOTE]: Vision analysis failed. Error: {error_msg}"
                        else:
                            add_log("🔒 SAFETY GATE: Vision disabled.")
                            vision_context = "\n\n[SYSTEM NOTE]: Vision Module LOCKED."

                    # --- B. LOGIC ROUTING ---
                    holistic_result = None
                    if 'preview_df' in st.session_state:
                        holistic_result = get_holistic_answer(prompt, st.session_state['preview_df'])

                    requested_patients = extract_patient_ids(prompt)

                    # PATH 1: HOLISTIC ANALYTICS
                    if holistic_result:
                        add_log("📊 ANALYTICS: Identified holistic query.")
                        response_text = holistic_result
                        st.markdown(response_text)
                    
                    # PATH 2: PATIENT SPECIFIC (Text Only)
                    elif requested_patients and not img_upload:
                        if 'preview_df' in st.session_state:
                            valid_ids = st.session_state['preview_df']['Patient_ID'].astype(str).unique()
                            found = [p for p in requested_patients if str(p) in valid_ids]
                            
                            if not found:
                                add_log("🛑 STOP: No valid records found.")
                                response_text = f"Could not find records for: {', '.join(requested_patients)}"
                            else:
                                requested_patients = found
                                add_log(f"🔍 RETRIEVAL: Looking up {', '.join(requested_patients)}")

                                from utils import get_patient_current_state
                                ground_truth_context = ""

                                if 'preview_df' in st.session_state:
                                    for pid in requested_patients:
                                        state = get_patient_current_state(pid, st.session_state['preview_df'])
                                        if state:
                                            ground_truth_context += (
                                                f"=== GROUND TRUTH: PATIENT {pid} (LATEST VISIT: {state['last_visit']}) ===\n"
                                                f"⚠️ STATUS: {state['severity']}\n"
                                                f"📏 SIZE: {state['wound_dims']}\n"
                                                f"📝 LATEST NOTE: {state['narrative']}\n"
                                                f"==========================================================\n\n"
                                            )
                                
                                # RAG Retrieval
                                filters = MetadataFilters(
                                    filters=[ExactMatchFilter(key="patient_id", value=str(pid)) for pid in requested_patients],
                                    condition="or"
                                )
                                retriever = VectorIndexRetriever(index=index, similarity_top_k=len(requested_patients)+1, filters=filters)
                                synth = get_response_synthesizer(streaming=True)
                                query_engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=synth)
                                
                                # Prompt Construction
                                full_prompt = (
                                    f"### ROLE & CONTEXT:\n"
                                    f"You are the **AI Medical Director** for a wound care clinic reviewing records for: {', '.join(requested_patients)}.\n"
                                    f"{vision_context}\n\n"
                                    
                                    f"### 🚨 CRITICAL DATA (SOURCE OF TRUTH):\n"
                                    f"{ground_truth_context}\n\n"
                                    
                                    f"### INSTRUCTIONS:\n"
                                    f"Analyze the user's request and categorize it into one of two scenarios. Adapt your response accordingly.\n\n"
                                    
                                    f"**SCENARIO A: GENERAL ASSESSMENT (User asks: 'Status?', 'Summary?', 'Assessment?', 'Update?')**\n"
                                    f"If the user wants a full review, generate a comprehensive clinical report:\n"
                                    f"1. **Current Status:** Summarize latest measurements ({state['wound_dims'] if 'state' in locals() and state else 'N/A'}) and narrative.\n"
                                    f"2. **Wound Progression:** Analyze healing trends (improving/stalled/deteriorating). Flag specific risk factors (Diabetes, Ischemia).\n"
                                    f"3. **Clinical Recommendations:**\n"
                                    f"   - **Dressings:** Suggest specific products based on exudate/tissue type (e.g., 'Consider silver alginate for heavy exudate').\n"
                                    f"   - **Diagnostics:** Flag missing tests (e.g., 'Order ABI/TBI if not done in 6 months').\n"
                                    f"   - **Referrals:** Vascular or ID referrals if critical signs are present.\n"
                                    f"4. **Gap Identification:** Explicitly state what data is missing (A1C, Offloading status).\n\n"
                                    
                                    f"**SCENARIO B: SPECIFIC QUESTION (User asks: 'Should we use Santyl?', 'Is it infected?', 'Plan for next week?')**\n"
                                    f"1. **Direct Answer:** Answer the specific question immediately. Do NOT generate a full report.\n"
                                    f"2. **Evidence:** Cite specific data points (dates, measurements) from the records to support your answer.\n"
                                    f"3. **Rationale:** Briefly explain the clinical reasoning (e.g., 'Santyl is appropriate because necrotic tissue is present...').\n"
                                    f"4. **Safety:** If the user suggests an unsafe action, flagged it immediately.\n\n"
                                    
                                    f"### USER QUESTION:\n{prompt}"
                                )
                                
                                add_log("🧠 SYNTHESIS: Sending to Gemma 2...")
                                response = query_engine.query(full_prompt)
                                response_text = st.write_stream(response.response_gen)
                                response_obj = response

                    # PATH 3: TRIAGE / MULTIMODAL (Image + Text)
                    else:
                        if img_upload and enable_vision and vision_data is not None and "error" not in vision_data:
                            add_log("🚑 MULTIMODAL MODE: Fusion Analysis (Image + History).")
                            
                            # 1. LINK TO PATIENT DATA
                            patient_context_str = "No specific patient ID linked. Analysis based on visual evidence only."
                            
                            if patient_id_for_image and 'preview_df' in st.session_state:
                                df = st.session_state['preview_df']
                                # Find the matching row
                                patient_record = df[df['Patient_ID'].astype(str) == str(patient_id_for_image)]
                                
                                if not patient_record.empty:
                                    rec = patient_record.iloc[0]
                                    patient_context_str = (
                                        f"✅ LINKED PATIENT RECORD ({patient_id_for_image}):\n"
                                        f"- Age/Sex: {rec.get('Age', '?')} / {rec.get('Sex', '?')}\n"
                                        f"- Known Comorbidities: {rec.get('Comorbidities', 'None listed')}\n"
                                        f"- Last Clinical Wound Size: {rec.get('Wound_Size_Length_cm', '?')} x {rec.get('Wound_Size_Width_cm', '?')} cm\n"
                                        f"- Latest Nurse Note: {rec.get('Narrative', 'None')}"
                                    )
                                    add_log(f"✅ DATA MERGE: Linked Image to Patient {patient_id_for_image} History.")
                                else:
                                    add_log(f"⚠️ DATA ERROR: Patient ID {patient_id_for_image} not found in CSV.")

                            # # 2. BUILD THE "MEDICAL DIRECTOR" PROMPT (Aggressive Mode)
                            # triage_prompt = (
                            #     f"### SYSTEM ROLE:\n"
                            #     f"You are an Expert Medical Director. You are speaking to a physician, NOT a patient.\n"
                            #     f"### TASK:\n"
                            #     f"Synthesize the VISUAL EVIDENCE (from LLaVA) and PATIENT HISTORY (from CSV) to create a treatment plan.\n\n"
                                
                            #     f"=== 👁️ VISUAL EVIDENCE ===\n"
                            #     f"{vision_context}\n\n"

                            #     f"=== 📋 PATIENT HISTORY ===\n"
                            #     f"{patient_context_str}\n\n"
                                
                            #     f"=== ❓ PHYSICIAN QUERY ===\n"
                            #     f"{prompt}\n\n"
                                
                            #     f"### CRITICAL INSTRUCTIONS:\n"
                            #     f"1. DO NOT say 'I am an AI' or 'Seek medical attention'. Assume the user is a doctor.\n"
                            #     f"2. If the CSV shows Comorbidities (e.g. Diabetes), explicitly factor them into your plan.\n"
                            #     f"3. Provide a numbered and cohesive Clinical Care Plan (e.g. Antibiotics, Debridement, Dressing etc).\n"
                            # )
                            
                            #response_text = str(Settings.llm.complete(triage_prompt))

                            # 2. BUILD THE "STRICT SYNTHESIS" PROMPT (AI WRITES THE REPORT)
                            synthesis_prompt = (
                            f"### ROLE: Expert Medical Director\n"
                            f"### TASK: Generate a structured clinical assessment (SOAP Note style).\n\n"
                            
                            f"--- INPUT 1: VISUAL EVIDENCE ---\n"
                            f"Findings: {vision_data.get('raw_output', 'No description')}\n"  # <--- The long paragraph goes here
                            f"Diagnosis: {vision_data.get('diagnosis', 'Unspecified')}\n\n"   # <--- The short label goes here
                            
                            f"--- INPUT 2: APPROVED PROTOCOL ---\n"
                            f"{vision_data.get('protocol', 'Refer to Guidelines')}\n\n"
                            
                            f"--- INPUT 3: PATIENT CONTEXT ---\n"
                            f"{patient_context_str}\n\n"
                            
                            f"### INSTRUCTIONS:\n"
                            # 1. Force Header to be short (Prevents the giant text bug)
                            f"1. **HEADER:**: '## 🏥 Clinical Assessment: {vision_data.get('diagnosis', 'Unspecified')}'\n"
                            
                            # 2. Give the long description its own home
                            f"2. **VISUAL OBSERVATIONS:** Create a section '## 👁️ Visual Evidence'. Summarize the visual findings input here. Do not put this in the header.\n"
                            
                            # 3. Patient Summary
                            f"3. **PATIENT SUMMARY:** Create a section '## 👤 Patient Factors'. Summarize the patient's age, comorbidities, and current status in 2 sentences.\n"
                            
                            # 4.Protocol Reference
                            f"4. **PROTOCOL REFERENCE:** Create a section '### 📋 Evidence-Based Standards'. List the approved protocol steps exactly as provided in Input 2. This serves as the safety reference.\n"
                            
                            # 5.Personalized Logic (The "Brain")
                            f"5. **PERSONALIZED CARE PLAN:** Create a section '### 🩺 Patient-Specific Plan'. Here, you must **APPLY** the protocol to the patient's context:\n"
                            f"   - If the patient has comorbidities (e.g., Hospice, Immobility), MODIFY the standard recommendation and explain why.\n"
                            f"   - Example: 'Standard protocol suggests TCC, but due to immobility, we recommend offloading boots.'\n"
                            f"   - Number the steps clearly.\n"
                            
                            # 6.Referral & Formatting
                            f"6. **REFERRAL:** End with a '### ⚠️ Specialist Recommendation' section.\n"
                            f"7. **FORMATTING:** Use bolding for key actions. Use clean bullet points."
                        )
                            
                            add_log("🧠 SYNTHESIS: Gemma 2 is formatting the final report...")
                            
                            # Generate response using Gemma
                            stream = Settings.llm.stream_complete(synthesis_prompt)
                            response_text = st.write_stream(part.delta for part in stream)
                        
                        # Check : did the vision fail?
                        elif img_upload and enable_vision and (vision_data is None or "error" in vision_data):
                            add_log("🛑 CRITICAL: Vision Model Failed. Falling back to text mode.")

                            error_detail = vision_data.get("error", "Unknown") if vision_data else "No response"
                            recovery_hint = vision_data.get("recovery", "") if vision_data else ""
                            response_text = (
                                f"**⚠️ System Error:** Vision Model Unavailable\n\n"
                                f"**Error:** {error_detail}\n\n"
                            )
                            
                            if recovery_hint == "restart_required":
                                response_text += (
                                    "**Recovery Steps:**\n"
                                    "1. Kill all Python/Streamlit processes\n"
                                    "2. Wait 5 seconds\n"
                                    "3. Restart the application\n\n"
                                )
                            
                            response_text += "You can still query patient records using text-only mode."

                        #Check: is vision diabled?
                        elif img_upload and not enable_vision:
                            add_log("🔒 SAFETY GATE: Triage unavailable.")
                            response_text = "Vision Module Locked."
                        
                        #Check: General Chat
                        else:
                            add_log("🧠 GENERAL QUERY: Answering from general knowledge.")
                            chat_engine = index.as_chat_engine(chat_mode="context", similarity_top_k=3)
                            response = chat_engine.stream_chat(prompt)
                            response_text = st.write_stream(response.response_gen)
                            response_obj = response

                    add_log("✅ COMPLETE: Response generated.")
                    # answer_placeholder.markdown(response_text)

                    # DEBUG EXPANDER
                    with st.expander("🕵️‍♂️ View Retrieved Evidence (Debug)"):
                        st.caption("Verify exactly what data the LLM used to generate the answer.")
                        
                        # CASE 1: Multimodal (Vision + Patient ID Link)
                        # This checks if we used the direct CSV lookup method
                        if 'patient_context_str' in locals() and 'vision_context' in locals():
                            st.info("🔍 Method: Direct Data Link & Vision")
                            
                            st.markdown("**1. Visual Analysis (LLaVA Output):**")
                            st.code(vision_context)
                            
                            st.markdown("**2. Patient History (CSV Lookup):**")
                            # This is the CRITICAL check. If this string says "Patient 10000", you found the bug.
                            st.code(patient_context_str) 

                        # CASE 2: Text-Only RAG (Standard Query)
                        # This checks if we used the Vector Database
                        elif 'response_obj' in locals() and hasattr(response_obj, 'source_nodes'):
                            st.info("🔍 Method: Vector Database Retrieval (RAG)")
                            
                            for i, node in enumerate(response_obj.source_nodes):
                                st.markdown(f"**📄 Source Chunk {i+1} (Match Score: {node.score:.2f})**")
                                # Show the actual text so you can read if it says "Patient 10770"
                                st.text(node.text) 
                                st.markdown("**Metadata:**")
                                st.json(node.metadata)
                                st.divider()
                        
                        else:
                            st.warning("No external context used (Pure LLM knowledge).")
                    
                    # Save to history
                    msg_data = {"role": "assistant", "content": response_text, "logs": logs}
                    if img_upload: msg_data["image_desc"] = vision_context
                    st.session_state.messages.append(msg_data)

    with tab2:
        st.subheader("📊 Data Inspector")
        if 'preview_df' in st.session_state:
            st.dataframe(st.session_state['preview_df'])