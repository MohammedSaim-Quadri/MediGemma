import streamlit as st
import pandas as pd
from clinical_rules import generate_priority_report

def render_dashboard(df):
    """
    Renders the Medical Director's Triage Console.
    """
    st.title("🏥 Medical Director Action Report")
    st.markdown("### ⚡ Prioritized Patient Triage")
    
    # 1. Run the Logic
    with st.spinner("Analyzing patient records for critical anomalies..."):
        triage_data = generate_priority_report(df)
    
    # 2. Key Metrics Row
    total_pat = len(df['Patient_ID'].unique())
    # Note: Keys must match clinical_rules.py (Capitalized)
    critical = len([x for x in triage_data if x['Severity'] == 'Critical'])
    urgent = len([x for x in triage_data if x['Severity'] == 'Urgent'])
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Patients", total_pat)
    c2.metric("🚨 Critical Action", critical, delta_color="inverse")
    c3.metric("⚠️ Urgent Review", urgent, delta_color="inverse")
    
    # Prepare data for download
    if triage_data:
        csv_data = pd.DataFrame(triage_data)
        if not csv_data.empty:
            # Clean up list format for CSV
            def clean_alert(alert_list):
                text = "; ".join(alert_list)
                # Remove Markdown
                text = text.replace("**", "")
                # Remove Emojis
                for icon in ["⚠️", "🔴", "☣️", "🐢"]:
                    text = text.replace(icon, "")
                return text.strip()
            
            # Apply cleaning
            csv_data['Alerts'] = csv_data['Alerts'].apply(clean_alert)
            
            # Encode with BOM for Excel
            csv = csv_data.to_csv(index=False).encode('utf-8-sig')
            
            st.download_button(
                label="📥 Download Triage Report (CSV)",
                data=csv,
                file_name="medical_director_triage_report.csv",
                mime="text/csv",
            )
    
    st.divider()
    
    # 3. The Triage Feed
    if not triage_data:
        st.success("✅ No critical alerts detected. All patients stable.")
        return

    for patient in triage_data:
        # Visual Styling
        if patient['Severity'] == 'Critical':
            icon = "🔴"
        else:
            icon = "🟠"
            
        # FIX 1: Handle "nan" display gracefully
        p_age = str(patient['Age'])
        if p_age.lower() == 'nan': p_age = "?"
        
        p_sex = str(patient['Sex'])
        if p_sex.lower() == 'nan': p_sex = "?"

        # FIX 2: Updated Keys to match clinical_rules.py ("Patient ID", "Date", "Alerts")
        header_text = f"{icon} **{patient['Patient ID']}** (Age: {p_age} | {p_sex}) - Last Visit: {patient['Date']}"
        
        with st.expander(header_text, expanded=(patient['Severity'] == 'Critical')):
            
            # Context Section
            st.markdown(f"**Condition:** {patient['Comorbidities']}")
            st.markdown(f"**Wound Size:** {patient['Wound Size (cm)']} cm")
            if patient['Latest Note']:
                st.markdown(f"**Latest Note:** _{patient['Latest Note']}_")
            
            st.divider()
            
            # Alert List
            for alert in patient['Alerts']:
                st.markdown(f"- {alert}")
            
            # FIX 3: Restore the Analyze Button (using correct key)
            if st.button(f"🔎 Analyze {patient['Patient ID']}", key=f"btn_{patient['Patient ID']}"):
                # This injects a prompt into the chat and switches tabs
                st.session_state.messages.append({
                    "role": "user", 
                    "content": f"Generate a detailed treatment plan and gap analysis for Patient {patient['Patient ID']}"
                })
                # Turn off dashboard mode to go back to chat
                st.session_state['md_mode_active'] = False
                st.rerun()