import pandas as pd
from llama_index.core import VectorStoreIndex, Document

def load_and_process_data(enc_file, pat_file=None):
    """
    Pure logic layer: reads CSVs, merges data, and builds vector index.

    Args:
        enc_file(str or buffer): Path or file-like object for encounters
        pat_file(str or buffer, optional): Path or file-like object for patients

    Returns:
        tuple: (index, dataframe)
    """
    # 1. load and merge data
    # pandas can read directly from file buffers(memory), so we dont need to save to disk
    df_enc = pd.read_csv(enc_file)

    if pat_file:
        df_pat = pd.read_csv(pat_file)
        df = pd.merge(df_enc, df_pat, on="Patient_ID", how="outer")
    else:
        df = df_enc

    # 2. construct LlamaIndex Documents
    documents = []

    # Group by patient to create structured records
    for patient_id, patient_visits in df.groupby('Patient_ID'):
        if patient_visits.empty:
            continue

        first_row = patient_visits.iloc[0]

        # A. Patient Demographics Header
        patient_summary = (
            f"=== PATIENT RECORD: {patient_id} ===\n"
            f"Patient ID: {patient_id}\n"
            f"Age: {first_row.get('Age', 'Unknown')} years old\n"
            f"Sex: {first_row.get('Sex', 'Unknown')}\n"
            f"Comorbidities: {first_row.get('Comorbidities', 'None listed')}\n"
            f"Notes: {first_row.get('Notes', 'None')}\n\n"
        )
        
        # B. Visit History Body
        visit_details = []
        for _, visit in patient_visits.iterrows():
            if pd.isna(visit.get('Encounter_Date')):
                continue
                
            visit_text = (
                f"--- Visit #{visit.get('Visit_Number', 'N/A')} on {visit.get('Encounter_Date', 'N/A')} ---\n"
                f"  Wound: {visit.get('Wound_Size_Length_cm', '?')}cm x {visit.get('Wound_Size_Width_cm', '?')}cm\n"
                f"  Pain Level: {visit.get('Pain_Level', 'N/A')}/10\n"
                f"  Narrative: {visit.get('Narrative', 'N/A')}\n"
            )
            visit_details.append(visit_text)
        
        # C. Combine into one Context Document
        if visit_details:
            full_text = patient_summary + f"VISIT HISTORY FOR PATIENT {patient_id}:\n" + "\n".join(visit_details)
        else:
            full_text = patient_summary + "VISIT HISTORY: No clinical visits recorded yet.\n"
        
        # Add metadata for filtering (Critical for RAG)
        documents.append(Document(text=full_text, metadata={"patient_id": str(patient_id)}))
    
    # 3. Build Index
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    
    return index, df