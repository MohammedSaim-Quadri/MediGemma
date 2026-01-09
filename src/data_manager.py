import pandas as pd
import logging
import os
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

logger = logging.getLogger(__name__)

class DataManager:
    """
    Centralized data handler.
    Features:
    - Auto-cleaning of dirty data.
    - LOCAL Embeddings (No OpenAI).
    - Vector Index for Clinical RAG.
    """
    def __init__(self):
        # self.enc_path = enc_path
        # self.pat_path = pat_path
        self.df = None
        self.index = None
        
        # 1. SETUP LOCAL EMBEDDINGS (Crucial for RAG)
        # Using a small, fast model that fits easily in memory
        logger.info("⚙️ Loading Local Embedding Model (BAAI/bge-small-en-v1.5)...")
        try:
            Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
            Settings.chunk_size = 512
        except Exception as e:
            logger.error(f"❌ Embedding Model Error: {e}")

    def load_data(self, enc_file, pat_file=None):
        try:
            logger.info("📂 Processing Uploaded Data...")
            
            # --- Load CSVs ---
            df_enc = pd.read_csv(enc_file)

            if 'Patient_ID' in df_enc.columns:
                df_enc['Patient_ID'] = df_enc['Patient_ID'].astype(str)
            
            if pat_file:
                df_pat = pd.read_csv(pat_file)
                if 'Patient_ID' in df_pat.columns:
                    df_pat['Patient_ID'] = df_pat['Patient_ID'].astype(str)
                self.df = pd.merge(df_enc, df_pat, on="Patient_ID", how="outer")
            else:
                self.df = df_enc
            
            # --- Clean Data (The "Diabetes Fix") ---
            if 'Comorbidities' in self.df.columns:
                self.df['Comorbidities'] = self.df['Comorbidities'].fillna("None").astype(str)
            
            text_cols = ['Narrative', 'Notes', 'Wound_Size_Length_cm', 'Treatment_Plan', 'Exudate_Type', 'Tissue_Exposed']
            for col in text_cols:
                if col in self.df.columns:
                    self.df[col] = self.df[col].fillna('N/A')

            # Clean numeric columns (fill with 0.0 to indicate missing)
            num_cols = ['Necrosis_Percent', 'Slough_Percent', 'Granulation_Percent']
            for col in num_cols:
                if col in self.df.columns:
                    self.df[col] = self.df[col].fillna(0.0)
            
            logger.info(f"✅ Data Loaded. Rows: {len(self.df)}")
            
            # --- Build RAG Index (Atomic Strategy) ---
            logger.info("🧠 Building Clinical Vector Index (Per-Visit Strategy)...")
            documents = []
            
            # Iterate through EVERY row in the dataframe (Every Visit)
            for _, row in self.df.iterrows():
                # Ensure Patient ID is a clean string for Metadata
                patient_id = str(row.get('Patient_ID', 'Unknown')).strip()
                if patient_id.endswith('.0'): # Fix float-string artifacts (e.g., "20152.0")
                    patient_id = patient_id[:-2]
                
                # 1. Context Header (Repeated for every doc so context is never lost)
                context_header = (
                    f"PATIENT CONTEXT: ID {patient_id}\n"
                    f"Demographics: Age {row.get('Age', 'N/A')}, Sex {row.get('Sex', 'N/A')}\n"
                    f"Medical History: {row.get('Comorbidities', 'None')}\n"
                )
                
                # 2. Specific Visit Details
                visit_details = (
                    f"ENCOUNTER DATE: {row.get('Encounter_Date', 'Unknown')}\n"
                    f"Wound Dims: {row.get('Wound_Size_Length_cm', '?')} x {row.get('Wound_Size_Width_cm', '?')} cm\n"
                    f"TISSUE: Necrosis {row.get('Necrosis_Percent', 0)}%, Slough {row.get('Slough_Percent', 0)}%, Granulation {row.get('Granulation_Percent', 0)}%\n"
                    f"EXUDATE: {row.get('Exudate_Type', 'N/A')}, Amount: {row.get('Exudate_Amount', 'N/A')}\n"
                    f"PAIN: {row.get('Pain_Level', 'N/A')}/10\n"
                    f"Narrative: {row.get('Narrative', 'No notes')}\n"
                    f"Plan: {row.get('Treatment_Plan', 'None')}\n"
                )
                
                # 3. Combine
                full_text = context_header + "--- VISIT DETAILS ---\n" + visit_details
                
                # 4. Create Document with Metadata (Crucial for filtering)
                doc = Document(
                    text=full_text, 
                    metadata={
                        "patient_id": patient_id,
                        "date": str(row.get('Encounter_Date', 'Unknown'))
                    }
                )
                documents.append(doc)
            
            if documents:
                self.index = VectorStoreIndex.from_documents(documents)
                logger.info(f"✅ Index Built Successfully ({len(documents)} clinical encounters).")
            
            return True

        except Exception as e:
            logger.error(f"❌ Data Manager Error: {e}")
            return False

    def get_preview(self):
        return self.df
