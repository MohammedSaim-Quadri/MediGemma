import pandas as pd
import logging
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataManager:
    """
    Centralized data handler.
    Features:
    - Auto-cleaning of dirty data (Fixes 'N/A' crashes).
    - LOCAL Embeddings (No OpenAI).
    - Rich Vector Index for Clinical RAG.
    """
    def __init__(self):
        self.df = None
        self.index = None
        
        # --- 1. SETUP LOCAL EMBEDDINGS (Crucial for RAG) ---
        try:
            logger.info("⚙️ Loading Local Embedding Model (BAAI/bge-small-en-v1.5)...")
            self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
            
            # CRITICAL: Apply to Global Settings
            Settings.embed_model = self.embed_model
            
            # Explicitly set LLM to None here to prevent OpenAI default
            Settings.llm = None
            Settings.chunk_size = 512 
            
        except Exception as e:
            logger.error(f"❌ Failed to load Embedding Model: {e}")

    def load_data(self, enc_file, pat_file=None):
        try:
            logger.info("📂 Processing Uploaded Data...")
            
            # --- Load CSVs ---
            df_enc = pd.read_csv(enc_file)

            # Clean column names (strip whitespace)
            # 2. Normalize Patient_ID (Force to String)
            # This fixes the merge failure (Int vs String mismatch)
            if 'Patient_ID' in df_enc.columns:
                df_enc['Patient_ID'] = df_enc['Patient_ID'].astype(str).str.strip()
            elif 'patient_id' in df_enc.columns:
                df_enc['Patient_ID'] = df_enc['patient_id'].astype(str).str.strip()
            
            # 3. Merge Demographics (If provided)
            if pat_file:
                df_pat = pd.read_csv(pat_file)
                
                # Normalize Patient_ID in demographics too
                if 'Patient_ID' in df_pat.columns:
                    df_pat['Patient_ID'] = df_pat['Patient_ID'].astype(str).str.strip()
                elif 'patient_id' in df_pat.columns:
                    df_pat['Patient_ID'] = df_pat['patient_id'].astype(str).str.strip()
                
                # Perform Left Merge
                self.df = pd.merge(df_enc, df_pat, on='Patient_ID', how='outer', suffixes=('', '_demo'))
                logger.info(f"✅ Merged demographics. Shape: {self.df.shape}")
            else:
                self.df = df_enc
                logger.warning("⚠️ No demographics file provided. Using encounters only.")

            # --- 🛡️ CRITICAL FIX: AGGRESSIVE TYPE CLEANING ---
            # 1. Force Numeric Columns (Fixes ArrowInvalid Crash)
            numeric_cols = [
                'Wound_Size_Length_cm', 
                'Wound_Size_Width_cm', 
                'Wound_Size_Area_cm2',
                'Patient_Age',
                'Age',
                'Necrosis_Percent', 
                'Slough_Percent', 
                'Granulation_Percent',
                'Pain_Level'
            ]
            
            for col in numeric_cols:
                if col in self.df.columns:
                    # Coerce errors (turn "N/A" -> NaN)
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                    # Fill NaN with 0.0 (Safe for Dashboard)
                    self.df[col] = self.df[col].fillna(0.0)

            # Ensure Comorbidities is never null (Fixes 'History: None')
            if 'Comorbidities' in self.df.columns:
                self.df['Comorbidities'] = self.df['Comorbidities'].fillna("No medical history documented").astype(str)
            else:
                self.df['Comorbidities'] = "No medical history documented"
            
            text_cols = ['Narrative', 'Notes', 'Treatment_Plan', 'Exudate_Type', 'Tissue_Exposed']
            for col in text_cols:
                if col in self.df.columns:
                    self.df[col] = self.df[col].fillna('N/A').astype(str)

            logger.info(f"✅ Data Loaded & Cleaned. Rows: {len(self.df)}")
            
            # --- Build RAG Index (Rich Context Strategy) ---
            self._build_index()
            
            return True

        except Exception as e:
            logger.error(f"❌ Data Manager Error: {e}")
            return False

    def _build_index(self):
        logger.info("🧠 Building Clinical Vector Index (Per-Visit Strategy)...")
        
        # Ensure Settings are persistent
        if Settings.embed_model is None:
             Settings.embed_model = self.embed_model

        documents = []
        
        for _, row in self.df.iterrows():
            
            # 1. Context Header (Rich Patient Context)
            context_header = (
                f"PATIENT CONTEXT: ID {row.get('Patient_ID', 'Unknown')}\n"
                f"Demographics: Age {row.get('Age', 'N/A')}, Sex {row.get('Sex', 'N/A')}\n"
                f"Medical History: {row.get('Comorbidities', 'None')}\n"
            )
            
            # 2. Specific Visit Details
            # Note: We use .get with defaults to handle missing cols safely
            visit_details = (
                f"ENCOUNTER DATE: {row.get('Encounter_Date', 'Unknown')}\n"
                f"Wound Dims: {row.get('Wound_Size_Length_cm', '?')} x {row.get('Wound_Size_Width_cm', '?')} cm\n"
                f"TISSUE: Necrosis {row.get('Necrosis_Percent', 0)}%, Slough {row.get('Slough_Percent', 0)}%, Granulation {row.get('Granulation_Percent', 0)}%\n"
                f"EXUDATE: {row.get('Exudate_Type', 'N/A')}\n"
                f"PAIN: {row.get('Pain_Level', 'N/A')}/10\n"
                f"Narrative: {row.get('Narrative', 'No notes')}\n"
                f"Plan: {row.get('Treatment_Plan', 'None')}\n"
            )
            
            # 3. Combine
            full_text = context_header + "--- VISIT DETAILS ---\n" + visit_details
            
            # 4. Create Document with Metadata
            doc = Document(
                text=full_text, 
                metadata={
                    "patient_id": str(row.get('Patient_ID', 'Unknown')),
                    "date": str(row.get('Encounter_Date', 'Unknown'))
                }
            )
            documents.append(doc)
        
        if documents:
            self.index = VectorStoreIndex.from_documents(documents)
            logger.info(f"✅ Index Built Successfully ({len(documents)} clinical encounters).")

    def get_preview(self):
        return self.df

    # In src/data_manager.py - Add this method to the DataManager class

    def get_patient_current_state(self, patient_id):
        """
        Retrieves the absolute latest clinical state (Ground Truth).
        """
        try:
            # Ensure ID is string
            pid = str(patient_id).strip()
            
            # Filter for patient
            p_data = self.df[self.df['Patient_ID'] == pid].copy()
            
            if p_data.empty:
                return None

            p_data['Encounter_Date'] = pd.to_datetime(p_data['Encounter_Date'])
            
            # Sort by Date AND Visit Number (Latest date, highest visit number)
            # This handles cases where a patient has 2 visits on the same day
            if 'Visit_Number' in p_data.columns:
                p_data = p_data.sort_values(by=['Encounter_Date', 'Visit_Number'], ascending=[False, False])
            else:
                p_data = p_data.sort_values(by='Encounter_Date', ascending=False)
                
            latest = p_data.iloc[0]
            
            # Convert back to string for display
            date_str = latest['Encounter_Date'].strftime('%Y-%m-%d')
            
            return {
                "last_visit": date_str,
                "wound_dims": f"{latest.get('Wound_Size_Length_cm', '?')} x {latest.get('Wound_Size_Width_cm', '?')} cm",
                "narrative": str(latest.get('Narrative', 'No notes'))[:200] + "...", # Truncate for token limits
                "severity": "Critical" if "deteriorating" in str(latest.get('Narrative', '')).lower() else "Stable"
            }
        except Exception as e:
            logger.error(f"Error fetching state: {e}")
            return None
