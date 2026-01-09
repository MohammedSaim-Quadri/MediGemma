import sys
import os
import logging

# Ensure we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.core.router import IntentRouter, QueryIntent
from src.engine.analytics import AnalyticsEngine
from src.engine.rag import ClinicalRAGEngine
from src.data_manager import DataManager
from src.engine.generator import LLMEngine

# Configure logging to be minimal
logging.basicConfig(level=logging.ERROR)
print("--- 🔬 DIAGNOSTIC TEST: PROBLEM QUERIES ---")

# 1. Initialize & Load Data
print("\n1. 📂 Loading Data...")
dm = DataManager()
# Manual load for testing (simulating user upload)
if os.path.exists("encounters.csv") and os.path.exists("patients.csv"):
    dm.load_data("encounters.csv", "patients.csv")
    print(f"   ✅ Data Loaded: {len(dm.df)} rows")
else:
    print("   ❌ Critical: CSV files not found in root. Cannot test RAG/Analytics.")
    exit()

print("\n--- DATA DIAGNOSTIC ---")
print(f"Unique patients in dataset: {dm.df['Patient_ID'].nunique()}")
print(f"Patient ID data type: {dm.df['Patient_ID'].dtype}")
print(f"Sample Patient IDs: {dm.df['Patient_ID'].unique()[:5]}")

# Check if IDs are strings or integers
if dm.df['Patient_ID'].dtype == 'object':
    print(f"Patient 20152 exists: {'20152' in dm.df['Patient_ID'].values}")
    print(f"Patient 31579 exists: {'31579' in dm.df['Patient_ID'].values}")
    if '20152' in dm.df['Patient_ID'].values:
        print(f"Patient 20152 encounters: {len(dm.df[dm.df['Patient_ID']=='20152'])}")
else:
    print(f"Patient 20152 exists: {20152 in dm.df['Patient_ID'].values}")
    print(f"Patient 31579 exists: {31579 in dm.df['Patient_ID'].values}")
    if 20152 in dm.df['Patient_ID'].values:
        print(f"Patient 20152 encounters: {len(dm.df[dm.df['Patient_ID']==20152])}")

# Check metadata in index
if dm.index:
    print("\n--- METADATA DIAGNOSTIC ---")
    docs = dm.index.docstore.docs
    if docs:
        sample_doc = list(docs.values())[0]
        print(f"Sample metadata: {sample_doc.metadata}")
        print(f"Sample patient_id type: {type(sample_doc.metadata.get('patient_id'))}")
    else:
        print("   ❌ Critical: No documents in index. Cannot test RAG/Analytics.")
        exit()

# 2. Initialize Engines
print("2. 🧠 Initializing Engines...")
llm = LLMEngine()
llm.initialize()

router = IntentRouter(llm_engine=llm)
analytics = AnalyticsEngine(data_manager=dm)
rag = ClinicalRAGEngine(index=dm.index)

# 3. The "Problem" Queries
queries = [
    "How many patients have ages more then 50?",        # FAILED BEFORE (Hallucinated -> Fixed by Pandas)
    "How many patients have diabetes?",                 # FAILED BEFORE (0/Crash -> Fixed by Data Cleaning)
    "Should we recommend santyl to patient 20152?",     # FAILED BEFORE (Safety -> Fixed by RAG)
    "Hows the wound progressing for patient 10770"      # FAILED BEFORE (Mismatch -> Fixed by RAG)
]

# 4. Run Simulation
print("\n--- 🏃 RUNNING SIMULATION ---")

for q in queries:
    print(f"\n❓ Query: '{q}'")
    
    # A. Route
    intent = router.classify(q)
    print(f"   🔀 Route: {intent.name}")
    
    # B. Execute
    response = ""
    try:
        if intent == QueryIntent.DATA_ANALYTICS:
            print("   ⚙️ Engine: Analytics (Pandas)")
            result = analytics.execute_query(q)
            response = result['text']
            
        elif intent == QueryIntent.PATIENT_ASSESSMENT:
            print("   ⚙️ Engine: Clinical RAG (Vector Search)")
            if rag.index is None:
                response = "Error: RAG Index not built."
            else:
                response = rag.chat(q)
        
        elif intent == QueryIntent.SIMPLE_FACT:
             print("   ⚙️ Engine: LLM (General Knowledge)")
             response = llm.generate(q)
             
    except Exception as e:
        response = f"❌ ERROR: {str(e)}"
        
    print(f"   📝 Answer: {response}")

print("\n--- 🏁 TEST COMPLETE ---")
