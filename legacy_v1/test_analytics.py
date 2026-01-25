import sys
import logging
import pandas as pd

# Ensure we can import from src
sys.path.insert(0, 'src')

from src.data_manager import DataManager
from src.engine.analytics import AnalyticsEngine

# Configure logging to see the "Merge" success message
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("--- 🧪 STARTING INTEGRATION TEST ---")

# 1. Initialize Real DataManager (It will auto-find and merge both CSVs)
print("\n📂 Loading Data...")
dm = DataManager()

# Verify columns exist before running the engine
if dm.df is not None:
    cols = dm.df.columns.tolist()
    print(f"✅ Data Columns: {cols}")
    if "Age" not in cols:
        print("❌ CRITICAL: 'Age' column missing! Merge failed.")
    else:
        print("✅ Merge Successful: 'Age' column found.")
else:
    print("❌ CRITICAL: Dataframe is None.")
    exit()

# 2. Initialize Analytics Engine
print("\n🧠 Initializing Analytics Engine...")
analytics = AnalyticsEngine(dm)
analytics.initialize()

# 3. Critical Funding Queries
queries = [
    "How many patients are in the database?",
    "How many patients have diabetes?",
    "What is the average age?"
]

print("\n--- 🚀 RUNNING QUERIES ---")
for q in queries:
    print(f"\n❓ Query: {q}")
    result = analytics.execute_query(q)
    print(f"📝 Answer: {result['text']}")

print("\n--- 🏁 TEST COMPLETE ---")
