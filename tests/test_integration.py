import sys
import os
import pytest
import pandas as pd
from unittest.mock import MagicMock

# Path Setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.core.orchestrator import ClinicalOrchestrator
from src.core.router import QueryIntent
from src.data_manager import DataManager
from src.engine.analytics import AnalyticsEngine

# --- FIXTURES (Setup) ---
@pytest.fixture
def mock_orchestrator():
    """
    Creates an Orchestrator with:
    1. REAL DataManager (with dummy data)
    2. REAL AnalyticsEngine (to test logic)
    3. MOCKED RAG/LLM (to avoid GPU/API usage)
    """
    # 1. Setup Data
    dm = DataManager()
    data = [{
        'Patient_ID': '10770',
        'Age': 45,
        'Wound_Size_Area_cm2': 5.5,
        'Pain_Level': 2,
        'Narrative': 'Healing well.',
        'Encounter_Date': '2025-01-01',
        'Severity': 'Stable' # Simulate the enriched column
    }]
    dm.df = pd.DataFrame(data)
    
    # 2. Setup Real Analytics (We want to verify the Pandas logic runs)
    analytics = AnalyticsEngine(dm)
    
    # 3. Setup Mocks for Heavy Engines
    rag_engine = MagicMock()
    # Mock the response object structure
    mock_response = MagicMock()
    mock_response.response = "Clinical RAG Response"
    rag_engine.chat.return_value = mock_response
    rag_engine.index = True # Pretend index exists
    
    llm_engine = MagicMock()
    
    # 4. Create Orchestrator
    orchestrator = ClinicalOrchestrator(analytics, rag_engine, llm_engine, dm)
    
    return orchestrator, rag_engine, llm_engine

# --- INTEGRATION TESTS ---

def test_orchestrator_data_flow(mock_orchestrator):
    """
    Integration: User Query -> Router -> Analytics -> Result
    Verifies that a 'How many' question actually executes Pandas code.
    """
    orch, _, _ = mock_orchestrator
    
    # Ask a data question
    response, source, _, intent = orch.process_query("How many patients are in the dataset?", [])
    
    # Assertions
    assert intent == QueryIntent.DATA
    assert source == "Analytics Engine"
    assert "1" in response or "one" in response.lower() # Should find 1 patient

def test_ground_truth_injection(mock_orchestrator):
    """
    Integration: User Query -> Regex -> Ground Truth Injection -> RAG
    CRITICAL: Verifies that the Orchestrator modifies the prompt before sending to RAG.
    """
    orch, mock_rag, _ = mock_orchestrator
    
    query = "What is the status of Patient 10770?"
    
    # Execute
    orch.process_query(query, [])
    
    # Check what the RAG engine was actually called with
    # We expect the query to be modified with the [SYSTEM INJECTED...] block
    call_args = mock_rag.chat.call_args[0][0] # First arg of first call
    
    assert "10770" in call_args
    assert "[SYSTEM INJECTED GROUND TRUTH" in call_args
    assert "2025-01-01" in call_args # The date from our dummy data
    print(f"\n✅ Verified Injection:\n{call_args[:100]}...")

def test_fallback_to_llm(mock_orchestrator):
    """
    Integration: No Data Loaded -> Fallback to General LLM
    """
    orch, _, mock_llm = mock_orchestrator
    
    # Simulate empty data
    orch.dm.df = None 
    orch.rag.index = None
    
    # Mock LLM response
    mock_msg = MagicMock()
    mock_msg.message.content = "General Medical Advice"
    mock_llm.chat.return_value = mock_msg
    
    # Query
    response, source, _, intent = orch.process_query("What is diabetes?", [])
    
    assert source == "LLM (General Knowledge)"
    assert response == "General Medical Advice"