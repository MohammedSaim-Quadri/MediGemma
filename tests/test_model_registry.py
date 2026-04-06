"""
Tests for the global model registry in src/engine/test_models.py.

Covers: register_model, get_registered_model, clear_registry,
        cleanup_python_models, and thread safety.
"""
import sys
import os
import threading
from unittest.mock import patch, MagicMock

import pytest

# Ensure we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.engine.test_models import (
    register_model,
    get_registered_model,
    clear_registry,
    cleanup_python_models,
    set_model_loading,
    get_model_loading,
)


# ---------------------------------------------------------------------------
# Fixture: ensure every test starts with a clean registry
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _clean_registry():
    """Clear the global registry before AND after every test."""
    clear_registry()
    yield
    clear_registry()


# ---------------------------------------------------------------------------
# Fake model / processor helpers
# ---------------------------------------------------------------------------
class FakeProcessor:
    def __init__(self, tag: str = "default"):
        self.tag = tag

    def __repr__(self):
        return f"FakeProcessor({self.tag!r})"


class FakeModel:
    def __init__(self, tag: str = "default"):
        self.tag = tag

    def __repr__(self):
        return f"FakeModel({self.tag!r})"


# ===== 1. test_registry_starts_empty =====
def test_registry_starts_empty():
    """After clear_registry (via fixture), get_registered_model returns all Nones."""
    name, processor, model = get_registered_model()
    assert name is None
    assert processor is None
    assert model is None


# ===== 2. test_register_and_get =====
def test_register_and_get():
    """register_model stores name/processor/model; get_registered_model retrieves them."""
    proc = FakeProcessor("medgemma")
    mdl = FakeModel("medgemma")

    register_model("medgemma", proc, mdl)

    name, processor, model = get_registered_model()
    assert name == "medgemma"
    assert processor is proc
    assert model is mdl


# ===== 3. test_register_replaces_previous =====
def test_register_replaces_previous():
    """Registering a new model completely replaces the old one."""
    proc_old = FakeProcessor("old")
    mdl_old = FakeModel("old")
    register_model("hulumed", proc_old, mdl_old)

    proc_new = FakeProcessor("new")
    mdl_new = FakeModel("new")
    register_model("mg4b", proc_new, mdl_new)

    name, processor, model = get_registered_model()
    assert name == "mg4b"
    assert processor is proc_new
    assert model is mdl_new
    # Old objects are no longer referenced by the registry
    assert processor is not proc_old
    assert model is not mdl_old


# ===== 4. test_clear_registry =====
def test_clear_registry():
    """clear_registry resets everything to None."""
    register_model("medgemma", FakeProcessor(), FakeModel())

    clear_registry()

    name, processor, model = get_registered_model()
    assert name is None
    assert processor is None
    assert model is None


# ===== 5. test_register_after_clear =====
def test_register_after_clear():
    """After clearing, a fresh register_model works correctly."""
    register_model("hulumed", FakeProcessor("first"), FakeModel("first"))
    clear_registry()

    proc = FakeProcessor("second")
    mdl = FakeModel("second")
    register_model("mg4b", proc, mdl)

    name, processor, model = get_registered_model()
    assert name == "mg4b"
    assert processor is proc
    assert model is mdl


# ===== 6. test_thread_safety =====
def test_thread_safety():
    """
    Concurrent register/get calls must not raise exceptions or corrupt state.
    After all threads finish the registry must hold a valid (name, proc, model)
    tuple from one of the writers -- or (None, None, None) if a clear sneaked in last.
    """
    errors: list[Exception] = []
    num_threads = 20
    iterations_per_thread = 200

    def writer(thread_id: int):
        try:
            for i in range(iterations_per_thread):
                tag = f"t{thread_id}_i{i}"
                register_model(tag, FakeProcessor(tag), FakeModel(tag))
        except Exception as exc:
            errors.append(exc)

    def reader():
        try:
            for _ in range(iterations_per_thread):
                name, proc, mdl = get_registered_model()
                # All three must be consistently None or consistently non-None
                nones = [name is None, proc is None, mdl is None]
                assert all(nones) or not any(nones), (
                    f"Inconsistent registry state: name={name}, proc={proc}, mdl={mdl}"
                )
        except Exception as exc:
            errors.append(exc)

    def clearer():
        try:
            for _ in range(iterations_per_thread):
                clear_registry()
        except Exception as exc:
            errors.append(exc)

    threads = []
    for tid in range(num_threads // 2):
        threads.append(threading.Thread(target=writer, args=(tid,)))
        threads.append(threading.Thread(target=reader))
    # Add a couple of clearers for extra contention
    threads.append(threading.Thread(target=clearer))
    threads.append(threading.Thread(target=clearer))

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert errors == [], f"Thread errors: {errors}"

    # Final state must still be consistent
    name, proc, mdl = get_registered_model()
    nones = [name is None, proc is None, mdl is None]
    assert all(nones) or not any(nones), (
        f"Inconsistent final state: name={name}, proc={proc}, mdl={mdl}"
    )


# ===== 7. test_cleanup_python_models_clears_registry =====
def test_cleanup_python_models_clears_registry():
    """
    cleanup_python_models() should:
      - clear the global registry (model, processor, model_name all None)
      - clean up legacy session_state keys
      - set loaded flags to False
    We mock streamlit.session_state, torch, and gc to avoid GPU / Streamlit deps.
    """
    # Pre-populate the registry
    register_model("hulumed", FakeProcessor("hulu"), FakeModel("hulu"))

    # Verify it is populated before cleanup
    name, _, _ = get_registered_model()
    assert name == "hulumed"

    # Build a fake session_state dict with some legacy keys
    fake_session_state = {
        "medgemma_model": FakeModel("legacy_mg"),
        "hulumed_processor": FakeProcessor("legacy_hp"),
        "vision_engine": object(),
        # loaded flags (will be overwritten to False)
        "medgemma_loaded": True,
        "medgemma4b_loaded": True,
        "hulumed_loaded": True,
    }

    mock_st = MagicMock()
    mock_st.session_state = fake_session_state

    with patch.dict("sys.modules", {"streamlit": mock_st}), \
         patch("src.engine.test_models.force_vram_cleanup"):
        cleanup_python_models()

    # Registry must be cleared
    name, processor, model = get_registered_model()
    assert name is None
    assert processor is None
    assert model is None

    # Legacy keys must have been deleted
    assert "medgemma_model" not in fake_session_state
    assert "hulumed_processor" not in fake_session_state
    assert "vision_engine" not in fake_session_state

    # Loaded flags must be False
    assert fake_session_state["medgemma_loaded"] is False
    assert fake_session_state["medgemma4b_loaded"] is False
    assert fake_session_state["hulumed_loaded"] is False


def test_cleanup_python_models_when_registry_empty():
    """
    cleanup_python_models() should not error when the registry is already empty.
    """
    # Registry is already cleared by the fixture

    fake_session_state = {
        "medgemma_loaded": True,
        "medgemma4b_loaded": True,
        "hulumed_loaded": True,
    }

    mock_st = MagicMock()
    mock_st.session_state = fake_session_state

    with patch.dict("sys.modules", {"streamlit": mock_st}), \
         patch("src.engine.test_models.force_vram_cleanup"):
        # Should not raise
        cleanup_python_models()

    name, processor, model = get_registered_model()
    assert name is None
    assert processor is None
    assert model is None

    assert fake_session_state["medgemma_loaded"] is False
    assert fake_session_state["medgemma4b_loaded"] is False
    assert fake_session_state["hulumed_loaded"] is False


# ===== 9. test_loading_flag =====
def test_loading_flag():
    """set_model_loading / get_model_loading round-trip."""
    assert get_model_loading() is None

    set_model_loading("medgemma")
    assert get_model_loading() == "medgemma"

    set_model_loading(None)
    assert get_model_loading() is None


# ===== 10. test_cleanup_preserves_loading_flag =====
def test_cleanup_preserves_loading_flag():
    """
    cleanup_python_models() must NOT clear the loading flag.
    This prevents race conditions where master_evict_with_retry()
    clears the flag that was just set by set_model_loading().
    """
    register_model("hulumed", FakeProcessor(), FakeModel())
    set_model_loading("medgemma")  # Simulates: user clicked "Load MedGemma"

    fake_session_state = {
        "medgemma_loaded": False,
        "medgemma4b_loaded": False,
        "hulumed_loaded": True,
    }
    mock_st = MagicMock()
    mock_st.session_state = fake_session_state

    with patch.dict("sys.modules", {"streamlit": mock_st}), \
         patch("src.engine.test_models.force_vram_cleanup"):
        cleanup_python_models()

    # Registry model should be cleared
    name, _, _ = get_registered_model()
    assert name is None

    # But loading flag must STILL be "medgemma"
    assert get_model_loading() == "medgemma"

    # Cleanup
    set_model_loading(None)
