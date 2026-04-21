"""Tests for eval_viewer pure functions."""
import json
import pytest
from pathlib import Path


@pytest.fixture
def sample_jsonl(tmp_path):
    """Create a minimal JSONL file for testing."""
    records = [
        {
            "case_id": "case_001",
            "timestamp": "2026-02-08T19:00:00",
            "model_name": "test_model",
            "profile_name": "default",
            "prompt_template": "clinician_v1",
            "image_path": "/fake/path/img1.jpg",
            "question": "Identify the wound type and classify it.",
            "system_prompt": "You are a specialist.",
            "generation_params": {"max_new_tokens": 2048},
            "model_output": "This is a **test** output.",
            "inference_time_sec": 5.5,
            "error": None,
            "env": {"gpu_name": "RTX 4090", "torch_version": "2.9.1"},
        },
        {
            "case_id": "case_002",
            "timestamp": "2026-02-08T19:01:00",
            "model_name": "test_model",
            "profile_name": "default",
            "prompt_template": "clinician_v1",
            "image_path": "/fake/path/img1.jpg",
            "question": "Assess for visual signs of wound infection.",
            "system_prompt": "You are a specialist.",
            "generation_params": {"max_new_tokens": 2048},
            "model_output": "Infection assessment output.",
            "inference_time_sec": 8.2,
            "error": None,
            "env": {},
        },
        {
            "case_id": "case_003",
            "timestamp": "2026-02-08T19:02:00",
            "model_name": "test_model",
            "profile_name": "default",
            "prompt_template": "clinician_v1",
            "image_path": "/fake/path/img2.jpg",
            "question": "Unknown question type here.",
            "system_prompt": "You are a specialist.",
            "generation_params": {},
            "model_output": "",
            "inference_time_sec": 0.0,
            "error": "Tensor size mismatch",
            "env": {},
        },
    ]
    filepath = tmp_path / "test_eval.jsonl"
    with open(filepath, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return filepath


def test_load_question_labels():
    from src.interface.eval_viewer import load_question_labels
    labels = load_question_labels()
    assert len(labels) == 9
    assert "Q1" in labels
    assert "Q9" in labels
    assert "Classification" in labels["Q1"]


def test_detect_question_label():
    from src.interface.eval_viewer import load_question_labels, detect_question_label
    labels_map = load_question_labels()

    # Use actual question text that contains the YAML prefix
    qid, label = detect_question_label(
        "Identify the wound type (e.g., pressure injury, diabetic foot ulcer)", labels_map
    )
    assert qid == "Q1"

    qid, label = detect_question_label(
        "Assess for visual signs of wound infection. Evaluate each", labels_map
    )
    assert qid == "Q5"

    qid, label = detect_question_label("Something completely unknown", labels_map)
    assert qid == "Q?"


def test_load_eval_data(sample_jsonl):
    from src.interface.eval_viewer import load_eval_data_raw
    records = load_eval_data_raw(sample_jsonl)
    assert len(records) == 3
    assert records[0]["case_id"] == "case_001"
    assert records[2]["error"] == "Tensor size mismatch"


def test_load_eval_data_bad_json(tmp_path):
    filepath = tmp_path / "bad.jsonl"
    filepath.write_text('{"valid": true}\nnot json at all\n{"also": "valid"}\n')
    from src.interface.eval_viewer import load_eval_data_raw
    records = load_eval_data_raw(filepath)
    assert len(records) == 2


def test_load_eval_data_empty(tmp_path):
    filepath = tmp_path / "empty.jsonl"
    filepath.write_text("")
    from src.interface.eval_viewer import load_eval_data_raw
    records = load_eval_data_raw(filepath)
    assert records == []


def test_get_jsonl_files(tmp_path):
    (tmp_path / "a.jsonl").write_text("{}\n")
    (tmp_path / "b.jsonl").write_text("{}\n")
    (tmp_path / "c.md").write_text("not jsonl")
    from src.interface.eval_viewer import get_jsonl_files
    files = get_jsonl_files(tmp_path)
    assert len(files) == 2
    assert all(f.suffix == ".jsonl" for f in files)


def test_is_safe_image_path():
    from src.interface.eval_viewer import is_safe_image_path, PROJECT_ROOT
    safe = str(PROJECT_ROOT / "data" / "datasets" / "test.jpg")
    assert is_safe_image_path(safe) is True
    assert is_safe_image_path("/etc/passwd") is False
    assert is_safe_image_path(str(PROJECT_ROOT / ".." / "etc" / "passwd")) is False


def test_compute_summary():
    from src.interface.eval_viewer import compute_summary
    records = [
        {"model_name": "m1", "inference_time_sec": 5.0, "error": None, "image_path": "/a.jpg", "env": {"gpu_name": "RTX"}},
        {"model_name": "m1", "inference_time_sec": 10.0, "error": None, "image_path": "/b.jpg", "env": {}},
        {"model_name": "m1", "inference_time_sec": 0.0, "error": "fail", "image_path": "/a.jpg", "env": {}},
    ]
    s = compute_summary(records)
    assert s["total"] == 3
    assert s["errors"] == 1
    assert s["avg_time"] == 7.5
    assert s["unique_images"] == 2
