"""Tests for --questions-from YAML loading in run_benchmark."""
import pytest
import yaml
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def questions_yaml(tmp_path):
    """Create a minimal benchmark_questions.yaml."""
    data = {
        "version": "1.0",
        "total_questions": 2,
        "uncertainty_instruction": "If uncertain, state so.",
        "questions": [
            {"id": "Q1", "module": "Classification", "question": "What type of wound is this?"},
            {"id": "Q2", "module": "Tissue", "question": "Describe the tissue types."},
        ],
    }
    path = tmp_path / "questions.yaml"
    path.write_text(yaml.dump(data))
    return str(path)


@pytest.fixture
def questions_yaml_no_uncertainty(tmp_path):
    """YAML without uncertainty_instruction."""
    data = {
        "questions": [
            {"id": "Q1", "module": "Test", "question": "Simple question?"},
        ],
    }
    path = tmp_path / "questions.yaml"
    path.write_text(yaml.dump(data))
    return str(path)


def test_load_questions_from_yaml(questions_yaml):
    from scripts.run_benchmark import load_questions_from_yaml
    questions = load_questions_from_yaml(questions_yaml)
    assert len(questions) == 2
    assert "What type of wound is this?" in questions[0]
    assert "If uncertain, state so." in questions[0]


def test_load_questions_prepends_uncertainty(questions_yaml):
    from scripts.run_benchmark import load_questions_from_yaml
    questions = load_questions_from_yaml(questions_yaml)
    for q in questions:
        assert q.startswith("If uncertain, state so.")


def test_load_questions_without_uncertainty(questions_yaml_no_uncertainty):
    from scripts.run_benchmark import load_questions_from_yaml
    questions = load_questions_from_yaml(questions_yaml_no_uncertainty)
    assert len(questions) == 1
    assert questions[0] == "Simple question?"
