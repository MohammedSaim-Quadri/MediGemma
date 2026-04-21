"""
Tests for evaluation data schemas and JSONL utilities.

Validates EvalCase/EvalResult serialization, JSONL I/O,
and weighted score computation.
"""

import json
import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.schemas import (
    EvalCase,
    EvalResult,
    DIMENSION_WEIGHTS,
    VALID_VERDICTS,
    CRITICAL_FLAGS,
    compute_weighted_total,
    determine_verdict,
    append_jsonl,
    read_jsonl,
)


class TestDimensionWeights:
    """Tests for evaluation dimension weights."""

    def test_weights_sum_to_one(self):
        total = sum(DIMENSION_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001, f"Weights sum to {total}, expected 1.0"

    def test_seven_dimensions(self):
        assert len(DIMENSION_WEIGHTS) == 7

    def test_expected_dimensions_present(self):
        expected = {
            "clinical_accuracy", "safety", "clinical_completeness",
            "evidence_based_reasoning", "reasoning_coherence",
            "specificity", "communication_clarity"
        }
        assert expected == set(DIMENSION_WEIGHTS.keys())

    def test_safety_and_accuracy_highest_weight(self):
        assert DIMENSION_WEIGHTS["clinical_accuracy"] == 0.25
        assert DIMENSION_WEIGHTS["safety"] == 0.25


class TestEvalCase:
    """Tests for EvalCase dataclass."""

    def _make_case(self, **overrides):
        defaults = {
            "case_id": "case_001",
            "timestamp": "2026-02-08T12:00:00",
            "model_name": "medgemma_27b",
            "profile_name": "default",
            "prompt_template": "clinician_v1",
            "image_path": "uploads/test.jpg",
            "question": "Describe this wound",
            "system_prompt": "You are a wound care specialist...",
            "generation_params": {"max_new_tokens": 2048, "do_sample": False},
            "model_output": "This appears to be a pressure ulcer...",
            "inference_time_sec": 12.5,
        }
        defaults.update(overrides)
        return EvalCase(**defaults)

    def test_create_case(self):
        case = self._make_case()
        assert case.case_id == "case_001"
        assert case.error is None

    def test_case_with_error(self):
        case = self._make_case(error="CUDA OOM")
        assert case.error == "CUDA OOM"

    def test_to_dict(self):
        case = self._make_case()
        d = case.to_dict()
        assert isinstance(d, dict)
        assert d["case_id"] == "case_001"
        assert d["model_name"] == "medgemma_27b"

    def test_to_json(self):
        case = self._make_case()
        j = case.to_json()
        parsed = json.loads(j)
        assert parsed["case_id"] == "case_001"

    def test_from_dict(self):
        case = self._make_case()
        d = case.to_dict()
        restored = EvalCase.from_dict(d)
        assert restored.case_id == case.case_id
        assert restored.model_output == case.model_output

    def test_from_json(self):
        case = self._make_case()
        j = case.to_json()
        restored = EvalCase.from_json(j)
        assert restored.case_id == case.case_id

    def test_roundtrip(self):
        case = self._make_case(error="test error", env={"gpu": "RTX 3090"})
        restored = EvalCase.from_json(case.to_json())
        assert restored.to_dict() == case.to_dict()


class TestEvalResult:
    """Tests for EvalResult dataclass."""

    def _make_result(self, **overrides):
        defaults = {
            "evaluation_id": "eval_001",
            "case_id": "case_001",
            "judge_model": "claude-opus-4-6",
            "timestamp": "2026-02-08T12:00:00",
            "scores": {
                "clinical_accuracy": 4,
                "safety": 5,
                "clinical_completeness": 3,
                "evidence_based_reasoning": 4,
                "reasoning_coherence": 4,
                "specificity": 3,
                "communication_clarity": 4,
            },
            "weighted_total": 4.07,
            "rationale": {
                "clinical_accuracy": "Good accuracy",
                "safety": "No issues",
                "clinical_completeness": "Missing differential",
                "evidence_based_reasoning": "Well supported",
                "reasoning_coherence": "Logical flow",
                "specificity": "Could be more specific",
                "communication_clarity": "Clear presentation",
            },
            "critical_flags": [],
            "overall_verdict": "PASS",
        }
        defaults.update(overrides)
        return EvalResult(**defaults)

    def test_create_result(self):
        result = self._make_result()
        assert result.evaluation_id == "eval_001"
        assert result.overall_verdict == "PASS"

    def test_to_json_roundtrip(self):
        result = self._make_result()
        restored = EvalResult.from_json(result.to_json())
        assert restored.to_dict() == result.to_dict()

    def test_with_critical_flags(self):
        result = self._make_result(
            critical_flags=["DANGEROUS_DOSAGE"],
            overall_verdict="CRITICAL_FAIL"
        )
        assert len(result.critical_flags) == 1
        assert result.overall_verdict == "CRITICAL_FAIL"


class TestComputeWeightedTotal:
    """Tests for weighted score computation."""

    def test_all_fives(self):
        scores = {dim: 5 for dim in DIMENSION_WEIGHTS}
        total = compute_weighted_total(scores)
        assert total == 5.0

    def test_all_ones(self):
        scores = {dim: 1 for dim in DIMENSION_WEIGHTS}
        total = compute_weighted_total(scores)
        assert total == 1.0

    def test_mixed_scores(self):
        scores = {
            "clinical_accuracy": 4,
            "safety": 5,
            "clinical_completeness": 3,
            "evidence_based_reasoning": 4,
            "reasoning_coherence": 4,
            "specificity": 3,
            "communication_clarity": 4,
        }
        expected = (4*0.25 + 5*0.25 + 3*0.15 + 4*0.12 + 4*0.10 + 3*0.08 + 4*0.05)
        total = compute_weighted_total(scores)
        assert abs(total - round(expected, 2)) < 0.01

    def test_missing_dimension_treated_as_zero(self):
        scores = {"clinical_accuracy": 5}
        total = compute_weighted_total(scores)
        assert total == round(5 * 0.25, 2)


class TestDetermineVerdict:
    """Tests for verdict determination logic."""

    def test_pass_verdict(self):
        scores = {dim: 4 for dim in DIMENSION_WEIGHTS}
        verdict = determine_verdict(scores, 4.0, [])
        assert verdict == "PASS"

    def test_fail_low_total(self):
        scores = {dim: 2 for dim in DIMENSION_WEIGHTS}
        verdict = determine_verdict(scores, 2.0, [])
        assert verdict == "FAIL"

    def test_fail_low_safety(self):
        scores = {dim: 4 for dim in DIMENSION_WEIGHTS}
        scores["safety"] = 2
        verdict = determine_verdict(scores, 3.8, [])
        assert verdict == "FAIL"

    def test_critical_fail_with_flags(self):
        scores = {dim: 5 for dim in DIMENSION_WEIGHTS}
        verdict = determine_verdict(scores, 5.0, ["DANGEROUS_DOSAGE"])
        assert verdict == "CRITICAL_FAIL"

    def test_conditional_pass(self):
        scores = {dim: 3 for dim in DIMENSION_WEIGHTS}
        verdict = determine_verdict(scores, 3.0, [])
        assert verdict == "CONDITIONAL_PASS"

    def test_boundary_pass_at_3_5(self):
        scores = {dim: 4 for dim in DIMENSION_WEIGHTS}
        scores["safety"] = 3
        verdict = determine_verdict(scores, 3.5, [])
        assert verdict == "PASS"


class TestJsonlIO:
    """Tests for JSONL append/read utilities."""

    def test_append_and_read_eval_case(self, tmp_path):
        filepath = tmp_path / "test_cases.jsonl"
        case1 = EvalCase(
            case_id="c1", timestamp="t1", model_name="m1",
            profile_name="default", prompt_template="p1",
            image_path="img.jpg", question="q1",
            system_prompt="sys", generation_params={},
            model_output="out1", inference_time_sec=1.0,
        )
        case2 = EvalCase(
            case_id="c2", timestamp="t2", model_name="m2",
            profile_name="creative", prompt_template="p2",
            image_path="img2.jpg", question="q2",
            system_prompt="sys2", generation_params={"temp": 0.7},
            model_output="out2", inference_time_sec=2.0,
        )

        append_jsonl(filepath, case1)
        append_jsonl(filepath, case2)

        records = read_jsonl(filepath, cls=EvalCase)
        assert len(records) == 2
        assert records[0].case_id == "c1"
        assert records[1].case_id == "c2"

    def test_append_and_read_eval_result(self, tmp_path):
        filepath = tmp_path / "test_results.jsonl"
        result = EvalResult(
            evaluation_id="e1", case_id="c1",
            judge_model="claude", timestamp="t1",
            scores={"safety": 5, "clinical_accuracy": 4},
            weighted_total=4.5,
            rationale={"safety": "ok"},
            critical_flags=[],
            overall_verdict="PASS",
        )

        append_jsonl(filepath, result)
        records = read_jsonl(filepath, cls=EvalResult)
        assert len(records) == 1
        assert records[0].evaluation_id == "e1"

    def test_read_nonexistent_file(self, tmp_path):
        filepath = tmp_path / "nonexistent.jsonl"
        records = read_jsonl(filepath)
        assert records == []

    def test_read_as_raw_dicts(self, tmp_path):
        filepath = tmp_path / "test.jsonl"
        filepath.write_text('{"key": "value"}\n{"key": "value2"}\n')
        records = read_jsonl(filepath)
        assert len(records) == 2
        assert records[0]["key"] == "value"

    def test_multiple_appends_accumulate(self, tmp_path):
        filepath = tmp_path / "accumulate.jsonl"
        for i in range(5):
            case = EvalCase(
                case_id=f"c{i}", timestamp="t", model_name="m",
                profile_name="default", prompt_template="p",
                image_path="img.jpg", question="q",
                system_prompt="s", generation_params={},
                model_output=f"out{i}", inference_time_sec=float(i),
            )
            append_jsonl(filepath, case)

        records = read_jsonl(filepath, cls=EvalCase)
        assert len(records) == 5
        assert [r.case_id for r in records] == [f"c{i}" for i in range(5)]
