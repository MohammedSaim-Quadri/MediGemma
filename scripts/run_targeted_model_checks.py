#!/usr/bin/env python3
"""
Run targeted benchmark checks across multiple models.

This script filters benchmark questions by question IDs (e.g. Q5/Q8) and then
runs scripts/run_benchmark.py once per model with a consistent config.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import yaml


def load_filtered_questions(questions_yaml: Path, question_ids: list[str]) -> dict:
    """Load benchmark YAML and keep only selected question IDs."""
    data = yaml.safe_load(questions_yaml.read_text(encoding="utf-8"))
    all_questions = data.get("questions", [])

    wanted = {q.strip().upper() for q in question_ids}
    filtered = [q for q in all_questions if str(q.get("id", "")).upper() in wanted]

    if not filtered:
        raise ValueError(f"No questions matched IDs: {question_ids}")

    found_ids = {str(q.get("id", "")).upper() for q in filtered}
    missing = [qid for qid in wanted if qid not in found_ids]
    if missing:
        print(f"[WARN] Missing question IDs in source file: {missing}")

    return {
        "version": data.get("version", "1.0"),
        "total_questions": len(filtered),
        "uncertainty_instruction": data.get("uncertainty_instruction", ""),
        "questions": filtered,
    }


def load_model_question_map(path: Path) -> dict[str, list[str]]:
    """Load YAML map: {model_name: [Qx, ...]}."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid model-question map in {path}: expected mapping")

    result: dict[str, list[str]] = {}
    for model, ids in raw.items():
        if isinstance(ids, str):
            ids = [ids]
        if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
            raise ValueError(
                f"Invalid question list for model '{model}' in {path}: expected list[str]"
            )
        result[str(model)] = ids
    return result


def run_for_model(
    model: str,
    profile: str,
    prompt: str,
    dataset_manifest: Path,
    filtered_questions_yaml: Path,
    output_path: Path,
    timeout_sec: int,
    seed: int | None,
    dry_run: bool,
) -> int:
    """Run one benchmark command for a single model."""
    cmd = [
        sys.executable,
        "scripts/run_benchmark.py",
        "--model",
        model,
        "--profile",
        profile,
        "--prompt",
        prompt,
        "--dataset-manifest",
        str(dataset_manifest),
        "--questions-from",
        str(filtered_questions_yaml),
        "--output",
        str(output_path),
        "--timeout",
        str(timeout_sec),
    ]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    if dry_run:
        cmd.append("--dry-run")

    print(f"[RUN] {' '.join(cmd)}")
    completed = subprocess.run(cmd)
    return completed.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run targeted checks for multiple models with selected question IDs."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["medgemma_27b", "hulumed", "gemma3"],
        help="Models to test",
    )
    parser.add_argument(
        "--profile",
        default="default",
        help="Profile to use for all models (default: default)",
    )
    parser.add_argument(
        "--prompt",
        default="clinician_v1",
        help="Prompt template to use for all models (default: clinician_v1)",
    )
    parser.add_argument(
        "--question-ids",
        nargs="+",
        default=["Q3", "Q5", "Q6", "Q8"],
        help="Question IDs to keep from benchmark YAML",
    )
    parser.add_argument(
        "--model-question-map",
        default=None,
        help="Optional YAML mapping model->question IDs, e.g. {hulumed:[Q4,Q5,Q6,Q8]}",
    )
    parser.add_argument(
        "--questions-from",
        default="config/benchmark_questions.yaml",
        help="Source benchmark question YAML",
    )
    parser.add_argument(
        "--dataset-manifest",
        default="data/datasets/WoundcareVQA/subset_mini/manifest.yaml",
        help="Image manifest YAML",
    )
    parser.add_argument(
        "--output-dir",
        default="eval_data/targeted_checks",
        help="Directory for output JSONL files",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-inference timeout seconds passed to run_benchmark.py",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configurations only",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_question_map: dict[str, list[str]] = {}
    if args.model_question_map:
        model_question_map = load_model_question_map(Path(args.model_question_map))
        print(f"[INFO] Loaded model-question map from {args.model_question_map}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    failures: list[tuple[str, int]] = []

    for model in args.models:
        question_ids = model_question_map.get(model, args.question_ids)
        filtered_yaml_data = load_filtered_questions(Path(args.questions_from), question_ids)

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=f"_{model}_targeted_questions.yaml",
            prefix="benchmark_",
            delete=False,
        ) as tmp:
            yaml.safe_dump(filtered_yaml_data, tmp, sort_keys=False, allow_unicode=True)
            filtered_yaml = Path(tmp.name)

        print(f"[INFO] {model}: questions -> {', '.join(question_ids)}")
        print(f"[INFO] {model}: filtered question file -> {filtered_yaml}")

        output_path = output_dir / f"{model}_targeted_{timestamp}.jsonl"
        code = run_for_model(
            model=model,
            profile=args.profile,
            prompt=args.prompt,
            dataset_manifest=Path(args.dataset_manifest),
            filtered_questions_yaml=filtered_yaml,
            output_path=output_path,
            timeout_sec=args.timeout,
            seed=args.seed,
            dry_run=args.dry_run,
        )
        if code != 0:
            failures.append((model, code))
        else:
            print(f"[OK] {model} -> {output_path}")

    if failures:
        print("[ERROR] Some model runs failed:")
        for model, code in failures:
            print(f"  - {model}: exit code {code}")
        return 1

    print("[DONE] All targeted checks completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
