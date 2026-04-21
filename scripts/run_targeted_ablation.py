#!/usr/bin/env python3
"""
Run per-model targeted ablations and append intermediate results to a progress doc.

For each (model, profile, prompt) config:
1) Build model-specific filtered question YAML by question IDs.
2) Run scripts/run_benchmark.py.
3) Run scripts/llm_judge.py on produced JSONL.
4) Append metrics to a markdown progress document immediately.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run targeted per-model ablation and log progress.")
    parser.add_argument(
        "--matrix",
        default="config/targeted_model_ablation_matrix.yaml",
        help="YAML matrix mapping model -> list[{profile,prompt}]",
    )
    parser.add_argument(
        "--model-question-map",
        default="config/targeted_model_questions.yaml",
        help="YAML mapping model -> [Qx,...]",
    )
    parser.add_argument(
        "--questions-from",
        default="config/benchmark_questions.yaml",
        help="Source benchmark question YAML",
    )
    parser.add_argument(
        "--dataset-manifest",
        default="data/datasets/WoundcareVQA/subset_mini/manifest.yaml",
        help="Dataset manifest YAML",
    )
    parser.add_argument(
        "--output-dir",
        default="eval_data/targeted_ablation",
        help="Directory for inference/eval outputs",
    )
    parser.add_argument(
        "--progress-doc",
        default="docs/plans/2026-02-09-targeted-ablation-progress.md",
        help="Markdown file to append intermediate summaries",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for benchmark runs")
    parser.add_argument("--timeout", type=int, default=300, help="Per-inference timeout seconds")
    parser.add_argument("--dry-run", action="store_true", help="Validate commands only")
    return parser.parse_args()


def load_yaml(path: Path) -> object:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_filtered_questions(questions_yaml: Path, question_ids: list[str]) -> dict:
    data = load_yaml(questions_yaml)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid questions YAML: {questions_yaml}")

    all_questions = data.get("questions", [])
    wanted = {q.strip().upper() for q in question_ids}
    filtered = [q for q in all_questions if str(q.get("id", "")).upper() in wanted]

    if not filtered:
        raise ValueError(f"No questions matched IDs {question_ids} in {questions_yaml}")

    return {
        "version": data.get("version", "1.0"),
        "total_questions": len(filtered),
        "uncertainty_instruction": data.get("uncertainty_instruction", ""),
        "questions": filtered,
    }


def init_progress_doc(path: Path, matrix_path: Path, qmap_path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return

    now = datetime.now().isoformat(timespec="seconds")
    lines = [
        "# Targeted Ablation Progress",
        "",
        f"- Start: `{now}`",
        f"- Matrix: `{matrix_path}`",
        f"- Model question map: `{qmap_path}`",
        f"- Seed: `{seed}`",
        "",
        "## Runs",
        "",
        "| Time | Model | Profile | Prompt | Questions | Cases | PASS | CRITICAL_FAIL | Weighted Avg | Safety Avg | Avg Sec | Status | Output |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def append_progress_row(path: Path, row: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(row + "\n")


def run_cmd(cmd: list[str]) -> int:
    print(f"[RUN] {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


def summarize_run(infer_path: Path, eval_path: Path) -> dict:
    infer_rows = [json.loads(l) for l in infer_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    eval_rows = [json.loads(l) for l in eval_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    pass_count = sum(1 for r in eval_rows if r.get("overall_verdict") == "PASS")
    critical_fail = sum(1 for r in eval_rows if r.get("overall_verdict") == "CRITICAL_FAIL")
    weighted_avg = mean(r.get("weighted_total", 0.0) for r in eval_rows)
    safety_avg = mean(r.get("scores", {}).get("safety", 0.0) for r in eval_rows)
    avg_sec = mean(r.get("inference_time_sec", 0.0) for r in infer_rows)
    flags = Counter(f for r in eval_rows for f in r.get("critical_flags", []))
    by_q = defaultdict(list)
    for r in eval_rows:
        by_q[r.get("question_id", "Q?")].append(r.get("weighted_total", 0.0))
    low_q = sorted(((q, mean(vals)) for q, vals in by_q.items()), key=lambda x: x[1])[:2]

    return {
        "cases": len(eval_rows),
        "pass_count": pass_count,
        "critical_fail": critical_fail,
        "weighted_avg": weighted_avg,
        "safety_avg": safety_avg,
        "avg_sec": avg_sec,
        "flags": dict(flags),
        "lowest_questions": low_q,
    }


def main() -> int:
    args = parse_args()
    matrix_path = Path(args.matrix)
    qmap_path = Path(args.model_question_map)
    questions_src = Path(args.questions_from)
    output_dir = Path(args.output_dir)
    progress_doc = Path(args.progress_doc)

    matrix = load_yaml(matrix_path)
    model_questions = load_yaml(qmap_path)
    if not isinstance(matrix, dict) or not isinstance(model_questions, dict):
        raise ValueError("Matrix and model-question map must both be YAML mappings.")

    output_dir.mkdir(parents=True, exist_ok=True)
    init_progress_doc(progress_doc, matrix_path, qmap_path, args.seed)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    failures: list[str] = []
    for model, configs in matrix.items():
        if model not in model_questions:
            failures.append(f"{model}: missing question IDs in {qmap_path}")
            continue
        if not isinstance(configs, list):
            failures.append(f"{model}: matrix entry must be a list")
            continue
        question_ids = [str(q) for q in model_questions[model]]

        filtered_data = load_filtered_questions(questions_src, question_ids)
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=f"_{model}_ablation_questions.yaml",
            prefix="benchmark_",
            delete=False,
        ) as tmp:
            yaml.safe_dump(filtered_data, tmp, sort_keys=False, allow_unicode=True)
            filtered_yaml = Path(tmp.name)

        for cfg in configs:
            profile = str(cfg["profile"])
            prompt = str(cfg["prompt"])
            stamp = datetime.now().isoformat(timespec="seconds")
            base = f"{model}_{profile}_{prompt}_{run_ts}".replace("/", "_")
            infer_path = output_dir / f"{base}.jsonl"

            bench_cmd = [
                sys.executable,
                "scripts/run_benchmark.py",
                "--model",
                model,
                "--profile",
                profile,
                "--prompt",
                prompt,
                "--dataset-manifest",
                str(args.dataset_manifest),
                "--questions-from",
                str(filtered_yaml),
                "--output",
                str(infer_path),
                "--timeout",
                str(args.timeout),
                "--seed",
                str(args.seed),
            ]
            if args.dry_run:
                bench_cmd.append("--dry-run")

            code = run_cmd(bench_cmd)
            if code != 0:
                row = (
                    f"| {stamp} | {model} | {profile} | {prompt} | {','.join(question_ids)} "
                    f"| - | - | - | - | - | - | BENCH_FAIL({code}) | `{infer_path}` |"
                )
                append_progress_row(progress_doc, row)
                failures.append(f"{model}/{profile}/{prompt}: benchmark failed ({code})")
                continue

            if args.dry_run:
                row = (
                    f"| {stamp} | {model} | {profile} | {prompt} | {','.join(question_ids)} "
                    f"| - | - | - | - | - | - | DRY_RUN_OK | `{infer_path}` |"
                )
                append_progress_row(progress_doc, row)
                continue

            judge_cmd = [
                sys.executable,
                "scripts/llm_judge.py",
                str(infer_path),
            ]
            code = run_cmd(judge_cmd)
            eval_path = infer_path.with_name(f"{infer_path.stem}_eval_results.jsonl")
            if code != 0 or not eval_path.exists():
                row = (
                    f"| {stamp} | {model} | {profile} | {prompt} | {','.join(question_ids)} "
                    f"| - | - | - | - | - | - | JUDGE_FAIL({code}) | `{infer_path}` |"
                )
                append_progress_row(progress_doc, row)
                failures.append(f"{model}/{profile}/{prompt}: judge failed ({code})")
                continue

            s = summarize_run(infer_path, eval_path)
            row = (
                f"| {stamp} | {model} | {profile} | {prompt} | {','.join(question_ids)} "
                f"| {s['cases']} | {s['pass_count']} | {s['critical_fail']} "
                f"| {s['weighted_avg']:.2f} | {s['safety_avg']:.2f} | {s['avg_sec']:.2f} "
                f"| OK | `{infer_path}` |"
            )
            append_progress_row(progress_doc, row)

            low_q_text = ", ".join(f"{q}:{v:.2f}" for q, v in s["lowest_questions"])
            flags_text = ", ".join(f"{k}:{v}" for k, v in s["flags"].items()) or "none"
            append_progress_row(
                progress_doc,
                f"  - notes `{model}/{profile}/{prompt}` low_q=({low_q_text}) flags=({flags_text})",
            )

    end_stamp = datetime.now().isoformat(timespec="seconds")
    append_progress_row(progress_doc, "")
    append_progress_row(progress_doc, f"- End: `{end_stamp}`")
    append_progress_row(progress_doc, f"- Failures: `{len(failures)}`")
    for item in failures:
        append_progress_row(progress_doc, f"  - {item}")

    if failures:
        print("[ERROR] Some runs failed:")
        for item in failures:
            print(" -", item)
        return 1

    print("[DONE] All ablation runs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

