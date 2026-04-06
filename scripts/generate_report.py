#!/usr/bin/env python3
"""
Report Generator for Vision Model Evaluation Results.

Reads eval_results.jsonl (+ optional eval_outputs.jsonl for metadata),
generates a Markdown comparison report grouped by model × profile × prompt.

Usage:
    python scripts/generate_report.py \
        --results eval_data/eval_results.jsonl \
        --cases eval_data/eval_outputs.jsonl \
        --output eval_data/report.md
"""

import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.schemas import EvalCase, EvalResult, read_jsonl, DIMENSION_WEIGHTS


def _avg(values: list) -> float:
    return sum(values) / len(values) if values else 0.0


def generate_report(results_path: str, cases_path: str | None, output_path: str):
    """Generate a Markdown comparison report from evaluation results."""
    results = read_jsonl(results_path, cls=EvalResult)

    if not results:
        print(f"No results found in {results_path}")
        sys.exit(1)

    print(f"Loaded {len(results)} evaluation results")

    # Build case_id → EvalCase lookup for grouping metadata
    case_lookup: dict[str, EvalCase] = {}
    if cases_path:
        cases = read_jsonl(cases_path, cls=EvalCase)
        case_lookup = {c.case_id: c for c in cases}
        print(f"Loaded {len(cases)} case records for grouping")

    # Group results by (model, profile, prompt)
    groups: dict[tuple, list[EvalResult]] = defaultdict(list)
    for r in results:
        case = case_lookup.get(r.case_id)
        if case:
            key = (case.model_name, case.profile_name, case.prompt_template)
        else:
            key = ("unknown", "unknown", "unknown")
        groups[key].append(r)

    dimensions = list(DIMENSION_WEIGHTS.keys())

    lines = []
    lines.append("# Evaluation Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total evaluations: {len(results)}")
    lines.append(f"Groups: {len(groups)} (model × profile × prompt)")
    lines.append("")

    # ── Overall verdict distribution ──
    verdict_counts = defaultdict(int)
    for r in results:
        verdict_counts[r.overall_verdict] += 1

    lines.append("## Verdict Distribution")
    lines.append("")
    lines.append("| Verdict | Count | Percentage |")
    lines.append("|---------|-------|------------|")
    for verdict in ["PASS", "CONDITIONAL_PASS", "FAIL", "CRITICAL_FAIL"]:
        count = verdict_counts.get(verdict, 0)
        pct = (count / len(results) * 100) if results else 0
        lines.append(f"| {verdict} | {count} | {pct:.1f}% |")
    lines.append("")

    # ── Per-group comparison table ──
    lines.append("## Configuration Comparison")
    lines.append("")

    # Header
    header = "| Model | Profile | Prompt | N |"
    separator = "|-------|---------|--------|---|"
    for dim in dimensions:
        short = dim.replace("_", " ").title()
        header += f" {short} |"
        separator += "---|"
    header += " Weighted |"
    separator += "---|"
    lines.append(header)
    lines.append(separator)

    # Rows sorted by weighted total descending
    group_avgs = []
    for (model, profile, prompt), group_results in groups.items():
        avg_scores = {}
        for dim in dimensions:
            avg_scores[dim] = _avg([r.scores.get(dim, 0) for r in group_results])
        avg_total = _avg([r.weighted_total for r in group_results])
        group_avgs.append((model, profile, prompt, len(group_results), avg_scores, avg_total))

    group_avgs.sort(key=lambda x: x[5], reverse=True)

    for model, profile, prompt, n, avg_scores, avg_total in group_avgs:
        row = f"| {model} | {profile} | {prompt} | {n} |"
        for dim in dimensions:
            row += f" {avg_scores[dim]:.1f} |"
        row += f" **{avg_total:.2f}** |"
        lines.append(row)
    lines.append("")

    # ── Overall average scores per dimension ──
    lines.append("## Average Scores by Dimension (All Groups)")
    lines.append("")
    lines.append("| Dimension | Weight | Avg Score | Min | Max |")
    lines.append("|-----------|--------|-----------|-----|-----|")

    for dim in dimensions:
        weight = DIMENSION_WEIGHTS[dim]
        scores = [r.scores.get(dim, 0) for r in results]
        if scores:
            avg = _avg(scores)
            mn = min(scores)
            mx = max(scores)
        else:
            avg = mn = mx = 0
        lines.append(f"| {dim} | {weight} | {avg:.2f} | {mn} | {mx} |")

    totals = [r.weighted_total for r in results]
    if totals:
        avg_total = _avg(totals)
        lines.append("")
        lines.append(f"**Average Weighted Total: {avg_total:.2f}**")
    lines.append("")

    # ── Critical flags summary ──
    all_flags = []
    for r in results:
        all_flags.extend(r.critical_flags)

    lines.append("## Critical Safety Flags")
    lines.append("")
    if all_flags:
        flag_counts = defaultdict(int)
        for f in all_flags:
            flag_counts[f] += 1
        lines.append("| Flag | Count |")
        lines.append("|------|-------|")
        for flag, count in sorted(flag_counts.items(), key=lambda x: -x[1]):
            lines.append(f"| {flag} | {count} |")
    else:
        lines.append("No critical safety flags triggered.")
    lines.append("")

    # ── Best and worst cases ──
    sorted_results = sorted(results, key=lambda r: r.weighted_total, reverse=True)

    lines.append("## Top 5 Best Scoring Cases")
    lines.append("")
    lines.append("| Case ID | Weighted Total | Verdict |")
    lines.append("|---------|----------------|---------|")
    for r in sorted_results[:5]:
        lines.append(f"| {r.case_id} | {r.weighted_total:.2f} | {r.overall_verdict} |")
    lines.append("")

    lines.append("## Top 5 Worst Scoring Cases")
    lines.append("")
    lines.append("| Case ID | Weighted Total | Verdict |")
    lines.append("|---------|----------------|---------|")
    for r in sorted_results[-5:]:
        lines.append(f"| {r.case_id} | {r.weighted_total:.2f} | {r.overall_verdict} |")
    lines.append("")

    # ── Per-group verdict breakdown ──
    lines.append("## Per-Configuration Verdict Breakdown")
    lines.append("")
    lines.append("| Model | Profile | Prompt | PASS | COND_PASS | FAIL | CRIT_FAIL |")
    lines.append("|-------|---------|--------|------|-----------|------|-----------|")
    for model, profile, prompt, n, _, _ in group_avgs:
        group_results = groups[(model, profile, prompt)]
        vc = defaultdict(int)
        for r in group_results:
            vc[r.overall_verdict] += 1
        lines.append(
            f"| {model} | {profile} | {prompt} "
            f"| {vc.get('PASS', 0)} | {vc.get('CONDITIONAL_PASS', 0)} "
            f"| {vc.get('FAIL', 0)} | {vc.get('CRITICAL_FAIL', 0)} |"
        )
    lines.append("")

    # Write report
    report = "\n".join(lines)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(report, encoding="utf-8")

    print(f"Report written to {output_path}")
    print(f"  Total evaluations: {len(results)}")
    print(f"  Groups: {len(groups)}")
    if totals:
        print(f"  Average weighted total: {avg_total:.2f}")
    print(f"  Verdicts: {dict(verdict_counts)}")


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation comparison report")
    parser.add_argument("--results", required=True,
                        help="Input eval_results.jsonl file")
    parser.add_argument("--cases", default=None,
                        help="Input eval_outputs.jsonl file (for model/profile/prompt grouping)")
    parser.add_argument("--output", default="eval_data/report.md",
                        help="Output Markdown file")

    args = parser.parse_args()
    generate_report(args.results, args.cases, args.output)


if __name__ == "__main__":
    main()
