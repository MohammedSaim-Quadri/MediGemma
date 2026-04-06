"""
Standalone Streamlit app for viewing model evaluation results.
Run: streamlit run src/interface/eval_viewer.py --server.port 8502
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import yaml

# --- Project root ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.schemas import DIMENSION_WEIGHTS

EVAL_DATA_DIR = PROJECT_ROOT / "eval_data"
BENCHMARK_QUESTIONS_PATH = PROJECT_ROOT / "config" / "benchmark_questions.yaml"

DIMENSION_ORDER: list[str] = list(DIMENSION_WEIGHTS.keys())

DIMENSION_SHORT_NAMES: dict[str, str] = {
    "clinical_accuracy": "Accuracy",
    "safety": "Safety",
    "clinical_completeness": "Completeness",
    "evidence_based_reasoning": "Evidence",
    "reasoning_coherence": "Coherence",
    "specificity": "Specificity",
    "communication_clarity": "Clarity",
}

VERDICT_COLORS: dict[str, str] = {
    "PASS": "#28a745",
    "CONDITIONAL_PASS": "#ffc107",
    "FAIL": "#dc3545",
    "CRITICAL_FAIL": "#721c24",
}

VERDICT_TEXT_COLORS: dict[str, str] = {
    "PASS": "#ffffff",
    "CONDITIONAL_PASS": "#000000",
    "FAIL": "#ffffff",
    "CRITICAL_FAIL": "#ffffff",
}

BENCHMARK_QUALITY_DIMS: list[str] = [
    "relevance", "conciseness", "readability", "uncertainty_handling",
]


# ============================================================
# Pure functions (safe to import in tests, no Streamlit calls)
# ============================================================

def score_to_color(score: int | float) -> str:
    """Map a 1-5 score to a hex color."""
    score = round(score)
    return {
        5: "#28a745",
        4: "#8bc34a",
        3: "#ffc107",
        2: "#ff9800",
        1: "#dc3545",
    }.get(score, "#6c757d")


def load_question_labels() -> dict[str, str]:
    """Load Q-id -> module label mapping from benchmark_questions.yaml.

    Returns: {"Q1": "Wound Identification & Classification", ...}
    """
    if not BENCHMARK_QUESTIONS_PATH.exists():
        return {}
    with open(BENCHMARK_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    labels = {}
    for q in cfg.get("questions", []):
        labels[q["id"]] = q["module"]
    return labels


def load_question_details() -> list[dict]:
    """Load full question details from benchmark_questions.yaml.

    Returns list of {"id": "Q1", "report_section": "Classification", "question": "..."}.
    """
    if not BENCHMARK_QUESTIONS_PATH.exists():
        return []
    with open(BENCHMARK_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    details = []
    for q in cfg.get("questions", []):
        details.append({
            "id": q["id"],
            "report_section": q.get("report_section", ""),
            "question": q.get("question", "").strip(),
        })
    return details


def _build_question_prefix_map(labels: dict[str, str]) -> dict[str, tuple[str, str]]:
    """Build prefix -> (qid, label) lookup from benchmark_questions.yaml."""
    if not BENCHMARK_QUESTIONS_PATH.exists():
        return {}
    with open(BENCHMARK_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    prefix_map = {}
    for q in cfg.get("questions", []):
        qid = q["id"]
        text = q["question"].strip()
        prefix = text[:40]
        prefix_map[prefix] = (qid, labels.get(qid, qid))
    return prefix_map


_PREFIX_MAP: dict | None = None


def _get_prefix_map(labels: dict[str, str]) -> dict[str, tuple[str, str]]:
    global _PREFIX_MAP
    if _PREFIX_MAP is None:
        _PREFIX_MAP = _build_question_prefix_map(labels)
    return _PREFIX_MAP


def detect_question_label(question_text: str, labels: dict[str, str]) -> tuple[str, str]:
    """Match question text to (Q-id, module label) using prefix matching."""
    prefix_map = _get_prefix_map(labels)
    for prefix, (qid, label) in prefix_map.items():
        if prefix in question_text:
            return qid, label
    return "Q?", "Unknown Question"


def get_subdirectories(directory: Path | None = None) -> list[Path]:
    """List subdirectories inside eval_data/ that contain .jsonl files."""
    d = directory or EVAL_DATA_DIR
    if not d.exists():
        return []
    return sorted(
        p for p in d.iterdir()
        if p.is_dir() and any(p.glob("*.jsonl"))
    )


def get_jsonl_files(directory: Path | None = None) -> list[Path]:
    """List all .jsonl files in the given directory."""
    d = directory or EVAL_DATA_DIR
    if not d.exists():
        return []
    return sorted(d.glob("*.jsonl"))


def get_inference_files(directory: Path | None = None) -> list[Path]:
    """List inference JSONL files (exclude eval results)."""
    all_files = get_jsonl_files(directory)
    return [f for f in all_files
            if "_eval_results" not in f.name]


def get_eval_results_files(directory: Path | None = None) -> list[Path]:
    """List eval results JSONL files."""
    all_files = get_jsonl_files(directory)
    return [f for f in all_files if f.name.endswith("_eval_results.jsonl")]


def find_matching_eval_file(inference_path: Path) -> Path | None:
    """Given an inference file like gemma3_mini.jsonl, find gemma3_eval_results.jsonl."""
    stem = inference_path.stem  # e.g. "gemma3_mini"
    # Remove common suffixes to extract model prefix
    for suffix in ("_mini", "_full", "_benchmark"):
        if stem.endswith(suffix):
            model_prefix = stem[: -len(suffix)]
            break
    else:
        model_prefix = stem

    eval_path = inference_path.parent / f"{model_prefix}_eval_results.jsonl"
    if eval_path.exists():
        return eval_path
    return None


def load_eval_data_raw(filepath: Path) -> list[dict]:
    """Load JSONL records as raw dicts, skipping bad lines."""
    if not filepath.exists():
        return []
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def build_eval_lookup(eval_records: list[dict]) -> dict[str, dict]:
    """Build case_id -> eval_record lookup from eval results."""
    return {r["case_id"]: r for r in eval_records if "case_id" in r}


def resolve_image_path(image_path: str) -> Path:
    """Resolve image path (absolute or relative to PROJECT_ROOT)."""
    p = Path(image_path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def is_safe_image_path(image_path: str) -> bool:
    """Check that image path is within the project directory."""
    try:
        resolved = resolve_image_path(image_path)
        return str(resolved).startswith(str(PROJECT_ROOT.resolve()))
    except (ValueError, OSError):
        return False


def compute_summary(records: list[dict]) -> dict:
    """Compute summary statistics from records."""
    total = len(records)
    errors = sum(1 for r in records if r.get("error"))
    times = [r.get("inference_time_sec", 0) for r in records if not r.get("error")]
    avg_time = sum(times) / max(len(times), 1)
    unique_images = len(set(r.get("image_path", "") for r in records))
    model_name = records[0].get("model_name", "Unknown") if records else "Unknown"
    env = records[0].get("env", {}) if records else {}
    return {
        "model_name": model_name,
        "total": total,
        "errors": errors,
        "avg_time": round(avg_time, 2),
        "unique_images": unique_images,
        "env": env,
    }


# ============================================================
# Streamlit UI (all st.* calls inside main())
# ============================================================

def main():
    """Streamlit UI entry point."""
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    st.set_page_config(
        page_title="Eval Results Viewer",
        page_icon="📊",
        layout="wide",
    )

    st.markdown("""
    <style>
    [data-testid="stMetric"] {
        background-color: #f0f2f6;
        padding: 12px 16px;
        border-radius: 8px;
    }
    .block-container { padding-top: 2rem; }
    .verdict-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85em;
        letter-spacing: 0.02em;
    }
    .score-bar-container {
        display: flex;
        align-items: center;
        gap: 6px;
        margin: 2px 0;
    }
    .score-bar-label {
        font-size: 0.78em;
        width: 80px;
        text-align: right;
        color: #555;
    }
    .score-bar {
        height: 14px;
        border-radius: 3px;
        display: inline-block;
    }
    .score-bar-value {
        font-size: 0.78em;
        font-weight: 600;
        min-width: 20px;
    }
    .critical-flag {
        background-color: #f8d7da;
        color: #721c24;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85em;
        margin: 2px 0;
        display: inline-block;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("📊 Evaluation Results Viewer")

    # --- Load question labels ---
    q_labels = load_question_labels()

    # --- Cached loaders ---
    @st.cache_data
    def _load(path: str) -> list[dict]:
        return load_eval_data_raw(Path(path))

    @st.cache_data
    def _load_eval(path: str) -> dict[str, dict]:
        records = load_eval_data_raw(Path(path))
        return build_eval_lookup(records)

    @st.cache_data
    def _load_all_eval_results() -> dict[str, list[dict]]:
        """Load all eval results files, keyed by model name."""
        result: dict[str, list[dict]] = {}
        for f in get_eval_results_files():
            records = load_eval_data_raw(f)
            if records:
                model = records[0].get("model_name", f.stem)
                result[model] = records
        return result

    # --- Data directory: use latest subdirectory if available ---
    subdirs = get_subdirectories()
    selected_dir = subdirs[0] if subdirs else EVAL_DATA_DIR

    # --- Sidebar: Compare Models toggle ---
    st.sidebar.markdown("#### Mode")
    compare_mode = st.sidebar.toggle("Compare Models", value=False)

    @st.cache_data
    def _load_all_eval_results_from(dir_path: str) -> dict[str, list[dict]]:
        """Load all eval results files from a specific directory."""
        result: dict[str, list[dict]] = {}
        for f in get_eval_results_files(Path(dir_path)):
            records = load_eval_data_raw(f)
            if records:
                model = records[0].get("model_name", f.stem)
                result[model] = records
        return result

    if compare_mode:
        load_fn = lambda: _load_all_eval_results_from(str(selected_dir))
        _render_model_comparison(st, pd, px, go, q_labels, load_fn)
        return

    # --- File Selector (sidebar) ---
    inference_files = get_inference_files(selected_dir)
    if not inference_files:
        st.error(f"No inference .jsonl files found in {dir_labels.get(selected_dir, selected_dir)}")
        st.stop()

    selected_file = st.sidebar.selectbox(
        "Select inference file",
        inference_files,
        format_func=lambda f: f.name,
    )

    # --- Load inference data ---
    records = _load(str(selected_file))
    if not records:
        st.warning("Selected file is empty.")
        st.stop()

    # --- Auto-load matching eval results ---
    eval_file = find_matching_eval_file(selected_file)
    eval_lookup: dict[str, dict] = {}
    if eval_file is not None:
        eval_lookup = _load_eval(str(eval_file))

    has_eval = len(eval_lookup) > 0

    # --- Summary Metrics ---
    summary = compute_summary(records)
    profile = records[0].get("profile_name", "default")
    st.markdown(f"### {summary['model_name']}  ·  profile: `{profile}`")
    cols = st.columns(5 if has_eval else 4)
    cols[0].metric("Total Cases", summary["total"])
    cols[1].metric("Errors", summary["errors"],
                   delta=None if summary["errors"] == 0 else f"-{summary['errors']}",
                   delta_color="inverse")
    cols[2].metric("Avg Time", f"{summary['avg_time']:.1f}s")
    cols[3].metric("Images", summary["unique_images"])

    if has_eval:
        eval_records_list = list(eval_lookup.values())
        avg_weighted = sum(
            r.get("weighted_total", 0) for r in eval_records_list
        ) / max(len(eval_records_list), 1)
        cols[4].metric("Avg Eval Score", f"{avg_weighted:.2f}/5")

    # --- Env Info ---
    env = summary["env"]
    if env:
        parts = []
        if env.get("gpu_name"):
            parts.append(f"**GPU:** {env['gpu_name']}")
        if env.get("gpu_vram_gb"):
            parts.append(f"**VRAM:** {env['gpu_vram_gb']}GB")
        if env.get("torch_version"):
            parts.append(f"**PyTorch:** {env['torch_version']}")
        if env.get("git_commit"):
            parts.append(f"**Commit:** `{env['git_commit']}`")
        if parts:
            st.caption(" · ".join(parts))

    if has_eval:
        eval_file_name = eval_file.name if eval_file else "N/A"
        st.caption(f"Eval results loaded from: `{eval_file_name}` "
                   f"({len(eval_lookup)} records)")

    # --- Sidebar: Benchmark Questions Reference ---
    st.sidebar.divider()
    st.sidebar.markdown("#### Benchmark Questions")
    q_details = load_question_details()
    for qd in q_details:
        with st.sidebar.expander(f"**{qd['id']}** — {qd['report_section']}"):
            st.caption(qd["question"])

    # --- Sidebar: Generation Params ---
    st.sidebar.divider()
    st.sidebar.markdown("#### Generation Config")
    gen_params = records[0].get("generation_params", {})
    for k, v in gen_params.items():
        st.sidebar.text(f"{k}: {v}")

    st.sidebar.divider()
    st.sidebar.markdown("#### System Prompt")
    sys_prompt = records[0].get("system_prompt", "N/A")
    st.sidebar.text_area("sys_prompt_display", sys_prompt, height=150,
                         disabled=True, label_visibility="collapsed")

    st.divider()

    # --- Group records by image ---
    by_image: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_image[r.get("image_path", "unknown")].append(r)

    for img_path in by_image:
        by_image[img_path].sort(
            key=lambda r: detect_question_label(r.get("question", ""), q_labels)[0]
        )

    # --- Image tabs ---
    image_paths = sorted(by_image.keys())
    image_names = [Path(p).stem for p in image_paths]

    if not image_names:
        st.warning("No image groups found.")
        st.stop()

    tabs = st.tabs(image_names)

    for tab, img_path in zip(tabs, image_paths):
        with tab:
            img_records = by_image[img_path]
            left_col, right_col = st.columns([1, 2])

            with left_col:
                resolved_img = resolve_image_path(img_path)
                if is_safe_image_path(img_path) and resolved_img.exists():
                    st.image(str(resolved_img), caption=resolved_img.name,
                             width="stretch")
                else:
                    st.warning(f"Image not found or outside project: {Path(img_path).name}")

            with right_col:
                for r in img_records:
                    qid, qlabel = detect_question_label(
                        r.get("question", ""), q_labels
                    )
                    has_error = bool(r.get("error"))
                    case_id = r.get("case_id", "")
                    eval_rec = eval_lookup.get(case_id)

                    header = f"{qid}: {qlabel}"
                    if has_error:
                        header += " -- ERROR"
                    else:
                        t = r.get("inference_time_sec", 0)
                        header += f"  |  {t:.1f}s"
                        if eval_rec is not None:
                            verdict = eval_rec.get("overall_verdict", "?")
                            wt = eval_rec.get("weighted_total", 0)
                            header += f"  |  {verdict} ({wt:.2f})"

                    with st.expander(header, expanded=(qid == "Q1")):
                        if has_error:
                            st.error(f"**Error:** {r['error']}")
                        else:
                            # --- Eval scores above the output ---
                            if eval_rec is not None:
                                _render_eval_scores_inline(
                                    st, eval_rec
                                )
                                st.markdown("---")

                            st.markdown(r.get("model_output", "*(empty)*"))

                            # --- Rationale sub-expander ---
                            if eval_rec is not None:
                                rationale = eval_rec.get("rationale", {})
                                if rationale:
                                    with st.expander("Rationale (per dimension)"):
                                        for dim in DIMENSION_ORDER:
                                            text = rationale.get(dim, "")
                                            if text:
                                                short = DIMENSION_SHORT_NAMES.get(dim, dim)
                                                st.markdown(
                                                    f"**{short}:** {text}"
                                                )

    # --- Inference Time Chart ---
    st.divider()
    st.markdown("### Inference Time Analysis")

    chart_data = []
    for r in records:
        if not r.get("error"):
            qid, _ = detect_question_label(r.get("question", ""), q_labels)
            chart_data.append({
                "Question": qid,
                "Image": Path(r.get("image_path", "unknown")).stem,
                "Time (s)": r.get("inference_time_sec", 0),
            })

    if chart_data:
        df = pd.DataFrame(chart_data)

        avg_by_q = df.groupby("Question")["Time (s)"].mean().reset_index()
        avg_by_q = avg_by_q.sort_values("Question")
        fig = px.bar(
            avg_by_q, x="Question", y="Time (s)",
            title="Average Inference Time per Question",
            text_auto=".1f",
        )
        fig.update_layout(height=350, showlegend=False,
                          margin=dict(t=30))
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, width="stretch")

        col1, col2 = st.columns(2)
        with col1:
            st.caption("Per-question stats")
            stats = df.groupby("Question")["Time (s)"].agg(
                ["mean", "min", "max"]
            ).round(2)
            stats.columns = ["Avg (s)", "Min (s)", "Max (s)"]
            st.dataframe(stats, width="stretch")
        with col2:
            st.caption("Per-image stats")
            img_stats = df.groupby("Image")["Time (s)"].agg(
                ["mean", "sum", "count"]
            ).round(2)
            img_stats.columns = ["Avg (s)", "Total (s)", "Questions"]
            st.dataframe(img_stats, width="stretch")
    else:
        st.info("No successful inference records to chart.")

    # --- Evaluation Summary Section ---
    if has_eval:
        st.divider()
        _render_evaluation_summary(st, pd, px, go, eval_lookup, q_labels)

    # --- Error Log ---
    error_records = [r for r in records if r.get("error")]
    if error_records:
        st.divider()
        st.markdown("### Error Log")
        for r in error_records:
            qid, _ = detect_question_label(r.get("question", ""), q_labels)
            img_name = Path(r.get("image_path", "unknown")).stem
            st.error(f"**{img_name} / {qid}:** {r['error']}")


def _render_verdict_badge(st_module, verdict: str) -> None:
    """Render a colored verdict badge using st.markdown with HTML."""
    bg = VERDICT_COLORS.get(verdict, "#6c757d")
    fg = VERDICT_TEXT_COLORS.get(verdict, "#ffffff")
    display = verdict.replace("_", " ")
    st_module.markdown(
        f'<span class="verdict-badge" style="background-color:{bg};color:{fg};">'
        f'{display}</span>',
        unsafe_allow_html=True,
    )


def _render_eval_scores_inline(st_module, eval_rec: dict) -> None:
    """Render eval scores compactly inside a question expander."""
    verdict = eval_rec.get("overall_verdict", "?")
    weighted_total = eval_rec.get("weighted_total", 0)
    scores = eval_rec.get("scores", {})
    critical_flags = eval_rec.get("critical_flags", [])

    # Row 1: Verdict badge + weighted total
    c1, c2, c3 = st_module.columns([1, 1, 2])
    with c1:
        _render_verdict_badge(st_module, verdict)
    with c2:
        st_module.metric("Weighted Total", f"{weighted_total:.2f}/5")
    with c3:
        if critical_flags:
            flags_html = " ".join(
                f'<span class="critical-flag">{f}</span>'
                for f in critical_flags
            )
            st_module.markdown(flags_html, unsafe_allow_html=True)

    # Row 2: Dimension score bars
    bar_html_parts = []
    for dim in DIMENSION_ORDER:
        score = scores.get(dim)
        short = DIMENSION_SHORT_NAMES.get(dim, dim)
        weight = DIMENSION_WEIGHTS.get(dim, 0)
        if score is None:
            bar_html_parts.append(
                f'<div class="score-bar-container">'
                f'<span class="score-bar-label">{short} ({weight:.0%})</span>'
                f'<span class="score-bar-value" style="color:#aaa;">N/A</span>'
                f'</div>'
            )
        else:
            color = score_to_color(score)
            width_pct = max(score / 5 * 100, 2)
            bar_html_parts.append(
                f'<div class="score-bar-container">'
                f'<span class="score-bar-label">{short} ({weight:.0%})</span>'
                f'<span class="score-bar" style="width:{width_pct}%;'
                f'background-color:{color};"></span>'
                f'<span class="score-bar-value" style="color:{color};">'
                f'{score}</span>'
                f'</div>'
            )
    st_module.markdown("".join(bar_html_parts), unsafe_allow_html=True)

    # Row 3: Benchmark quality (compact)
    bq = eval_rec.get("benchmark_quality", {})
    if bq:
        bq_parts = []
        for dim in BENCHMARK_QUALITY_DIMS:
            val = bq.get(dim, "?")
            bq_parts.append(f"{dim}: **{val}**")
        st_module.caption("Benchmark quality: " + " | ".join(bq_parts))


def _render_evaluation_summary(st_module, pd, px, go, eval_lookup, q_labels):
    """Render the Evaluation Summary section with charts."""
    st_module.markdown("### Evaluation Summary")

    eval_records = list(eval_lookup.values())
    if not eval_records:
        st_module.info("No evaluation records available.")
        return

    # --- Row 1: Verdict distribution + Dimension radar ---
    col_left, col_right = st_module.columns(2)

    with col_left:
        # Verdict distribution pie chart
        verdict_counts: dict[str, int] = defaultdict(int)
        for r in eval_records:
            v = r.get("overall_verdict", "UNKNOWN")
            verdict_counts[v] += 1

        verdict_df = pd.DataFrame([
            {"Verdict": v, "Count": c}
            for v, c in sorted(verdict_counts.items())
        ])
        color_map = {**VERDICT_COLORS, "UNKNOWN": "#6c757d"}
        fig_verdict = px.pie(
            verdict_df, names="Verdict", values="Count",
            title="Verdict Distribution",
            color="Verdict",
            color_discrete_map=color_map,
        )
        fig_verdict.update_layout(height=380,
                                   uniformtext_minsize=10, uniformtext_mode="hide")
        fig_verdict.update_traces(
            textposition="auto", textinfo="value+percent"
        )
        st_module.plotly_chart(fig_verdict, width="stretch")

    with col_right:
        # Radar chart of average dimension scores
        dim_avgs = {}
        for dim in DIMENSION_ORDER:
            vals = [r.get("scores", {}).get(dim) for r in eval_records
                    if r.get("scores", {}).get(dim) is not None]
            dim_avgs[dim] = sum(vals) / max(len(vals), 1) if vals else 0

        theta = [DIMENSION_SHORT_NAMES.get(d, d) for d in DIMENSION_ORDER]
        r_vals = [dim_avgs[d] for d in DIMENSION_ORDER]
        # Close the polygon
        theta_closed = theta + [theta[0]]
        r_closed = r_vals + [r_vals[0]]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=r_closed,
            theta=theta_closed,
            fill="toself",
            name="Avg Score",
            line=dict(color="#4e79a7", width=2),
            fillcolor="rgba(78, 121, 167, 0.25)",
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 5], tickvals=[1, 2, 3, 4, 5]),
            ),
            title="Average Dimension Scores",
            height=380,
            showlegend=False,
        )
        st_module.plotly_chart(fig_radar, width="stretch")

    # --- Row 2: Per-question weighted total + benchmark quality ---
    col_left2, col_right2 = st_module.columns(2)

    with col_left2:
        # Per-question average weighted_total
        q_scores: dict[str, list[float]] = defaultdict(list)
        for r in eval_records:
            qid = r.get("question_id", "Q?")
            q_scores[qid].append(r.get("weighted_total", 0))

        q_avg_data = []
        for qid in sorted(q_scores.keys()):
            vals = q_scores[qid]
            avg = sum(vals) / max(len(vals), 1)
            label = q_labels.get(qid, qid)
            q_avg_data.append({
                "Question": qid,
                "Label": f"{qid}: {label}",
                "Avg Weighted Total": round(avg, 2),
            })

        if q_avg_data:
            q_df = pd.DataFrame(q_avg_data)
            fig_q = px.bar(
                q_df, x="Question", y="Avg Weighted Total",
                title="Average Weighted Total by Question",
                text_auto=".2f",
                hover_data=["Label"],
                color="Avg Weighted Total",
                color_continuous_scale=["#dc3545", "#ffc107", "#28a745"],
                range_color=[1, 5],
            )
            fig_q.update_layout(height=380, showlegend=False,
                                yaxis=dict(range=[0, 5.5]))
            fig_q.update_traces(textposition="outside")
            fig_q.update_coloraxes(showscale=False)
            st_module.plotly_chart(fig_q, width="stretch")

    with col_right2:
        # Benchmark quality averages
        bq_avgs: dict[str, float] = {}
        for dim in BENCHMARK_QUALITY_DIMS:
            vals = [
                r.get("benchmark_quality", {}).get(dim, 0)
                for r in eval_records
                if r.get("benchmark_quality", {}).get(dim) is not None
            ]
            bq_avgs[dim] = sum(vals) / max(len(vals), 1)

        bq_df = pd.DataFrame([
            {"Dimension": d.replace("_", " ").title(), "Avg Score": round(v, 2)}
            for d, v in bq_avgs.items()
        ])
        if not bq_df.empty:
            fig_bq = px.bar(
                bq_df, x="Dimension", y="Avg Score",
                title="Benchmark Quality Averages",
                text_auto=".2f",
                color="Avg Score",
                color_continuous_scale=["#dc3545", "#ffc107", "#28a745"],
                range_color=[1, 5],
            )
            fig_bq.update_layout(height=380, showlegend=False,
                                yaxis=dict(range=[0, 5.5]))
            fig_bq.update_traces(textposition="outside")
            fig_bq.update_coloraxes(showscale=False)
            st_module.plotly_chart(fig_bq, width="stretch")

    # --- Dimension score distribution table ---
    st_module.caption("Dimension score statistics")
    dim_stats_data = []
    for dim in DIMENSION_ORDER:
        vals = [r.get("scores", {}).get(dim) for r in eval_records
                if r.get("scores", {}).get(dim) is not None]
        short = DIMENSION_SHORT_NAMES.get(dim, dim)
        weight = DIMENSION_WEIGHTS.get(dim, 0)
        if vals:
            dim_stats_data.append({
                "Dimension": f"{short} ({weight:.0%})",
                "Avg": round(sum(vals) / len(vals), 2),
                "Min": min(vals),
                "Max": max(vals),
                "Median": round(sorted(vals)[len(vals) // 2], 2),
                "N": len(vals),
            })
        else:
            dim_stats_data.append({
                "Dimension": f"{short} ({weight:.0%})",
                "Avg": "N/A", "Min": "N/A", "Max": "N/A",
                "Median": "N/A", "N": 0,
            })
    dim_stats_df = pd.DataFrame(dim_stats_data)
    st_module.dataframe(dim_stats_df, width="stretch", hide_index=True)


def _render_model_comparison(st_module, pd, px, go, q_labels, load_all_fn):
    """Render the model comparison view."""
    st_module.markdown("### Model Comparison")

    all_eval = load_all_fn()
    if not all_eval:
        st_module.warning("No eval results files found in eval_data/.")
        return

    model_names = sorted(all_eval.keys())
    st_module.caption(f"Comparing {len(model_names)} models: {', '.join(model_names)}")

    # --- 1. Dimension comparison table ---
    st_module.markdown("#### Dimension Score Comparison")
    table_data: list[dict] = []
    for dim in DIMENSION_ORDER:
        row: dict[str, object] = {
            "Dimension": DIMENSION_SHORT_NAMES.get(dim, dim),
            "Weight": f"{DIMENSION_WEIGHTS.get(dim, 0):.0%}",
        }
        for model in model_names:
            vals = [
                r.get("scores", {}).get(dim)
                for r in all_eval[model]
                if r.get("scores", {}).get(dim) is not None
            ]
            avg = round(sum(vals) / len(vals), 2) if vals else "N/A"
            row[model] = avg
        table_data.append(row)

    # Add weighted total row
    wt_row: dict[str, object] = {"Dimension": "Weighted Total", "Weight": "100%"}
    for model in model_names:
        vals = [r.get("weighted_total", 0) for r in all_eval[model]]
        avg = sum(vals) / max(len(vals), 1)
        wt_row[model] = round(avg, 2)
    table_data.append(wt_row)

    comp_df = pd.DataFrame(table_data)
    st_module.dataframe(comp_df, width="stretch", hide_index=True)

    # --- 2. Weighted total grouped bar chart ---
    col_left, col_right = st_module.columns(2)

    with col_left:
        st_module.markdown("#### Weighted Total by Model")
        bar_data = []
        for model in model_names:
            vals = [r.get("weighted_total", 0) for r in all_eval[model]]
            avg = sum(vals) / max(len(vals), 1)
            bar_data.append({
                "Model": model,
                "Avg Weighted Total": round(avg, 2),
            })
        bar_df = pd.DataFrame(bar_data)
        fig_bar = px.bar(
            bar_df, x="Model", y="Avg Weighted Total",
            text_auto=".2f",
            color="Model",
        )
        fig_bar.update_layout(
            height=400, showlegend=False,
            yaxis=dict(range=[0, 5.3]),
            margin=dict(t=30),
        )
        fig_bar.update_traces(textposition="outside")
        st_module.plotly_chart(fig_bar, width="stretch")

    with col_right:
        # Radar chart overlay
        st_module.markdown("#### Dimension Radar Overlay")
        fig_radar = go.Figure()
        colors = px.colors.qualitative.Plotly
        for idx, model in enumerate(model_names):
            dim_avgs = {}
            for dim in DIMENSION_ORDER:
                vals = [r.get("scores", {}).get(dim) for r in all_eval[model]
                        if r.get("scores", {}).get(dim) is not None]
                dim_avgs[dim] = sum(vals) / len(vals) if vals else 0

            theta = [DIMENSION_SHORT_NAMES.get(d, d) for d in DIMENSION_ORDER]
            r_vals = [dim_avgs[d] for d in DIMENSION_ORDER]
            theta_closed = theta + [theta[0]]
            r_closed = r_vals + [r_vals[0]]
            color = colors[idx % len(colors)]

            fig_radar.add_trace(go.Scatterpolar(
                r=r_closed,
                theta=theta_closed,
                fill="toself",
                name=model,
                line=dict(width=2, color=color),
                opacity=0.7,
            ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 5], tickvals=[1, 2, 3, 4, 5]),
            ),
            height=400,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        )
        st_module.plotly_chart(fig_radar, width="stretch")

    # --- 3. Verdict distribution comparison ---
    st_module.markdown("#### Verdict Distribution Comparison")
    verdict_data = []
    for model in model_names:
        counts: dict[str, int] = defaultdict(int)
        for r in all_eval[model]:
            v = r.get("overall_verdict", "UNKNOWN")
            counts[v] += 1
        for verdict, count in counts.items():
            verdict_data.append({
                "Model": model,
                "Verdict": verdict,
                "Count": count,
            })

    if verdict_data:
        verdict_df = pd.DataFrame(verdict_data)
        color_map = {**VERDICT_COLORS, "UNKNOWN": "#6c757d"}
        fig_verdict = px.bar(
            verdict_df, x="Model", y="Count", color="Verdict",
            barmode="group",
            color_discrete_map=color_map,
            text_auto=True,
        )
        fig_verdict.update_layout(height=400, margin=dict(t=30))
        fig_verdict.update_traces(textposition="outside")
        st_module.plotly_chart(fig_verdict, width="stretch")

    # --- 4. Per-question comparison ---
    st_module.markdown("#### Per-Question Comparison")

    # Collect per-question per-model averages
    pq_data = []
    for model in model_names:
        q_scores: dict[str, list[float]] = defaultdict(list)
        for r in all_eval[model]:
            qid = r.get("question_id", "Q?")
            q_scores[qid].append(r.get("weighted_total", 0))
        for qid in sorted(q_scores.keys()):
            vals = q_scores[qid]
            avg = sum(vals) / max(len(vals), 1)
            pq_data.append({
                "Question": qid,
                "Model": model,
                "Avg Weighted Total": round(avg, 2),
            })

    if pq_data:
        pq_df = pd.DataFrame(pq_data)
        fig_pq = px.bar(
            pq_df, x="Question", y="Avg Weighted Total", color="Model",
            barmode="group",
            text_auto=".2f",
            title="Weighted Total by Question and Model",
        )
        fig_pq.update_layout(
            height=500,
            yaxis=dict(range=[0, 5.5]),
            margin=dict(t=40),
        )
        fig_pq.update_traces(textposition="outside", textfont_size=8)
        st_module.plotly_chart(fig_pq, width="stretch")

    # Best model per question table
    st_module.caption("Best model per question")
    best_data = []
    # Pivot: question -> model -> avg
    q_model_avgs: dict[str, dict[str, float]] = defaultdict(dict)
    for row in pq_data:
        q_model_avgs[row["Question"]][row["Model"]] = row["Avg Weighted Total"]

    for qid in sorted(q_model_avgs.keys()):
        model_scores = q_model_avgs[qid]
        best_model = max(model_scores, key=lambda m: model_scores[m])
        best_score = model_scores[best_model]
        label = q_labels.get(qid, qid)
        best_data.append({
            "Question": qid,
            "Label": label,
            "Best Model": best_model,
            "Score": best_score,
        })

    if best_data:
        best_df = pd.DataFrame(best_data)
        st_module.dataframe(best_df, width="stretch", hide_index=True)

    # --- 5. Benchmark quality comparison ---
    st_module.markdown("#### Benchmark Quality Comparison")
    bq_data = []
    for model in model_names:
        for dim in BENCHMARK_QUALITY_DIMS:
            vals = [
                r.get("benchmark_quality", {}).get(dim, 0)
                for r in all_eval[model]
                if r.get("benchmark_quality", {}).get(dim) is not None
            ]
            avg = sum(vals) / max(len(vals), 1)
            bq_data.append({
                "Dimension": dim.replace("_", " ").title(),
                "Model": model,
                "Avg Score": round(avg, 2),
            })

    if bq_data:
        bq_df = pd.DataFrame(bq_data)
        fig_bq = px.bar(
            bq_df, x="Dimension", y="Avg Score", color="Model",
            barmode="group",
            text_auto=".2f",
            title="Benchmark Quality by Model",
        )
        fig_bq.update_layout(
            height=450,
            yaxis=dict(range=[0, 5.8]),
            margin=dict(t=40),
        )
        fig_bq.update_traces(textposition="outside", textfont_size=9)
        st_module.plotly_chart(fig_bq, width="stretch")


if __name__ == "__main__":
    main()
