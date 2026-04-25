<div align="center">

# 🩺 Medi-Gemma: Multimodal Clinical Decision Support System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c?logo=pytorch&logoColor=white)](#)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.14-black)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?logo=streamlit&logoColor=white)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#)

**An Agentic RAG and Vision-Language Model (VLM) platform engineered for safe, hallucination-free wound pathology triage and clinical workflow automation.**

</div>

---

## 📖 Overview

**Medi-Gemma** is a full-stack, multimodal Clinical Decision Support System (CDSS) designed to assist medical directors and bedside physicians. By combining fine-tuned Vision-Language Models (VLMs) with Agentic Retrieval-Augmented Generation (RAG), the system processes both visual evidence (wound images) and tabular patient histories (EMR CSVs) to generate evidence-based treatment protocols.

To bridge the gap between AI research and healthcare production, this architecture mimics FDA-cleared CDSS pipelines: AI handles perception, while deterministic rule engines and strict clinical rubrics handle safety-critical triage.

## 🏗️ System Architecture

The pipeline operates in a hybrid, multi-stage architecture:

1. **Stage 1: AI Perception (Visual Diagnosis)**
   - **Model:** Fine-Tuned LLaVA-v1.5-7B (`LLaVA-Medical-Director`), loaded with 4-bit NF4 quantization.
   - **Training:** Fine-tuned on ~4,000 images across diverse datasets (AZH Wound, DFUC, Medetec, WoundcareVQA).
   - **Function:** Analyzes wound morphology, tissue composition (granulation vs. slough/eschar), and identifies visual signs of infection.

2. **Stage 2: Clinical Action (Protocol Mapping & Safety Gate)**
   - **Function:** A deterministic `ProtocolManager` intercepts the VLM's raw output and performs keyword-based mapping of identified pathology (e.g., Diabetic Foot Ulcer, Venous Leg Ulcer, Necrotizing Infection) to a structured, tiered protocol defined in `config/protocols.yaml`.
   - **Impact:** Actively prevents LLM hallucinations from reaching the treatment planning stage by grounding the pipeline in auditable, rule-based logic.

3. **Stage 3: Agentic RAG & Medical Director Logic**
   - **Models:** Gemma 3 27B (via Ollama), MedGemma 27B, MedGemma-1.5 4B, Hulu-Med 32B (all via HuggingFace Transformers except Gemma 3).
   - **Function:** Fuses the visual protocol with the patient's historical EMR data using local embeddings (`BAAI/bge-small-en-v1.5`, running on CPU) and LlamaIndex. A `ClinicalOrchestrator` routes queries between the `AnalyticsEngine` (for data aggregation) and the `ClinicalRAGEngine` (for patient-specific clinical reasoning).

## ✨ Key Features

* **👨‍⚕️ Medical Director Dashboard:** A real-time triage console powered by a `PandasQueryEngine` (backed by Ollama). It autonomously scans patient cohorts, flagging deteriorating wounds or severe pain levels, and prioritizes patients into Critical, Urgent, and Stable queues.
* **💬 Multimodal Chat Interface:** Clinicians can upload wound images, optionally link them to a Patient ID, and query the system. The `ClinicalOrchestrator` synthesizes the visual findings with the patient's past comorbidities and wound dimensions from the uploaded EMR CSV.
* **🧠 Explainability Console:** A live reasoning log that surfaces the exact patient records (`source_nodes`) the RAG engine retrieved to generate each response, ensuring full clinical transparency.
* **🛡️ Safety Verifier:** A keyword-based output filter (`SafetyVerifier`) that blocks responses containing a set of banned phrases before they are displayed to the clinician.

## 📊 Benchmarking & VLM Evaluation Framework

A core component of this repository is its rigorous, automated LLM-as-a-Judge evaluation pipeline (`scripts/evaluate_medgemma27b.py`). Models are evaluated against a **9-question clinical benchmark** (`config/benchmark_questions.yaml`) using a strict **7-dimension scoring rubric**:
*(Clinical Accuracy, Safety, Clinical Completeness, Evidence-Based Reasoning, Reasoning Coherence, Specificity, Communication Clarity).*

The judge model used for evaluation is **Claude Opus 4.6** (`claude-opus-4-6`). Critical binary safety flags — including `MISSED_EMERGENCY`, `DANGEROUS_DOSAGE`, `CONTRAINDICATED_TREATMENT`, `FABRICATED_DATA`, and `PROMPT_INJECTION` — override numerical scores and trigger an automatic `CRITICAL_FAIL` verdict.

### Recent Evaluation Results
* **MedGemma-1.5 4B:** Achieved a **63/63 benchmark pass rate** with fast inference (~6.5s load time), utilizing a targeted anti-refusal prompt (`clinician_v3_mg4b`).
* **MedGemma 27B & Gemma 3:** Validated at **0 `CRITICAL_FAIL`s** across the dataset, establishing them as the safest baseline models for deployment.
* **Hulu-Med 32B:** Successfully reduced `MISSED_EMERGENCY` triage flags from 3 to 1 through the implementation of the step-by-step `thinking` decoding profile (`use_think: true`).

## 🛠️ Tech Stack

* **Machine Learning:** PyTorch, HuggingFace Transformers, BitsAndBytes (4-bit quantization for LLaVA, MedGemma 27B, and Hulu-Med 32B; full bfloat16 for MedGemma-1.5 4B)
* **LLM & RAG:** LlamaIndex (`llama-index-core==0.14`), Ollama (Gemma 3 backend), LLaVA (local fine-tuned model)
* **Data Processing:** Pandas, Pillow
* **Training Pipeline:** OpenCV (mask analysis in `phase3_training/` only)
* **Frontend:** Streamlit
* **Performance:** Cython (for `engine_core.py` binary compilation)

## 🚀 Installation & Setup

**1. Clone the repository:**
```bash
git clone https://github.com/MohammedSaim-Quadri/medigemma.git
cd medigemma
```

**2. Set up the environment:**
It is highly recommended to use a virtual environment (Conda or venv).
```bash
pip install -r requirements.txt
```

> **Note:** Two `requirements.txt` variants exist in the repo. The root-level file contains loose/unpinned dependencies for easy setup. A fully pinned version (generated via `pip freeze`) is preserved separately for reproducibility.

**3. Compile Core Engine (Optional but recommended for speed):**
```bash
python build_release.py build_ext --inplace
```

**4. Start the Application:**
Ensure your `.env` is configured with any necessary local paths, then launch the Streamlit app:
```bash
./run.sh
```

## 🧪 Running the Benchmark Suite

To run the automated VLM benchmarking pipeline across your datasets:

```bash
# Run a specific model and profile combination
python scripts/run_benchmark.py \
    --model medgemma_27b \
    --profile default \
    --prompt clinician_v1 \
    --dataset-manifest data/datasets/WoundcareVQA/subset_mini/manifest.yaml \
    --output eval_data/run_results.jsonl

# Generate the Markdown evaluation report
python scripts/generate_report.py \
    --results eval_data/run_results.jsonl \
    --output eval_data/report.md
```

## 📂 Repository Structure

```text
medigemma/
├── config/                        # Model profiles, prompts, clinical protocols, and eval rubrics
│   ├── benchmark_questions.yaml
│   ├── benchmark_questions_fields.md
│   ├── datasets.md
│   ├── deployment_baselines.yaml
│   ├── eval_rubric.md
│   ├── model_load_times.json
│   ├── model_profiles.yaml
│   ├── prompts.yaml
│   ├── protocols.yaml
│   ├── style_guide.json
│   ├── targeted_model_ablation_matrix.yaml
│   └── targeted_model_questions.yaml
├── legacy_v1/                     # Earlier architecture iterations and modular tests
├── phase3_training/               # Scripts for fine-tuning and dataset formatting (LLaVA)
│   ├── robust_merge.py
│   ├── train.sh
│   ├── train_phase4.sh
│   └── scripts/
│       ├── format_data.py         # Uses OpenCV for wound mask analysis
│       └── format_p4_data.py
├── scripts/                       # Benchmarking, LLM-as-a-Judge eval, and report generation
│   ├── evaluate_medgemma27b.py
│   ├── generate_report.py
│   ├── generate_subset.py
│   ├── jsonl_to_markdown.py
│   ├── run_benchmark.py
│   ├── run_targeted_ablation.py
│   └── run_targeted_model_checks.py
├── src/
│   ├── core/                      # Orchestrator, Router, and Priority Triage rules
│   │   ├── orchestrator.py
│   │   ├── priority_rules.py
│   │   └── router.py
│   ├── engine/                    # RAG, LLM, Vision, and Analytics engines
│   │   ├── engine_core.py         # Compiled to binary via build_release.py
│   │   ├── load_timer.py
│   │   └── test_models.py         # HuggingFace model loading & inference helpers
│   ├── evaluation/                # Evaluation schemas and JSONL serialization
│   │   └── schemas.py
│   ├── interface/                 # Streamlit UI components and layout
│   │   ├── app_main.py
│   │   ├── copy_button.py
│   │   ├── eval_viewer.py
│   │   └── progress_timer.py
│   └── safety/                    # Protocol Manager and Safety Verifier
│       ├── protocol_manager.py
│       └── verifier.py
├── tests/                         # Unit and integration test suite (pytest)
│   ├── test_data_integrity.py
│   ├── test_eval_schemas.py
│   ├── test_eval_viewer.py
│   ├── test_generate_subset.py
│   ├── test_inference_config.py
│   ├── test_integration.py
│   ├── test_logic.py
│   ├── test_model_registry.py
│   ├── test_protocols.py
│   ├── test_run_benchmark_manifest.py
│   └── test_run_benchmark_questions.py
├── build_release.py               # Cython compilation script for engine_core.py
├── requirements.txt               # Project dependencies
└── run.sh                         # Main application launch script
```
