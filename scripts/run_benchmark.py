#!/usr/bin/env python3
"""
CLI Benchmark Script for Vision Model Evaluation.

Runs inference across model × profile × prompt × image × question combinations,
writes structured JSONL output for LLM-as-Judge evaluation.

Usage:
    # Specific combinations
    python scripts/run_benchmark.py \
        --model medgemma_27b \
        --profile default creative \
        --prompt clinician_v1 structured_output \
        --images uploads/test_wound_1.jpg \
        --questions "Describe the wound" "Assess infection risk" \
        --output eval_data/run_20260208.jsonl

    # Sweep all profiles × prompts for a model
    python scripts/run_benchmark.py \
        --model hulumed --sweep \
        --images uploads/ \
        --output eval_data/hulumed_sweep.jsonl

    # Dry-run (validate config only, no model loading)
    python scripts/run_benchmark.py --model medgemma_27b --dry-run
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path

import yaml

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.engine.test_models import (
    build_inference_config,
    load_model_profiles,
    load_prompt_templates,
    run_inference,
    InferenceConfig,
    MODEL_NAME_MAP,
)
from src.evaluation.schemas import EvalCase, append_jsonl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def get_env_metadata(seed: int | None = None) -> dict:
    """Collect environment metadata for reproducibility."""
    import torch

    env = {
        "timestamp": datetime.now().isoformat(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    # Git commit
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=5
        )
        if result.returncode == 0:
            env["git_commit"] = result.stdout.strip()
    except Exception:
        pass

    # GPU info
    if torch.cuda.is_available():
        env["gpu_name"] = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info()
        env["gpu_vram_gb"] = round(total / (1024**3), 1)

    if seed is not None:
        env["seed"] = seed

    return env


def collect_images(image_args: list[str]) -> list[str]:
    """Resolve image paths: expand directories, validate files exist."""
    images = []
    for path in image_args:
        p = Path(path)
        if p.is_dir():
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                images.extend(sorted(str(f) for f in p.glob(ext)))
        elif p.is_file():
            images.append(str(p))
        else:
            logger.warning(f"Image path not found: {path}")
    return images


def load_questions_from_yaml(yaml_path: str) -> list[str]:
    """Load benchmark questions from YAML, prepending uncertainty instruction."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    uncertainty = data.get("uncertainty_instruction", "").strip()
    questions: list[str] = []
    for q in data["questions"]:
        text = q["question"].strip()
        if uncertainty:
            text = f"{uncertainty}\n\n{text}"
        questions.append(text)
    return questions


def load_images_from_manifest(manifest_path: str) -> list[str]:
    """Load image paths from a subset manifest YAML.

    Resolves relative paths based on the manifest file's directory.
    Validates that all image files exist.
    """
    manifest_dir = Path(manifest_path).parent
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    images: list[str] = []
    for sample in manifest["samples"]:
        img_path = Path(sample["image_path"])
        # Resolve relative paths based on manifest directory
        if not img_path.is_absolute():
            img_path = manifest_dir / img_path
        img_path = img_path.resolve()
        if not img_path.exists():
            raise FileNotFoundError(
                f"Image not found: {img_path} (from manifest {manifest_path})"
            )
        images.append(str(img_path))
    return images


def load_model_for_benchmark(model_name: str):
    """Load a model and return (processor, model). For Ollama models, returns (None, None)."""
    profiles = load_model_profiles()
    model_cfg = profiles[model_name]

    if model_cfg["backend"] == "ollama":
        logger.info(f"Ollama model '{model_name}' — no Python model to load")
        return None, None

    # Import loaders
    from src.engine.test_models import (
        load_medgemma_27b,
        load_medgemma_4b,
        load_hulumed,
        master_evict_with_retry,
    )

    # Evict existing models
    logger.info("Evicting existing models from VRAM...")
    master_evict_with_retry(required_free_gb=18.0, max_retries=5)

    # Load the requested model
    loaders = {
        "medgemma_27b": load_medgemma_27b,
        "medgemma_4b": load_medgemma_4b,
        "hulumed": load_hulumed,
    }

    loader = loaders.get(model_name)
    if loader is None:
        raise ValueError(f"No loader for model '{model_name}'")

    logger.info(f"Loading model '{model_name}'...")
    processor, model = loader()

    if processor is None or model is None:
        raise RuntimeError(f"Failed to load model '{model_name}'")

    return processor, model


def run_benchmark(args):
    """Main benchmark execution loop."""
    profiles_data = load_model_profiles()
    prompts_data = load_prompt_templates()

    model_name = args.model
    if model_name not in profiles_data:
        logger.error(f"Unknown model: {model_name}. Available: {list(profiles_data.keys())}")
        sys.exit(1)

    model_cfg = profiles_data[model_name]

    # Determine profile and prompt combinations
    if args.sweep:
        profile_names = list(model_cfg["profiles"].keys())
        prompt_names = list(prompts_data.keys())
    else:
        profile_names = args.profile or ["default"]
        prompt_names = args.prompt or ["clinician_v1"]

    # Validate profiles and prompts
    for p in profile_names:
        if p not in model_cfg["profiles"]:
            logger.error(f"Unknown profile '{p}' for model '{model_name}'")
            sys.exit(1)
    for p in prompt_names:
        if p not in prompts_data:
            logger.error(f"Unknown prompt template '{p}'")
            sys.exit(1)

    # Collect images
    images = collect_images(args.images or [])

    # Load images from manifest if specified (overrides --images)
    if args.dataset_manifest:
        images = load_images_from_manifest(args.dataset_manifest)
        logger.info(f"Loaded {len(images)} images from manifest {args.dataset_manifest}")

    if not images and not args.dry_run:
        logger.error("No images found. Use --images or --dataset-manifest to specify.")
        sys.exit(1)

    questions = args.questions or ["Describe this wound and provide a clinical assessment."]

    # Load questions from YAML if specified (overrides --questions)
    if args.questions_from:
        questions = load_questions_from_yaml(args.questions_from)
        logger.info(f"Loaded {len(questions)} questions from {args.questions_from}")

    # Build all combinations
    combos = list(product(profile_names, prompt_names, images or ["<dry-run>"], questions))
    total_cases = len(combos)

    if args.max_cases and args.max_cases < total_cases:
        combos = combos[:args.max_cases]
        logger.info(f"Limiting to {args.max_cases} cases (out of {total_cases})")

    # Summary
    logger.info("=" * 60)
    logger.info(f"Benchmark Configuration:")
    logger.info(f"  Model:     {model_name}")
    logger.info(f"  Profiles:  {profile_names}")
    logger.info(f"  Prompts:   {prompt_names}")
    logger.info(f"  Images:    {len(images)} file(s)")
    logger.info(f"  Questions: {len(questions)}")
    logger.info(f"  Total cases: {len(combos)}")
    logger.info(f"  Output:    {args.output}")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("DRY RUN — validating configurations only")
        for profile, prompt, _, _ in combos:
            config = build_inference_config(model_name, profile, prompt)
            logger.info(f"  OK: {model_name}/{profile}/{prompt}")
        logger.info("All configurations valid.")
        return

    # Set seed if specified
    if args.seed is not None:
        import torch
        import random
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        logger.info(f"Random seed set to {args.seed}")

    # Load model
    processor, model = load_model_for_benchmark(model_name)
    env_meta = get_env_metadata(seed=args.seed)

    # Run inference
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed = 0
    errors = 0

    # Set up timeout handler (Unix only)
    timeout_sec = args.timeout

    class _InferenceTimeout(Exception):
        pass

    def _timeout_handler(signum, frame):
        raise _InferenceTimeout(f"Inference timed out after {timeout_sec}s")

    prev_handler = signal.signal(signal.SIGALRM, _timeout_handler)

    for i, (profile, prompt, image_path, question) in enumerate(combos):
        case_id = f"case_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i:04d}"
        logger.info(f"[{i+1}/{len(combos)}] {model_name}/{profile}/{prompt} — {Path(image_path).name}")

        config = build_inference_config(model_name, profile, prompt)

        # Enforce per-inference timeout
        signal.alarm(timeout_sec)
        try:
            result = run_inference(
                image_path=image_path,
                question=question,
                config=config,
                processor=processor,
                model=model,
            )
        except _InferenceTimeout as e:
            from src.engine.test_models import InferenceResult
            result = InferenceResult(
                model_name=model_name,
                profile_name=profile,
                prompt_template=prompt,
                output=f"Error: {e}",
                inference_time_sec=float(timeout_sec),
                generate_params=config.generate_params,
                system_prompt=config.system_prompt,
                error=str(e),
            )
            logger.warning(f"  TIMEOUT: {e}")
        finally:
            signal.alarm(0)

        # Merge generate_params + backend_options for complete parameter recording
        recorded_params = {**config.generate_params, **config.backend_options}

        case = EvalCase(
            case_id=case_id,
            timestamp=datetime.now().isoformat(),
            model_name=model_name,
            profile_name=profile,
            prompt_template=prompt,
            image_path=str(image_path),
            question=question,
            system_prompt=config.system_prompt,
            generation_params=recorded_params,
            model_output=result.output,
            inference_time_sec=result.inference_time_sec,
            error=result.error,
            env=env_meta,
        )

        append_jsonl(output_path, case)
        completed += 1

        if result.error:
            errors += 1
            logger.warning(f"  Error: {result.error}")
        else:
            logger.info(f"  Done in {result.inference_time_sec:.1f}s ({len(result.output)} chars)")

    # Restore previous signal handler
    signal.signal(signal.SIGALRM, prev_handler)

    # Summary
    logger.info("=" * 60)
    logger.info(f"Benchmark complete: {completed} cases, {errors} errors")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Run vision model benchmark for evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True,
                        help="Model config name (medgemma_27b, medgemma_4b, hulumed, gemma3)")
    parser.add_argument("--profile", nargs="+",
                        help="Profile name(s) to test (default: ['default'])")
    parser.add_argument("--prompt", nargs="+",
                        help="Prompt template name(s) (default: ['clinician_v1'])")
    parser.add_argument("--images", nargs="+",
                        help="Image file paths or directories")
    parser.add_argument("--dataset-manifest", default=None,
                        help="Load images from subset manifest YAML (e.g., data/datasets/WoundcareVQA/subset_mini/manifest.yaml)")
    parser.add_argument("--questions", nargs="+",
                        help="Question(s) to ask about each image")
    parser.add_argument("--questions-from", default=None,
                        help="Load questions from YAML file (e.g., config/benchmark_questions.yaml)")
    parser.add_argument("--output", default="eval_data/benchmark_output.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep all profiles × prompts for the model")
    parser.add_argument("--max-cases", type=int, default=None,
                        help="Maximum number of cases to run")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Per-inference timeout in seconds (default: 300)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate configurations without loading models")

    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
