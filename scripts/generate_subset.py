#!/usr/bin/env python3
"""Generate a minimal WoundcareVQA subset: 1 sample per wound type,
maximizing clinical diversity (infection severity, wound depth)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Ranking tables
# ---------------------------------------------------------------------------

INFECTION_RANK: dict[str, int] = {
    "infected": 3,
    "unclear": 2,
    "not_infected": 1,
}

THICKNESS_RANK: dict[str, int] = {
    "stage_IV": 6,
    "stage_III": 5,
    "unstageable": 4,
    "stage_II": 3,
    "stage_I": 2,
    "not_applicable": 1,
}

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def _load_all_encounters(dataset_dir: str | Path) -> list[dict[str, Any]]:
    """Load encounters from all three split JSONs, adding a ``split`` field."""
    dataset_dir = Path(dataset_dir)
    encounters: list[dict[str, Any]] = []
    for split_name in ("train", "valid", "test"):
        json_path = dataset_dir / f"{split_name}.json"
        if not json_path.exists():
            continue
        with open(json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for enc in data:
            # The JSON may already carry a ``split`` field; override to be
            # consistent with the filename.
            enc["split"] = split_name
            encounters.append(enc)
    return encounters


def _pick_best_per_type(
    encounters: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return one encounter per wound_type, preferring diversity.

    Selection key (descending priority):
      1. Single-image encounter (preferred)
      2. Higher infection rank (infected > unclear > not_infected)
      3. Deeper wound thickness (stage_IV > ... > not_applicable)
      4. Encounter ID (deterministic tie-break)
    """
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for enc in encounters:
        by_type[enc["wound_type"]].append(enc)

    selected: list[dict[str, Any]] = []
    for wound_type in sorted(by_type):
        candidates = by_type[wound_type]
        best = max(
            candidates,
            key=lambda d: (
                len(d["image_ids"]) == 1,
                INFECTION_RANK.get(d["infection"], 0),
                THICKNESS_RANK.get(d["wound_thickness"], 0),
                d["encounter_id"],
            ),
        )
        selected.append(best)
    return selected


def generate_subset(
    dataset_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Build the mini subset and return the manifest dict.

    Steps:
      1. Load all encounters across splits.
      2. Pick the best encounter per wound type.
      3. Copy the first image of each selected encounter into
         ``{output_dir}/images/``.
      4. Write ``{output_dir}/manifest.yaml``.
    """
    dataset_dir = Path(dataset_dir).resolve()
    output_dir = Path(output_dir).resolve()

    encounters = _load_all_encounters(dataset_dir)
    if not encounters:
        raise ValueError(f"No encounters found in {dataset_dir}")

    selected = _pick_best_per_type(encounters)

    # Prepare output directories
    images_out = output_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    samples: list[dict[str, Any]] = []
    for enc in selected:
        first_image = enc["image_ids"][0]
        split = enc["split"]

        # Locate source image
        src_image = dataset_dir / f"images_{split}" / first_image
        dst_image = images_out / first_image

        if not src_image.exists():
            raise FileNotFoundError(
                f"Source image not found: {src_image} "
                f"(encounter {enc['encounter_id']}, split={split})"
            )
        shutil.copy2(src_image, dst_image)

        # Build expert responses list
        expert_responses: list[str] = []
        for resp in enc.get("responses", []):
            content = resp.get("content_en", "")
            if content:
                expert_responses.append(content)

        sample: dict[str, Any] = {
            "encounter_id": enc["encounter_id"],
            "split": split,
            "wound_type": enc["wound_type"],
            "image": first_image,
            "image_path": f"images/{first_image}",
            "ground_truth": {
                "wound_type": enc["wound_type"],
                "wound_thickness": enc["wound_thickness"],
                "tissue_color": enc["tissue_color"],
                "drainage_amount": enc["drainage_amount"],
                "drainage_type": enc["drainage_type"],
                "infection": enc["infection"],
                "anatomic_locations": enc["anatomic_locations"],
            },
            "query_en": enc.get("query_content_en", ""),
            "expert_responses": expert_responses,
            "infection": enc["infection"],
        }
        samples.append(sample)

    wound_types_sorted = sorted({s["wound_type"] for s in samples})

    manifest: dict[str, Any] = {
        "metadata": {
            "dataset": "WoundcareVQA",
            "subset": "mini",
            "description": "1 sample per wound type, maximizing clinical diversity",
            "total_samples": len(samples),
            "wound_types": wound_types_sorted,
            "selection_criteria": (
                "single-image preferred, then infection > not, "
                "deeper stage > shallow, encounter_id tie-break"
            ),
        },
        "samples": samples,
    }

    manifest_path = output_dir / "manifest.yaml"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        yaml.dump(
            manifest, fh,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )

    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a minimal WoundcareVQA subset (1 per wound type).",
    )
    parser.add_argument(
        "--dataset",
        default="data/datasets/WoundcareVQA",
        help="Path to the WoundcareVQA dataset directory (default: data/datasets/WoundcareVQA)",
    )
    parser.add_argument(
        "--output",
        default="data/datasets/WoundcareVQA/subset_mini",
        help="Output directory for the subset (default: data/datasets/WoundcareVQA/subset_mini)",
    )
    args = parser.parse_args()

    manifest = generate_subset(args.dataset, args.output)
    total = manifest["metadata"]["total_samples"]
    types = manifest["metadata"]["wound_types"]
    print(f"Generated subset with {total} samples covering {len(types)} wound types:")
    for sample in manifest["samples"]:
        eid = sample["encounter_id"]
        wt = sample["wound_type"]
        img = sample["image"]
        print(f"  {eid}  {wt:12s}  {img}")
    out_path = Path(args.output).resolve() / "manifest.yaml"
    print(f"Manifest written to: {out_path}")


if __name__ == "__main__":
    main()
