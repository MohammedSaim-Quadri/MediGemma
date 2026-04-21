"""Tests for scripts/generate_subset.py"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import yaml

# Ensure project root is on sys.path so we can import the script.
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.generate_subset import (
    _load_all_encounters,
    _pick_best_per_type,
    generate_subset,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_encounter(
    encounter_id: str,
    wound_type: str,
    image_ids: list[str],
    infection: str = "not_infected",
    wound_thickness: str = "not_applicable",
    tissue_color: str = "red_moist",
    drainage_amount: str = "scant",
    drainage_type: str = "serous",
    anatomic_locations: list[str] | None = None,
    query_content_en: str = "Test query",
    responses: list[dict] | None = None,
) -> dict:
    if anatomic_locations is None:
        anatomic_locations = ["forearm"]
    if responses is None:
        responses = [
            {"author_id": "ann1", "content_en": "Expert response.", "content_zh": ""}
        ]
    return {
        "encounter_id": encounter_id,
        "image_ids": image_ids,
        "wound_type": wound_type,
        "wound_thickness": wound_thickness,
        "tissue_color": tissue_color,
        "drainage_amount": drainage_amount,
        "drainage_type": drainage_type,
        "infection": infection,
        "anatomic_locations": anatomic_locations,
        "query_title_en": "title",
        "query_title_zh": "title_zh",
        "query_content_en": query_content_en,
        "query_content_zh": "query_zh",
        "responses": responses,
    }


def _create_dummy_jpeg(path: Path) -> None:
    """Write a minimal (but valid enough) JPEG file."""
    # Minimal JFIF: SOI marker + padding
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\xff\xd8\xff\xe0" + b"\x00\x10JFIF\x00" + b"\x00" * 100 + b"\xff\xd9"
    )


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_dataset(tmp_path: Path) -> Path:
    """Create a minimal mock WoundcareVQA dataset.

    4 encounters in train.json:
      ENC0001 - pressure,  single-image, not_infected, stage_II
      ENC0002 - traumatic, single-image, infected,     unstageable
      ENC0003 - surgical,  multi-image,  unclear,       stage_I
      ENC0004 - surgical,  single-image, infected,      stage_II
    """
    dataset_dir = tmp_path / "WoundcareVQA"
    dataset_dir.mkdir()

    enc1 = _make_encounter(
        "ENC0001", "pressure",
        image_ids=["IMG_ENC0001_0001.jpg"],
        infection="not_infected",
        wound_thickness="stage_II",
    )
    enc2 = _make_encounter(
        "ENC0002", "traumatic",
        image_ids=["IMG_ENC0002_0001.jpg"],
        infection="infected",
        wound_thickness="unstageable",
    )
    enc3 = _make_encounter(
        "ENC0003", "surgical",
        image_ids=["IMG_ENC0003_0001.jpg", "IMG_ENC0003_0002.jpg"],
        infection="unclear",
        wound_thickness="stage_I",
    )
    enc4 = _make_encounter(
        "ENC0004", "surgical",
        image_ids=["IMG_ENC0004_0001.jpg"],
        infection="infected",
        wound_thickness="stage_II",
    )

    train_data = [enc1, enc2, enc3, enc4]
    (dataset_dir / "train.json").write_text(json.dumps(train_data, ensure_ascii=False))
    (dataset_dir / "valid.json").write_text("[]")
    (dataset_dir / "test.json").write_text("[]")

    # Create dummy images
    images_dir = dataset_dir / "images_train"
    for enc in train_data:
        for img_id in enc["image_ids"]:
            _create_dummy_jpeg(images_dir / img_id)

    return dataset_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGenerateSubset:

    def test_generate_subset_one_per_type(self, mock_dataset: Path, tmp_path: Path) -> None:
        """Exactly 3 wound types should be selected: pressure, traumatic, surgical."""
        output_dir = tmp_path / "output"
        manifest = generate_subset(mock_dataset, output_dir)

        wound_types = {s["wound_type"] for s in manifest["samples"]}
        assert wound_types == {"pressure", "traumatic", "surgical"}
        assert manifest["metadata"]["total_samples"] == 3

    def test_prefers_single_image_encounters(self, mock_dataset: Path, tmp_path: Path) -> None:
        """For surgical, ENC0004 (single-image) should be picked over ENC0003 (multi-image)."""
        output_dir = tmp_path / "output"
        manifest = generate_subset(mock_dataset, output_dir)

        surgical = [s for s in manifest["samples"] if s["wound_type"] == "surgical"]
        assert len(surgical) == 1
        assert surgical[0]["encounter_id"] == "ENC0004"

    def test_prefers_infected_over_not_infected(self, mock_dataset: Path, tmp_path: Path) -> None:
        """The surgical pick should have infection=infected."""
        output_dir = tmp_path / "output"
        manifest = generate_subset(mock_dataset, output_dir)

        surgical = [s for s in manifest["samples"] if s["wound_type"] == "surgical"]
        assert len(surgical) == 1
        assert surgical[0]["infection"] == "infected"

    def test_images_copied(self, mock_dataset: Path, tmp_path: Path) -> None:
        """All selected images must exist in the output images directory."""
        output_dir = tmp_path / "output"
        manifest = generate_subset(mock_dataset, output_dir)

        images_dir = output_dir / "images"
        for sample in manifest["samples"]:
            img_path = images_dir / sample["image"]
            assert img_path.exists(), f"Image not found: {img_path}"

    def test_manifest_yaml_written(self, mock_dataset: Path, tmp_path: Path) -> None:
        """manifest.yaml should exist with correct structure and total_samples=3."""
        output_dir = tmp_path / "output"
        generate_subset(mock_dataset, output_dir)

        manifest_path = output_dir / "manifest.yaml"
        assert manifest_path.exists()

        with open(manifest_path) as fh:
            loaded = yaml.safe_load(fh)

        assert "metadata" in loaded
        assert "samples" in loaded
        assert loaded["metadata"]["total_samples"] == 3
        assert loaded["metadata"]["dataset"] == "WoundcareVQA"
        assert loaded["metadata"]["subset"] == "mini"
        assert sorted(loaded["metadata"]["wound_types"]) == ["pressure", "surgical", "traumatic"]

        # Verify sample structure
        for sample in loaded["samples"]:
            assert "encounter_id" in sample
            assert "split" in sample
            assert "wound_type" in sample
            assert "image" in sample
            assert "image_path" in sample
            assert "ground_truth" in sample
            assert "query_en" in sample
            assert "expert_responses" in sample
            assert "infection" in sample
            gt = sample["ground_truth"]
            assert "wound_type" in gt
            assert "wound_thickness" in gt
            assert "tissue_color" in gt
            assert "drainage_amount" in gt
            assert "drainage_type" in gt
            assert "infection" in gt
            assert "anatomic_locations" in gt

    def test_deterministic_selection(self, mock_dataset: Path, tmp_path: Path) -> None:
        """Running generate_subset twice should produce identical manifests."""
        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"

        manifest1 = generate_subset(mock_dataset, out1)
        manifest2 = generate_subset(mock_dataset, out2)

        for s1, s2 in zip(manifest1["samples"], manifest2["samples"]):
            assert s1["encounter_id"] == s2["encounter_id"]
            assert s1["wound_type"] == s2["wound_type"]
            assert s1["image"] == s2["image"]
            assert s1["image_path"] == s2["image_path"]
            assert s1["ground_truth"] == s2["ground_truth"]
            assert s1["infection"] == s2["infection"]

        assert manifest1["metadata"]["total_samples"] == manifest2["metadata"]["total_samples"]
        assert manifest1["metadata"]["wound_types"] == manifest2["metadata"]["wound_types"]

    def test_image_path_is_relative(self, mock_dataset: Path, tmp_path: Path) -> None:
        """image_path in manifest should be relative (images/xxx.jpg), not absolute."""
        output_dir = tmp_path / "output"
        manifest = generate_subset(mock_dataset, output_dir)

        for sample in manifest["samples"]:
            assert sample["image_path"] == f"images/{sample['image']}"
            assert not Path(sample["image_path"]).is_absolute()

    def test_missing_source_image_raises(self, mock_dataset: Path, tmp_path: Path) -> None:
        """Should raise FileNotFoundError when a source image is missing."""
        # Delete one image to trigger the error
        img_to_remove = mock_dataset / "images_train" / "IMG_ENC0001_0001.jpg"
        img_to_remove.unlink()

        output_dir = tmp_path / "output"
        with pytest.raises(FileNotFoundError, match="Source image not found"):
            generate_subset(mock_dataset, output_dir)
