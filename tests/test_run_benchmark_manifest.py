"""Tests for --dataset-manifest support in run_benchmark."""
import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def manifest_yaml(tmp_path):
    """Create a mini manifest with 2 samples."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "img1.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
    (images_dir / "img2.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

    manifest = {
        "metadata": {"total_samples": 2},
        "samples": [
            {"encounter_id": "E1", "wound_type": "pressure", "image": "img1.jpg",
             "image_path": str(images_dir / "img1.jpg"), "ground_truth": {"wound_type": "pressure"}},
            {"encounter_id": "E2", "wound_type": "surgical", "image": "img2.jpg",
             "image_path": str(images_dir / "img2.jpg"), "ground_truth": {"wound_type": "surgical"}},
        ],
    }
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.dump(manifest))
    return str(path)


@pytest.fixture
def manifest_with_relative_paths(tmp_path):
    """Manifest with relative image paths."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "img1.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

    manifest = {
        "metadata": {"total_samples": 1},
        "samples": [
            {"encounter_id": "E1", "wound_type": "pressure", "image": "img1.jpg",
             "image_path": "images/img1.jpg", "ground_truth": {"wound_type": "pressure"}},
        ],
    }
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.dump(manifest))
    return str(path)


@pytest.fixture
def manifest_with_missing_image(tmp_path):
    """Manifest referencing a non-existent image."""
    manifest = {
        "metadata": {"total_samples": 1},
        "samples": [
            {"encounter_id": "E1", "wound_type": "pressure", "image": "missing.jpg",
             "image_path": str(tmp_path / "missing.jpg"), "ground_truth": {"wound_type": "pressure"}},
        ],
    }
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.dump(manifest))
    return str(path)


def test_load_images_from_manifest(manifest_yaml):
    from scripts.run_benchmark import load_images_from_manifest
    images = load_images_from_manifest(manifest_yaml)
    assert len(images) == 2
    assert all(img.endswith(".jpg") for img in images)


def test_manifest_images_exist(manifest_yaml):
    from scripts.run_benchmark import load_images_from_manifest
    images = load_images_from_manifest(manifest_yaml)
    for img in images:
        assert os.path.isfile(img), f"Image not found: {img}"


def test_manifest_relative_paths(manifest_with_relative_paths):
    from scripts.run_benchmark import load_images_from_manifest
    images = load_images_from_manifest(manifest_with_relative_paths)
    assert len(images) == 1
    assert os.path.isfile(images[0])


def test_manifest_missing_image_raises(manifest_with_missing_image):
    from scripts.run_benchmark import load_images_from_manifest
    with pytest.raises(FileNotFoundError, match="Image not found"):
        load_images_from_manifest(manifest_with_missing_image)
