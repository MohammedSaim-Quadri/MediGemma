# Wound Image Datasets for Benchmarking

This document lists publicly available wound image datasets suitable for evaluating vision models in the Medical Agent benchmark pipeline.

## Recommended Datasets

### 1. AZH Wound Dataset (Recommended for quick start)

- **Size**: 730 images
- **Wound Types**: Diabetic ulcers, pressure ulcers, venous ulcers, surgical wounds (4 classes)
- **Format**: JPG, 320-700 x 240-525 px
- **Labels**: Wound type classification, anatomical location metadata
- **License**: Research use
- **Download**: https://github.com/uwm-bigdata/Multi-modal-wound-classification-using-images-and-locations
- **Paper**: [Integrated image and location analysis for wound classification (Nature Scientific Reports, 2024)](https://www.nature.com/articles/s41598-024-56626-w)
- **Why recommended**: Easy GitHub download, 4-class classification labels align with our protocol categories, reasonable size for full benchmark sweep

### 2. WoundcareVQA (MEDIQA-WV 2025)

- **Size**: 748 images, 477 wound care cases, 768 expert responses
- **Wound Types**: 8 types (surgical, traumatic, pressure ulcer, diabetic, venous, arterial, burn, other)
- **Structured Labels**: 7 categories — anatomic location (41 classes), wound type (8 classes), wound thickness (6 classes), tissue color (6 classes), drainage amount (6 classes), drainage type (5 classes), infection status (3 classes)
- **Format**: Various image formats
- **License**: Research use
- **Download**: https://osf.io/xsj5u/
- **Paper**: [WoundcareVQA (Journal of Biomedical Informatics, 2025)](https://www.sciencedirect.com/science/article/pii/S1532046425001170)
- **Task Overview**: [MEDIQA-WV 2025 Shared Task (ACL Anthology)](https://aclanthology.org/2025.clinicalnlp-1.3/)
- **Why recommended**: Richest structured labels among available datasets. Expert-written responses can serve as reference answers. Note: does NOT cover all 9 benchmark dimensions — see coverage matrix below.

### 3. Medetec Wound Database

- **Size**: ~358 images
- **Wound Types**: Diabetic ulcers, pressure ulcers, arterial/venous leg ulcers, burns, surgical wounds
- **Format**: JPG, 358-560 x 371-560 px
- **License**: Free for research
- **Download**: http://medetec.co.uk/files/medetec-image-databases.html
- **Why suitable**: Diverse wound types, widely used in academic wound AI papers, good for cross-type evaluation

### 4. DFUC — Diabetic Foot Ulcer Challenge

- **Size**: 4,000 images (2,000 train + 2,000 test)
- **Wound Types**: Diabetic foot ulcers only
- **Labels**: Segmentation masks, ischemia and infection annotations (some editions)
- **Format**: Photos
- **License**: Challenge registration required
- **Download**: https://dfu-challenge.github.io/
- **Paper**: [DFUC Benchmark (Medical Image Analysis, 2024)](https://www.sciencedirect.com/science/article/pii/S1361841524000781)
- **Why suitable**: Large dataset for diabetic foot focus, segmentation ground truth available, recurring MICCAI challenge (4th edition in 2024)

### 5. WoundNet (via PWAT Publication)

- **Size**: 1,639 images
- **Wound Types**: Diabetic ulcers, pressure ulcers, vascular ulcers, surgical wounds
- **Labels**: 8 PWAT (Photographic Wound Assessment Tool) sub-scores per image (each scored 0-4)
- **License**: Available via publication request
- **Paper**: [Automated Prediction of PWAT in Chronic Wound Images (Journal of Medical Systems, 2023)](https://link.springer.com/article/10.1007/s10916-023-02029-9)
- **Related**: [Comprehensive Assessment with Patch-Based CNN (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8442961/)
- **Why suitable**: PWAT labels provide ground truth for wound bed, edges, necrotic tissue, and periulcer skin assessment — directly maps to our Q2-Q4 questions

### 6. Pressure Injury Image Dataset (PIID)

- **Size**: 1,091 images
- **Wound Types**: Pressure injuries only, Stages I-IV
- **Labels**: Stage annotations by trained experts
- **Format**: Smartphone-captured photos
- **License**: Research use
- **Paper**: See [ResearchGate - PIID dataset](https://www.researchgate.net/figure/Example-images-from-the-PIID-dataset_fig2_360382131)
- **Why suitable**: Stage labels for pressure injury classification evaluation, smartphone photos reflect real-world capture conditions

### 7. Chronic Wound Multimodal Database

- **Size**: 188 image sets (photos + thermal images + 3D meshes)
- **Wound Types**: Chronic wounds (mixed)
- **Labels**: Manual wound outlines by experts
- **Format**: Multi-modal (RGB + thermal + 3D)
- **License**: Public research access
- **Download**: https://chronicwounddatabase.eu/
- **Why suitable**: Multi-modal data, expert segmentation masks, smaller but high-quality dataset

---

## Dataset Selection Guide

| Use Case | Recommended Dataset | Reason |
|----------|-------------------|--------|
| **Quick benchmark (all wound types)** | AZH (730 images) | Easy download, 4-class labels, manageable size |
| **Comprehensive evaluation with ground truth** | WoundcareVQA (748 images) | 7 structured label categories, expert responses |
| **Tissue-level assessment validation** | WoundNet (1,639 images) | PWAT scores for wound bed, edges, skin |
| **Pressure injury staging** | PIID (1,091 images) | Stage I-IV labels |
| **Diabetic foot focus** | DFUC (4,000 images) | Largest single-type dataset |
| **Initial prototype testing** | Medetec (358 images) | Small, diverse, free |

## Download Priority

For the Medical Agent benchmark pipeline, we recommend downloading in this order:

1. **AZH** — immediate start, GitHub clone
2. **WoundcareVQA** — OSF download, richest structured labels
3. **Medetec** — supplement for wound types not well represented in AZH

These 3 datasets combined (~1,800 images) cover major wound types. See the coverage matrix below for which questions have ground truth vs. requiring LLM-as-Judge evaluation.

## Per-Question Ground Truth Coverage Matrix

Shows which datasets provide ground truth labels for each benchmark question. Questions without ground truth must be evaluated via LLM-as-Judge.

| Question | AZH | WoundcareVQA | Medetec | WoundNet | PIID | DFUC | Evaluation Method |
|----------|-----|--------------|---------|----------|------|------|-------------------|
| Q1: Wound Type | wound type (4 classes) | wound type (8 classes) | wound type | wound type (4 classes) | - | - | **Ground truth** |
| Q1: Staging | - | thickness (6 classes) | - | - | stage I-IV | - | **Partial GT** (pressure only via PIID) |
| Q1: Location | - | location (41 classes) | - | - | - | - | **GT via WoundcareVQA only** |
| Q2: Tissue Types | - | tissue color (6 classes) | - | PWAT necrotic type/amount + granulation type/amount | - | - | **Partial GT** (WoundNet PWAT scores) |
| Q3: Wound Edges | - | - | - | PWAT edges score | - | - | **Partial GT** (WoundNet only) |
| Q4: Periwound Skin | - | - | - | PWAT periulcer skin score | - | - | **Partial GT** (WoundNet only) |
| Q5: Infection | - | infection status (3 classes) | - | - | - | infection annotations (some editions) | **Partial GT** |
| Q6: Exudate | - | drainage type (5) + amount (6) | - | - | - | - | **GT via WoundcareVQA only** |
| Q7: Healing Status | - | - | - | - | - | - | **LLM-as-Judge only** |
| Q8: Severity/Triage | - | - | - | - | - | - | **LLM-as-Judge only** |
| Q9: Recommendations | - | expert responses (ref) | - | - | - | - | **LLM-as-Judge** (expert responses as reference) |

**Key**: GT = Ground Truth available for automated comparison. Partial GT = labels exist but may not fully cover the question's expected fields. LLM-as-Judge = no ground truth labels; must use evaluator model (Claude/Codex) with eval_rubric.md.
