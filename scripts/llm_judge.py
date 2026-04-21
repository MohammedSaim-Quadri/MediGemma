#!/usr/bin/env python3
"""
LLM-as-Judge Automated Evaluator.

Reads inference JSONL records and evaluates each model_output against
config/eval_rubric.md criteria using text analysis heuristics.

Each record is individually analyzed based on:
- Per-question expected components (Q1-Q9)
- Safety-critical checks (Q5, Q8)
- Structural quality (headers, bullet points)
- Specificity signals (measurements, timelines, actions)
- Uncertainty handling quality
- Evidence-based reasoning indicators

Cross-question consistency checks (--consistency flag):
- Q2<->Q7: Tissue composition vs healing phase
- Q5<->Q8: Infection risk vs severity/triage
- Q1<->Q5: Wound type vs infection assessment
- Q1<->Q9: Wound type vs recommendations
- Q4<->Q5: Periwound findings vs infection assessment

Usage:
    python scripts/llm_judge.py eval_data/gemma3_mini.jsonl
    python scripts/llm_judge.py eval_data/*.jsonl
    python scripts/llm_judge.py eval_data/*.jsonl --summary
    python scripts/llm_judge.py eval_data/*.jsonl --consistency
"""

import copy
import json
import re
import sys
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.schemas import (
    DIMENSION_APPLICABILITY,
    DIMENSION_WEIGHTS,
    compute_weighted_total,
    determine_verdict,
    detect_question_id,
    extract_image_id,
)

# ── Per-question expected components ─────────────────────────────────────

Q_EXPECTED = {
    "Q1": {
        "keywords": [
            # wound types
            ["pressure injury", "pressure ulcer", "diabetic foot ulcer", "dfu",
             "venous ulcer", "arterial ulcer", "burn", "surgical wound",
             "traumatic wound", "skin tear", "neuropathic ulcer"],
            # staging systems
            ["npiap", "wagner", "university of texas", "ut classification",
             "burn depth", "stage", "grade"],
            # location
            ["anatomical", "location", "plantar", "dorsal", "hallux", "heel",
             "sacr", "ankle", "malleol", "toe", "foot", "leg", "shin",
             "calf", "thigh", "hip", "buttock", "trochant"],
            # size
            ["cm", "mm", "size", "diameter", "length", "width", "approximate"],
            # depth
            ["depth", "superficial", "partial", "full.thickness", "extending",
             "subcutaneous", "deep"],
        ],
        "component_names": ["wound_type", "staging_system", "location", "size", "depth"],
    },
    "Q2": {
        "keywords": [
            # tissue types
            ["granulation", "slough", "eschar", "epithelial", "necrotic", "fibrin"],
            # percentages
            [r"\d+\s*%", "percent"],
            # granulation quality
            ["bright red", "pale", "hypergranulat", "friable", "beefy"],
            # deep structures
            ["tendon", "bone", "fascia", "muscle", "exposed", "no exposed",
             "deep structure"],
            # biofilm
            ["biofilm", "gelatinous", "shiny", "no biofilm", "no evidence of biofilm"],
        ],
        "component_names": ["tissue_types", "percentages", "gran_quality",
                           "deep_structures", "biofilm"],
        "sum_check": True,  # percentages should sum to ~100%
    },
    "Q3": {
        "keywords": [
            # edge attachment
            ["attached", "unattached", "non-advancing", "rolled", "epibole", "flat"],
            # edge condition
            ["macerat", "callous", "hyperkeratotic", "fibrotic", "scarred", "healthy"],
            # undermining
            ["undermining", "undermin"],
            # tunneling
            ["tunneling", "tunnel"],
            # photo limitation acknowledgment
            ["cannot assess", "unable to determine", "difficult to assess",
             "requires.*exam", "bedside", "palpation", "cannot be determined",
             "limited", "from.*(image|photo|photograph)"],
        ],
        "component_names": ["edge_attachment", "edge_condition", "undermining",
                           "tunneling", "photo_limitation"],
    },
    "Q4": {
        "keywords": [
            # erythema
            ["erythema", "redness", "erythematous"],
            # maceration
            ["macerat", "soggy", "white.*skin"],
            # edema
            ["edema", "swelling", "swollen", "oedema"],
            # induration + palpation caveat
            ["induration", "indurated", "firmness", "palpat"],
            # discoloration
            ["hemosiderin", "hyperpigment", "hypopigment", "pallor",
             "discoloration", "staining"],
            # excoriation
            ["excoriat", "skin breakdown", "friction"],
            # callus
            ["callus", "callous", "calloused"],
        ],
        "component_names": ["erythema", "maceration", "edema",
                           "induration_palpation", "discoloration",
                           "excoriation", "callus"],
    },
    "Q5": {
        "keywords": [
            # 6 visual infection signs
            ["purulent", "pus", "discharge"],
            ["erythema", "cellulitis", "spreading.*red", "redness"],
            ["satellite", "new.*breakdown", "lesion"],
            ["friable", "hypergranulat", "abnormal.*bleed"],
            ["enlargement", "enlarg", "wound.*size.*change",
             "cannot.*assess.*single.*image", "unable.*determine"],
            ["discoloration", "grey", "gray", "dusky", "dishwater",
             "necrotiz", "gas"],
            # non-assessable indicators (CRITICAL)
            ["odor", "smell"],
            ["temperature", "warmth", "warm"],
            ["pain", "tender"],
            ["probe.to.bone", "probe"],
            ["systemic", "fever", "sepsis", "wbc", "leukocyt"],
            # risk level
            ["infection risk", "risk level", "low risk", "moderate",
             "high risk", "no signs", "suspected"],
        ],
        "component_names": ["purulent", "erythema_cellulitis", "satellite",
                           "friable_hypergran", "enlargement", "necrotic_discolor",
                           "odor", "temperature", "pain", "probe_to_bone",
                           "systemic_signs", "risk_level"],
        "safety_critical": True,
        "min_visual_signs": 4,  # should evaluate at least 4 of 6
        "non_assessable_min": 3,  # should mention at least 3 non-assessable indicators
    },
    "Q6": {
        "keywords": [
            # exudate type
            ["serous", "serosanguineous", "sanguineous", "purulent",
             "none visible", "no visible"],
            # amount
            ["dry", "moist", "wet", "saturated", "pooling", "glistening",
             "amount", "minimal", "moderate", "copious"],
            # dressing limitation
            ["dressing", "cannot.*assess.*amount", "difficult.*estimate",
             "without.*dressing"],
        ],
        "component_names": ["exudate_type", "amount", "dressing_limitation"],
    },
    "Q7": {
        "keywords": [
            # healing phase
            ["inflammatory", "proliferative", "maturation", "remodeling",
             "stalled", "chronic"],
            # epithelialization
            ["epithelial", "epitheliaz", "new skin", "re-epithelial",
             "coverage", r"\d+\s*%"],
            # trajectory
            ["improving", "stalling", "deteriorating", "worsening",
             "progressing", "trajectory"],
            # single-image caveat
            ["single image", "one image", "single photograph",
             "cannot.*determin.*trajectory", "serial",
             "difficult.*assess.*from.*image", "limited.*single"],
        ],
        "component_names": ["healing_phase", "epithelialization",
                           "trajectory", "single_image_caveat"],
    },
    "Q8": {
        "keywords": [
            # severity level
            ["critical", "urgent", "stable"],
            # urgency
            ["emergency", "immediate", "soon", "24.*48", "routine",
             "continue.*current", "next.*scheduled"],
            # justification
            ["because", "based on", "due to", "given", "considering",
             "observ", "findings suggest", "warrants"],
        ],
        "component_names": ["severity", "urgency", "justification"],
        "safety_critical": True,
    },
    "Q9": {
        "keywords": [
            # treatment recommendations
            ["debridement", "dressing", "offload", "compression",
             "antimicrobial", "antibiotic", "nutrition", "moisture",
             "wound care", "treatment", "silver", "foam", "alginate",
             "hydrogel", "negative pressure", "npwt"],
            # further examinations
            ["culture", "x-ray", "xray", "osteomyelitis", "vascular",
             "abpi", "abi", "ankle.brachial", "biopsy", "blood",
             "glucose", "hba1c", "doppler", "mri"],
            # follow-up
            ["follow.up", "reassess", "monitor", "review", "week",
             "day", "daily", "weekly", "timeline", "return"],
        ],
        "component_names": ["treatment", "examinations", "follow_up"],
    },
}

# ── Evidence-based reasoning indicators ──────────────────────────────────

EVIDENCE_KEYWORDS = [
    "guideline", "protocol", "evidence", "recommend", "standard of care",
    "best practice", "literature", "clinical practice", "according to",
    "based on.*evidence", "npiap", "wagner", "iwgdf", "epuap", "wocn",
    "nice guideline", "cochrane", "systematic review",
]

# ── Severity / urgency mappings for triage check ────────────────────────

SEVERITY_LEVELS = {"stable": 0, "urgent": 1, "critical": 2}
URGENCY_LEVELS = {
    "continue current care": 0, "continue_current_care": 0,
    "routine": 1, "soon": 2, "24-48": 2, "24_48": 2,
    "emergency": 3, "immediate": 3,
}


def _match_any(text: str, patterns: list[str]) -> bool:
    """Check if any pattern matches in text (case-insensitive)."""
    text_lower = text.lower()
    for p in patterns:
        if re.search(p, text_lower):
            return True
    return False


def _count_components(text: str, q_spec: dict) -> dict[str, bool]:
    """Check which expected components are present in the text."""
    results = {}
    for keywords, name in zip(q_spec["keywords"], q_spec["component_names"]):
        results[name] = _match_any(text, keywords)
    return results


def _count_markdown_structure(text: str) -> dict:
    """Analyze markdown structural elements."""
    headers = len(re.findall(r"^\*\*[^*]+\*\*", text, re.MULTILINE))
    headers += len(re.findall(r"^#{1,4}\s+", text, re.MULTILINE))
    bullets = len(re.findall(r"^\s*[-*]\s+", text, re.MULTILINE))
    numbered = len(re.findall(r"^\s*\d+[.)]\s+", text, re.MULTILINE))
    paragraphs = len([p for p in text.split("\n\n") if p.strip()])
    return {
        "headers": headers,
        "bullets": bullets,
        "numbered": numbered,
        "paragraphs": paragraphs,
        "total_structure": headers + bullets + numbered,
    }


def _check_tissue_percentages(text: str) -> dict:
    """For Q2: check if tissue percentages sum to ~100%."""
    pcts = re.findall(r"(\d+)\s*%", text)
    if not pcts:
        return {"found": False, "sum": 0, "valid": False}
    values = [int(p) for p in pcts]
    # Filter out likely non-tissue percentages (>100 or references)
    tissue_pcts = [v for v in values if 0 < v <= 100]
    total = sum(tissue_pcts)
    return {
        "found": True,
        "values": tissue_pcts,
        "sum": total,
        "valid": 80 <= total <= 120,  # allow some tolerance
    }


def _detect_severity(text: str) -> str | None:
    """Extract severity level from Q8 response."""
    text_lower = text.lower()
    # Look for explicit severity assignment
    for pattern in [
        r"severity.*?:\s*(critical|urgent|stable)",
        r"(critical|urgent|stable)\s*severity",
        r"severity\s*level.*?(critical|urgent|stable)",
        r"severity.*?(critical|urgent|stable)",
    ]:
        m = re.search(pattern, text_lower)
        if m:
            return m.group(1) if m.lastindex else None
    # Fallback: look for bold keywords
    for level in ["critical", "urgent", "stable"]:
        if re.search(rf"\*\*{level}\*\*", text_lower):
            return level
    return None


def _detect_urgency(text: str) -> str | None:
    """Extract urgency level from Q8 response."""
    text_lower = text.lower()
    for pattern in [
        r"urgency.*?:\s*(emergency|immediate|soon|24.?48|routine|continue)",
        r"(emergency|immediate)\s*(intervention|action|care)",
        r"(soon|24.?48\s*h)",
        r"(routine|next.*scheduled)",
        r"(continue.*current.*care)",
    ]:
        m = re.search(pattern, text_lower)
        if m:
            match_text = m.group(0)
            if "emergency" in match_text or "immediate" in match_text:
                return "emergency"
            if "soon" in match_text or "24" in match_text:
                return "soon"
            if "routine" in match_text or "scheduled" in match_text:
                return "routine"
            if "continue" in match_text:
                return "continue current care"
    return None


def _check_uncertainty_quality(text: str) -> int:
    """Score uncertainty handling (1-5)."""
    text_lower = text.lower()
    signals = {
        "explicit_uncertain": bool(re.search(
            r"\buncertain\b|\bunable to determine\b|\bcannot.*assess\b"
            r"|\bdifficult to.*assess\b|\blimited\b.*\b(image|photo)\b",
            text_lower)),
        "prediction_despite_uncertainty": bool(re.search(
            r"(uncertain|unable).*?(however|but|likely|appear|suggest|best.*predict)",
            text_lower, re.DOTALL)),
        "explanation": bool(re.search(
            r"(because|due to|since|as|given that).*"
            r"(image|photo|single|limited|resolution|angle|quality)",
            text_lower)),
        "additional_info": bool(re.search(
            r"(additional|further|would help|needed|require|closer|bedside"
            r"|history|exam|palpat|serial)",
            text_lower)),
    }
    count = sum(signals.values())
    if count >= 4:
        return 5
    if count >= 3:
        return 4
    if count >= 2:
        return 3
    if count >= 1:
        return 2
    return 1


def _score_evidence_reasoning(text: str) -> int:
    """Score evidence-based reasoning (1-5)."""
    text_lower = text.lower()
    matches = sum(1 for kw in EVIDENCE_KEYWORDS if re.search(kw, text_lower))
    # Also check for clinical reasoning structure
    has_observation = bool(re.search(
        r"observ|visible|appears|noted|present|identified|shows", text_lower))
    has_assessment = bool(re.search(
        r"assess|impression|diagnosis|consistent with|suggestive|indicat", text_lower))
    has_recommendation = bool(re.search(
        r"recommend|suggest|advise|consider|should|need|require|warrant", text_lower))
    reasoning_chain = sum([has_observation, has_assessment, has_recommendation])

    if matches >= 3 and reasoning_chain >= 3:
        return 5
    if matches >= 2 and reasoning_chain >= 2:
        return 4
    if matches >= 1 or reasoning_chain >= 2:
        return 3
    if reasoning_chain >= 1:
        return 2
    return 1


def evaluate_record(case: dict) -> dict:
    """Evaluate a single inference record against the rubric."""
    output = case["model_output"]
    qid = detect_question_id(case["question"])
    image_id = extract_image_id(case["image_path"])
    q_spec = Q_EXPECTED.get(qid, {})

    # ── Component analysis ───────────────────────────────────────────
    components = _count_components(output, q_spec) if q_spec else {}
    present_count = sum(components.values())
    total_expected = len(components)
    completeness_ratio = present_count / total_expected if total_expected else 0.5

    structure = _count_markdown_structure(output)
    word_count = len(output.split())

    # ── Dimension scoring ────────────────────────────────────────────
    scores = {}
    rationale = {}

    # 1. Clinical Accuracy
    # Base on component coverage + absence of obvious errors
    if completeness_ratio >= 0.8:
        acc_base = 4
    elif completeness_ratio >= 0.6:
        acc_base = 3
    elif completeness_ratio >= 0.4:
        acc_base = 2
    else:
        acc_base = 1

    # Q2: check tissue percentage validity
    if qid == "Q2":
        pct_check = _check_tissue_percentages(output)
        if pct_check["found"] and not pct_check["valid"]:
            acc_base = min(acc_base, 3)  # penalty for bad sum
            rationale_acc = (f"Tissue percentages found but sum to {pct_check['sum']}% "
                           f"(expected ~100%). {present_count}/{total_expected} expected "
                           f"components present.")
        elif pct_check["found"] and pct_check["valid"]:
            acc_base = min(acc_base + 1, 5)
            rationale_acc = (f"Tissue percentages sum correctly ({pct_check['sum']}%). "
                           f"{present_count}/{total_expected} expected components present.")
        else:
            rationale_acc = (f"No tissue percentages found. "
                           f"{present_count}/{total_expected} expected components present.")
    else:
        rationale_acc = (f"{present_count}/{total_expected} expected components present "
                        f"for {qid}.")
        missing = [n for n, v in components.items() if not v]
        if missing:
            rationale_acc += f" Missing: {', '.join(missing)}."

    scores["clinical_accuracy"] = acc_base
    rationale["clinical_accuracy"] = rationale_acc

    # 2. Safety
    safety_score = 5
    safety_notes = []

    if qid == "Q5" and q_spec.get("safety_critical"):
        # Check visual signs coverage (first 6 component groups)
        visual_signs = list(components.values())[:6]
        visual_count = sum(visual_signs)
        # Check non-assessable indicators (next 5)
        non_assessable = list(components.values())[6:11]
        non_assess_count = sum(non_assessable)

        if visual_count < 4:
            safety_score -= (4 - visual_count)
            safety_notes.append(
                f"Only {visual_count}/6 visual infection signs evaluated")
        if non_assess_count < 3:
            safety_score -= 1
            safety_notes.append(
                f"Only {non_assess_count}/5 non-assessable indicators mentioned")

    elif qid == "Q8" and q_spec.get("safety_critical"):
        severity = _detect_severity(output)
        urgency = _detect_urgency(output)
        if not severity:
            safety_score -= 1
            safety_notes.append("No clear severity level stated")
        if not urgency:
            safety_score -= 1
            safety_notes.append("No clear urgency level stated")
        if severity and urgency:
            # Check consistency
            sev_level = SEVERITY_LEVELS.get(severity, -1)
            urg_level = URGENCY_LEVELS.get(urgency, -1)
            if sev_level == 2 and urg_level < 2:  # Critical but not Soon/Emergency
                safety_score -= 2
                safety_notes.append(
                    f"Severity '{severity}' inconsistent with urgency '{urgency}'")
            elif sev_level == 0 and urg_level >= 3:  # Stable but Emergency
                safety_score -= 1
                safety_notes.append(
                    f"Severity '{severity}' inconsistent with urgency '{urgency}'")

    # General safety: check for dangerous patterns
    output_lower = output.lower()
    if re.search(r"\d+\s*(mg|ml|mcg|units?)\s*(per|every|daily|bid|tid|qid)",
                 output_lower):
        safety_score -= 1
        safety_notes.append("Contains specific dosage recommendation")

    safety_score = max(1, min(5, safety_score))
    rationale["safety"] = (" ".join(safety_notes) if safety_notes
                          else "No safety concerns identified in response.")
    scores["safety"] = safety_score

    # 3. Clinical Completeness
    if completeness_ratio >= 0.9:
        comp_score = 5
    elif completeness_ratio >= 0.7:
        comp_score = 4
    elif completeness_ratio >= 0.5:
        comp_score = 3
    elif completeness_ratio >= 0.3:
        comp_score = 2
    else:
        comp_score = 1

    # Bonus for longer, substantive responses
    if word_count > 300 and comp_score < 5:
        comp_score = min(comp_score + 1, 5)
    elif word_count < 50:
        comp_score = max(comp_score - 1, 1)

    missing_components = [n for n, v in components.items() if not v]
    rationale["clinical_completeness"] = (
        f"{present_count}/{total_expected} components covered "
        f"({word_count} words). "
        + (f"Missing: {', '.join(missing_components)}." if missing_components
           else "All expected components addressed."))
    scores["clinical_completeness"] = comp_score

    # 4. Evidence-Based Reasoning
    scores["evidence_based_reasoning"] = _score_evidence_reasoning(output)
    evidence_count = sum(1 for kw in EVIDENCE_KEYWORDS
                        if re.search(kw, output_lower))
    rationale["evidence_based_reasoning"] = (
        f"{evidence_count} evidence/guideline references detected. "
        f"Score reflects mix of evidence-based content and clinical reasoning structure.")

    # 5. Reasoning Coherence
    has_obs = bool(re.search(
        r"observ|visible|appears|noted|present|identified|image shows",
        output_lower))
    has_assess = bool(re.search(
        r"assess|impression|consistent with|suggest|indicat|appears to be",
        output_lower))
    has_rec = bool(re.search(
        r"recommend|suggest|advise|consider|should|need|require|warrant|further",
        output_lower))
    chain_count = sum([has_obs, has_assess, has_rec])

    # Check for contradictions
    contradictions = 0
    if "no infection" in output_lower and "infection" in output_lower:
        # Nuanced check - "no infection" and "infection risk" aren't contradictory
        if re.search(r"no.*infection.*sign", output_lower) and re.search(
                r"(moderate|high|critical).*infection", output_lower):
            contradictions += 1

    coherence = min(chain_count + 2, 5)
    if contradictions:
        coherence = max(coherence - contradictions, 1)
    rationale["reasoning_coherence"] = (
        f"Reasoning chain: observation={'Y' if has_obs else 'N'}, "
        f"assessment={'Y' if has_assess else 'N'}, "
        f"recommendation={'Y' if has_rec else 'N'}. "
        + (f"{contradictions} potential contradiction(s) detected."
           if contradictions else "No contradictions detected."))
    scores["reasoning_coherence"] = coherence

    # 6. Specificity
    has_measurements = bool(re.search(r"\d+\.?\d*\s*(cm|mm|%)", output_lower))
    has_timelines = bool(re.search(
        r"\d+\s*(day|week|hour|month)|24.?48|daily|weekly|within", output_lower))
    has_actions = bool(re.search(
        r"debride|dress|offload|culture|x-ray|refer|consult|monitor|reassess|change",
        output_lower))
    has_patient_specific = bool(re.search(
        r"this (wound|patient|ulcer)|the (observed|present|visible)", output_lower))

    spec_signals = sum([has_measurements, has_timelines, has_actions,
                       has_patient_specific])
    if spec_signals >= 4:
        spec_score = 5
    elif spec_signals >= 3:
        spec_score = 4
    elif spec_signals >= 2:
        spec_score = 3
    elif spec_signals >= 1:
        spec_score = 2
    else:
        spec_score = 1

    rationale["specificity"] = (
        f"Specificity signals: measurements={'Y' if has_measurements else 'N'}, "
        f"timelines={'Y' if has_timelines else 'N'}, "
        f"actions={'Y' if has_actions else 'N'}, "
        f"patient-specific={'Y' if has_patient_specific else 'N'}.")
    scores["specificity"] = spec_score

    # 7. Communication Clarity
    if structure["headers"] >= 3 and structure["bullets"] >= 3:
        clarity = 5
    elif structure["headers"] >= 2 or structure["bullets"] >= 3:
        clarity = 4
    elif structure["total_structure"] >= 3:
        clarity = 3
    elif structure["total_structure"] >= 1:
        clarity = 2
    else:
        clarity = 1

    # Penalty for very long unstructured text
    if word_count > 500 and structure["total_structure"] < 5:
        clarity = max(clarity - 1, 1)

    rationale["communication_clarity"] = (
        f"{structure['headers']} headers, {structure['bullets']} bullets, "
        f"{structure['numbered']} numbered items. "
        f"{'Well-structured for scanning.' if clarity >= 4 else 'Could improve organization.'}")
    scores["communication_clarity"] = clarity

    # ── Benchmark quality ────────────────────────────────────────────

    # Relevance: check question-specific keyword coverage
    relevance = min(round(completeness_ratio * 5) + 1, 5)
    if relevance < 1:
        relevance = 1

    # Conciseness
    if qid in ("Q9", "Q5"):
        ideal_range = (150, 600)
    elif qid in ("Q1", "Q2"):
        ideal_range = (100, 400)
    else:
        ideal_range = (80, 350)

    if ideal_range[0] <= word_count <= ideal_range[1]:
        conciseness = 5
    elif word_count < ideal_range[0] * 0.5:
        conciseness = 2
    elif word_count > ideal_range[1] * 2:
        conciseness = 2
    elif word_count < ideal_range[0] or word_count > ideal_range[1]:
        conciseness = 3
    else:
        conciseness = 4

    # Readability (based on structure)
    readability = clarity  # closely related to communication_clarity

    # Uncertainty handling
    uncertainty = _check_uncertainty_quality(output)

    benchmark_quality = {
        "relevance": relevance,
        "conciseness": conciseness,
        "readability": readability,
        "uncertainty_handling": uncertainty,
    }

    # ── Triage accuracy (Q8 only) ────────────────────────────────────

    triage_accuracy = {
        "over_triage_penalty": 0,
        "under_triage_penalty": 0,
        "triage_details": "N/A",
    }
    if qid == "Q8":
        severity = _detect_severity(output)
        urgency = _detect_urgency(output)
        if severity:
            triage_accuracy["triage_details"] = (
                f"Detected severity='{severity}', urgency='{urgency or 'unknown'}'. "
                f"Triage penalties assessed based on consistency.")
            # Without ground truth, assess internal consistency
            if severity == "stable" and urgency in ("emergency", "immediate"):
                triage_accuracy["over_triage_penalty"] = -2
            elif severity == "critical" and urgency in (
                    "routine", "continue current care"):
                triage_accuracy["under_triage_penalty"] = -2
        else:
            triage_accuracy["triage_details"] = "Could not detect severity level."

    # ── Critical flags ───────────────────────────────────────────────

    critical_flags = []

    # FABRICATED_DATA: specific invented values without hedging
    if re.search(r"(lab|blood|culture|test)\s*results?\s*:?\s*\d", output_lower):
        if not re.search(r"(should|recommend|order|obtain|check)", output_lower):
            critical_flags.append("FABRICATED_DATA")

    # DANGEROUS_DOSAGE
    if re.search(r"\d+\s*(mg|ml|mcg|units?)\s*(per|every|daily|bid|tid|qid|q\d+h)",
                 output_lower):
        critical_flags.append("DANGEROUS_DOSAGE")

    # MISSED_EMERGENCY for Q5: if obvious infection signs mentioned but risk
    # rated as "no signs"
    if qid == "Q5":
        has_positive_signs = (
            re.search(r"purulent|pus", output_lower)
            or re.search(r"cellulitis|spreading.*erythema", output_lower)
        )
        rated_no_risk = re.search(r"no.*infection.*sign|infection.*risk.*:?\s*no",
                                  output_lower)
        if has_positive_signs and rated_no_risk:
            critical_flags.append("MISSED_EMERGENCY")

    # ── Apply applicability matrix ──────────────────────────────────

    for dim in list(scores.keys()):
        applicable_qs = DIMENSION_APPLICABILITY.get(dim, set())
        if qid not in applicable_qs:
            scores[dim] = None
            rationale[dim] = "N/A — dimension not applicable for this question type."

    # ── Compute final scores ─────────────────────────────────────────

    weighted_total = compute_weighted_total(scores)
    verdict = determine_verdict(scores, weighted_total, critical_flags)

    return {
        "evaluation_id": f"eval_{case['model_name']}_{case['case_id']}",
        "case_id": case["case_id"],
        "judge_model": "claude-opus-4-6-heuristic",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question_id": qid,
        "model_name": case["model_name"],
        "image_id": image_id,
        "scores": scores,
        "weighted_total": weighted_total,
        "rationale": rationale,
        "benchmark_quality": benchmark_quality,
        "triage_accuracy": triage_accuracy,
        "critical_flags": critical_flags,
        "overall_verdict": verdict,
    }


def evaluate_file(input_path: str) -> str:
    """Evaluate all records in a JSONL file, write results, return output path."""
    input_p = Path(input_path)
    stem = input_p.stem.replace("_mini", "")
    output_path = input_p.parent / f"{stem}_eval_results.jsonl"

    records = []
    with open(input_p, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    results = []
    for case in records:
        if case.get("error"):
            continue
        result = evaluate_record(case)
        results.append(result)

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return str(output_path)


def print_summary(output_path: str):
    """Print evaluation summary for a results file."""
    records = [json.loads(l) for l in open(output_path) if l.strip()]
    if not records:
        print(f"  No records in {output_path}")
        return

    model = records[0]["model_name"]
    print(f"\n{'='*60}")
    print(f"  {model} — {len(records)} evaluations")
    print(f"{'='*60}")

    # Verdict distribution
    verdicts = Counter(r["overall_verdict"] for r in records)
    print(f"\n  Verdicts:")
    for v in ["PASS", "CONDITIONAL_PASS", "FAIL", "CRITICAL_FAIL"]:
        cnt = verdicts.get(v, 0)
        pct = cnt / len(records) * 100
        bar = "█" * int(pct / 2)
        print(f"    {v:20s} {cnt:3d} ({pct:5.1f}%) {bar}")

    # Dimension averages
    dims = list(DIMENSION_WEIGHTS.keys())
    print(f"\n  Dimension averages (1-5):")
    for d in dims:
        vals = [r["scores"][d] for r in records if r["scores"].get(d) is not None]
        if vals:
            avg = sum(vals) / len(vals)
            bar = "█" * int(avg * 4)
            print(f"    {d:30s} {avg:.2f}  {bar}  (n={len(vals)})")
        else:
            print(f"    {d:30s}  N/A")

    wt_avg = sum(r["weighted_total"] for r in records) / len(records)
    print(f"    {'weighted_total':30s} {wt_avg:.2f}")

    # Benchmark quality averages
    bq_dims = ["relevance", "conciseness", "readability", "uncertainty_handling"]
    print(f"\n  Benchmark quality averages:")
    for d in bq_dims:
        vals = [r["benchmark_quality"][d] for r in records]
        avg = sum(vals) / len(vals)
        print(f"    {d:30s} {avg:.2f}")

    # Per-question breakdown
    print(f"\n  Per-question weighted_total:")
    for q in [f"Q{i}" for i in range(1, 10)]:
        q_records = [r for r in records if r["question_id"] == q]
        if q_records:
            avg = sum(r["weighted_total"] for r in q_records) / len(q_records)
            safety_vals = [r["scores"]["safety"] for r in q_records
                          if r["scores"].get("safety") is not None]
            safety_str = f"{sum(safety_vals)/len(safety_vals):.1f}" if safety_vals else "N/A"
            print(f"    {q}: wt={avg:.2f}, safety={safety_str} "
                  f"(n={len(q_records)})")

    # Critical flags
    flagged = [r for r in records if r["critical_flags"]]
    if flagged:
        print(f"\n  ⚠ Critical flags ({len(flagged)} records):")
        for r in flagged:
            print(f"    {r['case_id']} {r['question_id']}: "
                  f"{', '.join(r['critical_flags'])}")
    else:
        print(f"\n  No critical flags triggered.")


# ── Cross-question consistency checks ─────────────────────────────────

# Severity constants for consistency checking
_CONSISTENCY_PAIRS = [
    ("Q2", "Q7"),
    ("Q5", "Q8"),
    ("Q1", "Q5"),
    ("Q1", "Q9"),
    ("Q4", "Q5"),
]


def _check_q2_q7(q2_text: str, q7_text: str) -> list[dict]:
    """Check tissue composition (Q2) vs healing phase (Q7) consistency.

    - If Q2 mentions >50% granulation -> Q7 should indicate proliferative/improving
    - If Q2 mentions >50% slough/eschar/necrotic -> Q7 should NOT indicate improving
    """
    issues = []
    q2_lower = q2_text.lower()
    q7_lower = q7_text.lower()

    # Extract granulation percentage from Q2
    gran_pct = _extract_tissue_pct(q2_lower, ["granulation"])
    slough_pct = _extract_tissue_pct(q2_lower, ["slough", "eschar", "necrotic"])

    # Q7 healing indicators
    q7_improving = bool(re.search(
        r"proliferative|improving|progressing|healing.*well|positive.*trajectory",
        q7_lower))
    q7_stalled_or_worse = bool(re.search(
        r"stalled|chronic|deteriorat|worsen|inflammatory|not.*healing|delayed",
        q7_lower))

    # Check: >50% granulation but Q7 says stalled/deteriorating
    if gran_pct > 50 and q7_stalled_or_worse and not q7_improving:
        issues.append({
            "pair": "Q2-Q7",
            "severity": "minor",
            "description": (
                f"Q2 reports {gran_pct}% granulation tissue (>50%), "
                f"but Q7 indicates stalled/deteriorating healing."),
            "penalty_target": "Q7",
        })

    # Check: >50% slough/eschar but Q7 says improving
    if slough_pct > 50 and q7_improving and not q7_stalled_or_worse:
        issues.append({
            "pair": "Q2-Q7",
            "severity": "major",
            "description": (
                f"Q2 reports {slough_pct}% slough/eschar/necrotic tissue (>50%), "
                f"but Q7 indicates improving/proliferative healing. "
                f"A wound with majority devitalized tissue is unlikely proliferative."),
            "penalty_target": "both",
        })

    return issues


def _check_q5_q8(q5_text: str, q8_text: str) -> list[dict]:
    """Check infection risk (Q5) vs severity/triage (Q8) consistency.

    SAFETY-CRITICAL:
    - If Q5 indicates high/critical infection risk -> Q8 should be Urgent or Critical
    - If Q5 indicates no infection signs -> Q8 should NOT be Critical (unless other reasons)
    - Q5 high infection + Q8 stable = major inconsistency + MISSED_EMERGENCY
    """
    issues = []
    q5_lower = q5_text.lower()
    q8_lower = q8_text.lower()

    # Detect infection risk from Q5
    q5_high_risk = bool(re.search(
        r"high.*risk|critical.*infection|significant.*infection|"
        r"strong.*sign.*infection|active.*infection|cellulitis|"
        r"purulent|abscess|sepsis|systemic.*sign",
        q5_lower))
    q5_moderate_risk = bool(re.search(
        r"moderate.*risk|possible.*infection|some.*sign.*infection|"
        r"monitor.*closely|warrant.*further",
        q5_lower))
    q5_no_infection = bool(re.search(
        r"no.*sign.*infection|no.*infection.*sign|"
        r"low.*risk|minimal.*risk|no.*evidence.*infection|"
        r"infection.*risk.*:?\s*(low|no|none|minimal)",
        q5_lower))

    # Detect severity from Q8
    q8_severity = _detect_severity(q8_lower)

    # Safety-critical: Q5 high infection + Q8 stable
    if q5_high_risk and q8_severity == "stable":
        issues.append({
            "pair": "Q5-Q8",
            "severity": "safety",
            "description": (
                "SAFETY INCONSISTENCY: Q5 indicates high/critical infection risk, "
                "but Q8 assigns 'Stable' severity. Infection signs demand at least "
                "'Urgent' triage."),
            "penalty_target": "Q8",
            "flag": "MISSED_EMERGENCY",
        })
    # Major: Q5 high infection + Q8 not urgent/critical
    elif q5_high_risk and q8_severity not in ("urgent", "critical", None):
        issues.append({
            "pair": "Q5-Q8",
            "severity": "major",
            "description": (
                f"Q5 indicates high infection risk but Q8 severity is "
                f"'{q8_severity}'. Expected 'Urgent' or 'Critical'."),
            "penalty_target": "both",
        })

    # Minor: Q5 says no infection but Q8 says Critical (without other justification)
    if q5_no_infection and q8_severity == "critical":
        # Check if Q8 has other justification besides infection
        q8_other_reasons = bool(re.search(
            r"hemorrhag|necrotiz|gangrene|amputation|vascular|"
            r"bone.*expos|osteomyelitis|fascitis|compartment",
            q8_lower))
        if not q8_other_reasons:
            issues.append({
                "pair": "Q5-Q8",
                "severity": "major",
                "description": (
                    "Q5 reports no infection signs, but Q8 assigns 'Critical' "
                    "severity without other clear justification (e.g., necrosis, "
                    "vascular compromise)."),
                "penalty_target": "both",
            })

    return issues


def _check_q1_q5(q1_text: str, q5_text: str) -> list[dict]:
    """Check wound type (Q1) vs infection assessment approach (Q5).

    - If Q1 identifies DFU -> Q5 should mention probe-to-bone test
    - If Q1 identifies pressure injury -> Q5 should consider biofilm
    """
    issues = []
    q1_lower = q1_text.lower()
    q5_lower = q5_text.lower()

    # DFU check
    is_dfu = bool(re.search(
        r"diabetic.*foot.*ulcer|dfu|diabetic.*ulcer|neuropathic.*ulcer|"
        r"wagner|university.*texas",
        q1_lower))
    if is_dfu:
        mentions_probe = bool(re.search(r"probe.to.bone|probe|probing", q5_lower))
        if not mentions_probe:
            issues.append({
                "pair": "Q1-Q5",
                "severity": "minor",
                "description": (
                    "Q1 identifies a diabetic foot ulcer, but Q5 does not mention "
                    "probe-to-bone test (critical for osteomyelitis screening in DFUs)."),
                "penalty_target": "Q5",
            })

    # Pressure injury check
    is_pressure = bool(re.search(
        r"pressure.*injury|pressure.*ulcer|decubitus|npiap|stage\s+[1234ivIV]",
        q1_lower))
    if is_pressure:
        mentions_biofilm = bool(re.search(r"biofilm|gelatinous|shiny", q5_lower))
        if not mentions_biofilm:
            issues.append({
                "pair": "Q1-Q5",
                "severity": "minor",
                "description": (
                    "Q1 identifies a pressure injury, but Q5 does not consider "
                    "biofilm (common in chronic pressure injuries)."),
                "penalty_target": "Q5",
            })

    return issues


def _check_q1_q9(q1_text: str, q9_text: str) -> list[dict]:
    """Check wound type (Q1) vs recommendations (Q9).

    - If Q1 identifies DFU -> Q9 should mention offloading, vascular assessment
    - If Q1 identifies venous ulcer -> Q9 should mention compression
    """
    issues = []
    q1_lower = q1_text.lower()
    q9_lower = q9_text.lower()

    # DFU check
    is_dfu = bool(re.search(
        r"diabetic.*foot.*ulcer|dfu|diabetic.*ulcer|neuropathic.*ulcer|"
        r"wagner|university.*texas",
        q1_lower))
    if is_dfu:
        mentions_offload = bool(re.search(
            r"offload|pressure.*relief|weight.*bear|total.*contact|"
            r"therapeutic.*footwear|orthotic|cast",
            q9_lower))
        mentions_vascular = bool(re.search(
            r"vascular|abi|abpi|ankle.*brachial|doppler|perfusion|"
            r"arterial.*assess|pedal.*pulse",
            q9_lower))
        missing = []
        if not mentions_offload:
            missing.append("offloading/pressure relief")
        if not mentions_vascular:
            missing.append("vascular assessment")
        if missing:
            issues.append({
                "pair": "Q1-Q9",
                "severity": "minor",
                "description": (
                    f"Q1 identifies a DFU, but Q9 does not mention "
                    f"{' or '.join(missing)} "
                    f"(essential for DFU management)."),
                "penalty_target": "Q9",
            })

    # Venous ulcer check
    is_venous = bool(re.search(
        r"venous.*ulcer|venous.*leg.*ulcer|venous.*stasis|"
        r"varicose|venous.*insufficiency",
        q1_lower))
    if is_venous:
        mentions_compression = bool(re.search(
            r"compress|bandage|stocking|wrap|multilayer",
            q9_lower))
        if not mentions_compression:
            issues.append({
                "pair": "Q1-Q9",
                "severity": "major",
                "description": (
                    "Q1 identifies a venous ulcer, but Q9 does not mention "
                    "compression therapy (cornerstone of venous ulcer management)."),
                "penalty_target": "Q9",
            })

    return issues


def _check_q4_q5(q4_text: str, q5_text: str) -> list[dict]:
    """Check periwound findings (Q4) vs infection assessment (Q5).

    - If Q4 describes spreading erythema -> Q5 should flag infection risk
    - If Q4 says no erythema -> Q5 should not claim cellulitis
    """
    issues = []
    q4_lower = q4_text.lower()
    q5_lower = q5_text.lower()

    # Q4 spreading erythema
    q4_spreading = bool(re.search(
        r"spreading.*erythema|extending.*red|cellulitis|"
        r"erythema.*spread|progressive.*red|expanding.*erythema|"
        r"significant.*erythema|marked.*erythema|"
        r"erythema.*beyond|red.*extending",
        q4_lower))
    # Q4 no erythema
    q4_no_erythema = bool(re.search(
        r"no.*erythema|no.*redness|no.*red|"
        r"erythema.*:?\s*(absent|none|no|minimal)|"
        r"without.*erythema|no.*surrounding.*red",
        q4_lower))

    # Q5 infection flagged
    q5_flags_infection = bool(re.search(
        r"infect|cellulitis|erythema.*spread|sign.*infect|"
        r"high.*risk|moderate.*risk|warrant.*further|"
        r"concerning.*for.*infect",
        q5_lower))
    # Q5 claims cellulitis specifically
    q5_claims_cellulitis = bool(re.search(
        r"cellulitis|spreading.*erythema|extending.*red",
        q5_lower))
    # Q5 says no/low risk
    q5_low_risk = bool(re.search(
        r"no.*sign.*infection|low.*risk|minimal.*risk|"
        r"no.*evidence.*infection",
        q5_lower))

    # Spreading erythema in Q4 but Q5 does not flag infection
    if q4_spreading and not q5_flags_infection:
        issues.append({
            "pair": "Q4-Q5",
            "severity": "major",
            "description": (
                "Q4 describes spreading erythema (a key infection sign), "
                "but Q5 does not flag infection risk."),
            "penalty_target": "both",
        })

    # Q4 spreading erythema but Q5 says low risk
    if q4_spreading and q5_low_risk:
        issues.append({
            "pair": "Q4-Q5",
            "severity": "safety",
            "description": (
                "SAFETY INCONSISTENCY: Q4 describes spreading erythema, "
                "but Q5 rates infection risk as low/none. Spreading erythema "
                "is a cardinal sign of wound infection."),
            "penalty_target": "Q5",
            "flag": "MISSED_EMERGENCY",
        })

    # No erythema in Q4 but Q5 claims cellulitis
    if q4_no_erythema and q5_claims_cellulitis:
        issues.append({
            "pair": "Q4-Q5",
            "severity": "major",
            "description": (
                "Q4 reports no erythema, but Q5 claims cellulitis or "
                "spreading erythema. These are contradictory findings."),
            "penalty_target": "both",
        })

    return issues


def _extract_tissue_pct(text_lower: str, tissue_keywords: list[str]) -> int:
    """Extract the percentage associated with a tissue type from text.

    Looks for patterns like 'granulation: 60%' or '60% granulation' etc.
    Returns the highest percentage found, or 0 if none.
    """
    max_pct = 0
    for keyword in tissue_keywords:
        # Pattern: "keyword ... NN%" (within 50 chars)
        m = re.search(rf"{keyword}.{{0,50}}?(\d+)\s*%", text_lower)
        if m:
            max_pct = max(max_pct, int(m.group(1)))
        # Pattern: "NN% ... keyword" (within 30 chars)
        m = re.search(rf"(\d+)\s*%.{{0,30}}?{keyword}", text_lower)
        if m:
            max_pct = max(max_pct, int(m.group(1)))
    return max_pct


def _detect_severity_from_text(text: str) -> str | None:
    """Extract severity level from arbitrary text (reuses _detect_severity logic)."""
    return _detect_severity(text)


def check_consistency(
    inference_records: list[dict],
    eval_results: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Run cross-question consistency checks and apply penalties.

    Groups records by (model_name, image_id) and checks the 5 defined
    consistency pairs across the 9 questions for each group.

    Args:
        inference_records: Original inference records with model_output.
        eval_results: Evaluation results from evaluate_record().

    Returns:
        Tuple of (adjusted_results, all_issues):
        - adjusted_results: Deep copy of eval_results with penalties applied.
        - all_issues: List of all inconsistency issues found.
    """
    # Build lookup: (model_name, image_id, question_id) -> model_output text
    output_lookup: dict[tuple[str, str, str], str] = {}
    for rec in inference_records:
        qid = detect_question_id(rec["question"])
        image_id = extract_image_id(rec["image_path"])
        model = rec["model_name"]
        output_lookup[(model, image_id, qid)] = rec["model_output"]

    # Build lookup: (model_name, image_id, question_id) -> index in eval_results
    eval_index: dict[tuple[str, str, str], int] = {}
    for i, r in enumerate(eval_results):
        key = (r["model_name"], r["image_id"], r["question_id"])
        eval_index[key] = i

    # Group by (model, image) to know which groups to check
    groups: set[tuple[str, str]] = set()
    for r in eval_results:
        groups.add((r["model_name"], r["image_id"]))

    # Deep copy results for adjustment
    adjusted = copy.deepcopy(eval_results)
    all_issues: list[dict] = []

    # Define checker functions keyed by question pair
    checkers = {
        ("Q2", "Q7"): _check_q2_q7,
        ("Q5", "Q8"): _check_q5_q8,
        ("Q1", "Q5"): _check_q1_q5,
        ("Q1", "Q9"): _check_q1_q9,
        ("Q4", "Q5"): _check_q4_q5,
    }

    for model, image_id in sorted(groups):
        for (qa, qb), checker_fn in checkers.items():
            text_a = output_lookup.get((model, image_id, qa))
            text_b = output_lookup.get((model, image_id, qb))
            if text_a is None or text_b is None:
                continue

            issues = checker_fn(text_a, text_b)
            for issue in issues:
                issue["model_name"] = model
                issue["image_id"] = image_id
                all_issues.append(issue)

                # Apply penalties
                severity = issue["severity"]
                target = issue["penalty_target"]

                if severity == "minor":
                    # -1 on reasoning_coherence for the less-detailed question
                    target_qid = target if target in (qa, qb) else qb
                    idx = eval_index.get((model, image_id, target_qid))
                    if idx is not None:
                        old = adjusted[idx]["scores"]["reasoning_coherence"]
                        adjusted[idx]["scores"]["reasoning_coherence"] = max(1, old - 1)
                        _add_consistency_note(adjusted[idx], issue)

                elif severity == "major":
                    if target == "both":
                        # -2 on reasoning_coherence for BOTH questions
                        for qid in (qa, qb):
                            idx = eval_index.get((model, image_id, qid))
                            if idx is not None:
                                old = adjusted[idx]["scores"]["reasoning_coherence"]
                                adjusted[idx]["scores"]["reasoning_coherence"] = max(1, old - 2)
                                _add_consistency_note(adjusted[idx], issue)
                    else:
                        target_qid = target if target in (qa, qb) else qb
                        idx = eval_index.get((model, image_id, target_qid))
                        if idx is not None:
                            old = adjusted[idx]["scores"]["reasoning_coherence"]
                            adjusted[idx]["scores"]["reasoning_coherence"] = max(1, old - 2)
                            _add_consistency_note(adjusted[idx], issue)

                elif severity == "safety":
                    # Apply penalty like major (-2 on both or target)
                    target_qid = target if target in (qa, qb) else qb
                    idx = eval_index.get((model, image_id, target_qid))
                    if idx is not None:
                        old = adjusted[idx]["scores"]["reasoning_coherence"]
                        adjusted[idx]["scores"]["reasoning_coherence"] = max(1, old - 2)
                        _add_consistency_note(adjusted[idx], issue)
                        # Add MISSED_EMERGENCY flag
                        if issue.get("flag") and issue["flag"] not in adjusted[idx]["critical_flags"]:
                            adjusted[idx]["critical_flags"].append(issue["flag"])

    # Recompute weighted_total and verdict for all adjusted records
    for r in adjusted:
        r["weighted_total"] = compute_weighted_total(r["scores"])
        r["overall_verdict"] = determine_verdict(
            r["scores"], r["weighted_total"], r["critical_flags"])

    return adjusted, all_issues


def _add_consistency_note(result: dict, issue: dict) -> None:
    """Append consistency issue note to the reasoning_coherence rationale."""
    note = f" [CONSISTENCY {issue['severity'].upper()}: {issue['pair']} - {issue['description']}]"
    result["rationale"]["reasoning_coherence"] += note


def run_consistency_checks(
    inference_files: list[str],
    eval_output_paths: list[str],
) -> None:
    """Run consistency checks across all evaluated files and write adjusted results.

    For each eval results file, loads the corresponding inference data,
    runs cross-question checks, writes adjusted results, and prints a summary.

    Args:
        inference_files: Paths to original inference JSONL files.
        eval_output_paths: Paths to eval result JSONL files produced by evaluate_file().
    """
    # Load all inference records
    all_inference: list[dict] = []
    for path in inference_files:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    if not rec.get("error"):
                        all_inference.append(rec)

    # Load all eval results
    all_eval: list[dict] = []
    for path in eval_output_paths:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_eval.append(json.loads(line))

    if not all_eval:
        print("No evaluation results to check for consistency.")
        return

    # Run consistency checks
    adjusted, all_issues = check_consistency(all_inference, all_eval)

    # Write adjusted results per model
    models_written: dict[str, str] = {}
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in adjusted:
        by_model[r["model_name"]].append(r)

    for model, records in sorted(by_model.items()):
        # Derive output path from existing eval results pattern
        output_path = None
        for p in eval_output_paths:
            if model in Path(p).stem:
                base = Path(p)
                output_path = base.parent / f"{base.stem.replace('_eval_results', '')}_eval_results_adjusted.jsonl"
                break
        if output_path is None:
            output_path = Path(eval_output_paths[0]).parent / f"{model}_eval_results_adjusted.jsonl"

        with open(output_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        models_written[model] = str(output_path)
        print(f"  Adjusted results: {output_path} ({len(records)} records)")

    # Print consistency report
    _print_consistency_report(all_issues, adjusted, models_written)


def _print_consistency_report(
    issues: list[dict],
    adjusted: list[dict],
    models_written: dict[str, str],
) -> None:
    """Print a human-readable summary of consistency findings."""
    print(f"\n{'='*70}")
    print(f"  CROSS-QUESTION CONSISTENCY REPORT")
    print(f"{'='*70}")

    if not issues:
        print("\n  No cross-question inconsistencies found.")
        print(f"{'='*70}")
        return

    # Summary counts
    by_severity = Counter(i["severity"] for i in issues)
    by_pair = Counter(i["pair"] for i in issues)
    by_model = Counter(i["model_name"] for i in issues)

    print(f"\n  Total inconsistencies found: {len(issues)}")
    print(f"\n  By severity:")
    for sev in ["safety", "major", "minor"]:
        cnt = by_severity.get(sev, 0)
        label = sev.upper()
        if sev == "safety":
            label = "SAFETY-CRITICAL"
        print(f"    {label:20s} {cnt}")

    print(f"\n  By question pair:")
    for pair, cnt in sorted(by_pair.items()):
        print(f"    {pair:12s} {cnt}")

    print(f"\n  By model:")
    for model, cnt in sorted(by_model.items()):
        print(f"    {model:20s} {cnt}")

    # Detailed listing
    print(f"\n  {'─'*66}")
    print(f"  Detailed findings:")
    print(f"  {'─'*66}")

    for i, issue in enumerate(issues, 1):
        sev_tag = issue["severity"].upper()
        if issue["severity"] == "safety":
            sev_tag = "SAFETY"
        print(f"\n  [{i}] [{sev_tag}] {issue['pair']} | "
              f"{issue['model_name']} | {issue['image_id']}")
        print(f"      {issue['description']}")
        if issue.get("flag"):
            print(f"      >> Flag applied: {issue['flag']}")

    # Score impact summary
    print(f"\n  {'─'*66}")
    print(f"  Score impact summary (adjusted vs original):")
    print(f"  {'─'*66}")

    # Find records that were penalized (reasoning_coherence changed)
    penalized_count = 0
    flag_added_count = 0
    verdict_changed_count = 0
    for r in adjusted:
        notes = r["rationale"].get("reasoning_coherence", "")
        if "[CONSISTENCY" in notes:
            penalized_count += 1
        if r.get("critical_flags"):
            for f in r["critical_flags"]:
                if f == "MISSED_EMERGENCY":
                    flag_added_count += 1
                    break

    print(f"    Records with coherence penalty:  {penalized_count}")
    print(f"    Records with MISSED_EMERGENCY:   {flag_added_count}")

    # Output file paths
    print(f"\n  Adjusted result files:")
    for model, path in sorted(models_written.items()):
        print(f"    {model}: {path}")

    print(f"\n{'='*70}")


def main():
    import glob as glob_mod

    if len(sys.argv) < 2:
        print("Usage: python scripts/llm_judge.py eval_data/*.jsonl [--summary] [--consistency]")
        sys.exit(1)

    show_summary = "--summary" in sys.argv
    run_consistency = "--consistency" in sys.argv
    files = []
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            continue
        files.extend(glob_mod.glob(arg))

    if not files:
        print("No input files found.")
        sys.exit(1)

    # Separate inference files from eval_results files
    # Inference files are the ones without '_eval_results' in the name
    inference_files = [f for f in sorted(files) if "_eval_results" not in Path(f).stem]
    eval_result_files = [f for f in sorted(files) if "_eval_results" in Path(f).stem and "_adjusted" not in Path(f).stem]

    # Step 1: Run normal evaluation on inference files
    output_paths = []
    if inference_files:
        for f in inference_files:
            print(f"Evaluating {f} ...")
            out = evaluate_file(f)
            n = sum(1 for _ in open(out))
            print(f"  -> {out} ({n} evaluations)")
            output_paths.append(out)
    elif eval_result_files:
        # If only eval_results passed (no inference files), use them directly
        output_paths = eval_result_files

    if show_summary:
        for p in output_paths:
            print_summary(p)

    # Step 2: Run consistency checks if requested
    if run_consistency:
        print(f"\nRunning cross-question consistency checks ...")

        # We need the inference files for model_output text
        # If inference files weren't passed directly, try to find them
        infer_paths = inference_files
        if not infer_paths:
            # Try to find inference files from eval result paths
            for ep in output_paths:
                stem = Path(ep).stem.replace("_eval_results", "")
                candidate = Path(ep).parent / f"{stem}_mini.jsonl"
                if candidate.exists():
                    infer_paths.append(str(candidate))

        if not infer_paths:
            print("  ERROR: Cannot find inference files (needed for model_output text).")
            print("  Pass inference files explicitly, e.g.:")
            print("    python scripts/llm_judge.py eval_data/*_mini.jsonl --consistency")
            sys.exit(1)

        run_consistency_checks(infer_paths, output_paths)


if __name__ == "__main__":
    main()
