"""Programmatic, deterministic graders for each task (score in [0.0, 1.0])."""
from __future__ import annotations
from typing import Any, Dict
from .models import Action, Reward


def grade_spam_classification(action: Action, label: str, step: int, total: int) -> Reward:
    """Grade Task 1: binary spam classification."""
    predicted = action.spam_label
    if predicted is None:
        return Reward(
            score=0.0,
            feedback="No spam_label provided. Set spam_label to 'spam' or 'not_spam'.",
            breakdown={"classification": 0.0},
        )
    correct = predicted == label
    score = 1.0 if correct else 0.0
    feedback = (
        f"Correct! '{predicted}' matches ground truth." if correct
        else f"Incorrect. Predicted '{predicted}' but expected '{label}'."
    )
    # Small penalty for missing reasoning (encourage explanations)
    if not action.reasoning:
        score = max(0.0, score - 0.05)
        feedback += " (Tip: add reasoning for full credit)"
    return Reward(score=round(score, 3), feedback=feedback, breakdown={"classification": float(correct)})


def grade_email_prioritization(action: Action, label: Dict[str, str], step: int, total: int) -> Reward:
    """Grade Task 2: priority + category accuracy."""
    breakdown: Dict[str, float] = {}

    priority_score = 0.0
    if action.priority is None:
        priority_fb = "No priority provided."
    elif action.priority == label["priority"]:
        priority_score = 1.0
        priority_fb = f"Priority '{action.priority}' correct."
    else:
        # Partial credit for adjacent priority levels
        levels = ["low", "medium", "high"]
        pred_idx = levels.index(action.priority)
        true_idx = levels.index(label["priority"])
        diff = abs(pred_idx - true_idx)
        priority_score = max(0.0, 1.0 - 0.5 * diff)
        priority_fb = f"Priority '{action.priority}' (expected '{label['priority']}')."
    breakdown["priority"] = priority_score

    category_score = 0.0
    if action.category is None:
        category_fb = "No category provided."
    elif action.category == label["category"]:
        category_score = 1.0
        category_fb = f"Category '{action.category}' correct."
    else:
        category_fb = f"Category '{action.category}' (expected '{label['category']}')."
    breakdown["category"] = category_score

    combined = 0.5 * priority_score + 0.5 * category_score
    reasoning_bonus = 0.0
    if action.reasoning and len(action.reasoning.strip()) > 10:
        reasoning_bonus = 0.05
    score = min(1.0, combined + reasoning_bonus)
    feedback = f"{priority_fb} {category_fb}"

    return Reward(score=round(score, 3), feedback=feedback, breakdown=breakdown)


def grade_email_response(action: Action, criteria: Dict[str, Any], step: int, total: int) -> Reward:
    """Grade Task 3: multi-criteria response quality check (fully deterministic)."""
    response = (action.response_text or "").strip()
    breakdown: Dict[str, float] = {}

    if not response:
        return Reward(score=0.0, feedback="No response_text provided.", breakdown={})

    lower = response.lower()
    words = response.split()

    # 1. Keyword coverage
    required_kw = criteria.get("required_keywords", [])
    found_kw = [kw for kw in required_kw if kw.lower() in lower]
    kw_score = len(found_kw) / len(required_kw) if required_kw else 1.0
    breakdown["keyword_coverage"] = round(kw_score, 3)

    # 2. Structure (greeting + closing)
    required_elements = criteria.get("required_elements", {})
    structure_scores = []
    for element, options in required_elements.items():
        found = any(opt.lower() in lower for opt in options)
        structure_scores.append(1.0 if found else 0.0)
    structure_score = sum(structure_scores) / len(structure_scores) if structure_scores else 1.0
    breakdown["structure"] = round(structure_score, 3)

    # 3. Length adequacy
    min_words = criteria.get("min_words", 50)
    if len(words) >= min_words:
        length_score = 1.0
    elif len(words) >= min_words * 0.5:
        length_score = 0.5
    else:
        length_score = 0.1
    breakdown["length"] = round(length_score, 3)

    # 4. Tone (no forbidden phrases)
    forbidden = criteria.get("forbidden", [])
    violations = [f for f in forbidden if f.lower() in lower]
    tone_score = max(0.0, 1.0 - 0.25 * len(violations))
    breakdown["tone"] = round(tone_score, 3)

    # Weighted final score
    w_kw = criteria.get("weight_keywords", 0.35)
    w_st = criteria.get("weight_structure", 0.25)
    w_ln = criteria.get("weight_length", 0.20)
    w_tn = criteria.get("weight_tone", 0.20)

    score = (
        w_kw * kw_score +
        w_st * structure_score +
        w_ln * length_score +
        w_tn * tone_score
    )

    missing_kw = [kw for kw in required_kw if kw.lower() not in lower]
    feedback_parts = [f"Score: {score:.2f}."]
    if missing_kw:
        feedback_parts.append(f"Missing keywords: {', '.join(missing_kw)}.")
    if structure_score < 1.0:
        feedback_parts.append("Response missing greeting or closing.")
    if length_score < 1.0:
        feedback_parts.append(f"Response too short ({len(words)} words, need {min_words}+).")
    if violations:
        feedback_parts.append(f"Inappropriate phrases found: {', '.join(violations)}.")

    return Reward(
        score=round(min(1.0, score), 3),
        feedback=" ".join(feedback_parts),
        breakdown=breakdown,
    )
