"""
Reward computation for the MailSort environment.

Centralises reward shaping logic: dispatches to the correct grader
per task and step, applies action-validity penalties, and returns
a scalar reward in [0.0, 1.0].
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from server.tasks import (
    TASK_REGISTRY,
    grade_task1,
    grade_task2,
    grade_task3_step1,
    grade_task3_step2,
    grade_task3_step3,
)
from server.email_data import VALID_CATEGORIES, VALID_PRIORITIES, VALID_DEPARTMENTS


# ---------------------------------------------------------------------------
# Action validation
# ---------------------------------------------------------------------------

def _validate_action(task_id: str, step: int, action_data: Dict[str, Any]) -> Optional[str]:
    """
    Return an error string if the action is invalid, else None.

    Checks for required fields and valid enum values.
    """
    if not action_data:
        return "Empty action received."

    if task_id == "email_classify":
        classifications = action_data.get("classifications")
        if not classifications or not isinstance(classifications, list):
            return "Missing 'classifications' list."
        for c in classifications:
            cat = (c.get("category") or "").lower().strip()
            pri = (c.get("priority") or "").lower().strip()
            if cat and cat not in VALID_CATEGORIES:
                return (
                    f"Invalid category '{cat}'. "
                    f"Valid: {sorted(VALID_CATEGORIES)}"
                )
            if pri and pri not in VALID_PRIORITIES:
                return (
                    f"Invalid priority '{pri}'. "
                    f"Valid: {sorted(VALID_PRIORITIES)}"
                )

    elif task_id == "email_rank":
        if step == 1:
            rankings = action_data.get("rankings")
            if rankings is not None and not isinstance(rankings, list):
                return "'rankings' must be a list of email IDs."
            classifications = action_data.get("classifications")
            if classifications is not None and not isinstance(classifications, list):
                return "'classifications' must be a list."

    elif task_id == "email_triage":
        if step == 1:
            classifications = action_data.get("classifications")
            if not classifications or not isinstance(classifications, list):
                return "Step 1 requires a 'classifications' list."
        elif step == 2:
            routings = action_data.get("routings")
            if not routings or not isinstance(routings, list):
                return "Step 2 requires a 'routings' list."
            for r in routings:
                dept = (r.get("dept") or "").lower().strip()
                if dept and dept not in VALID_DEPARTMENTS:
                    return (
                        f"Invalid department '{dept}'. "
                        f"Valid: {sorted(VALID_DEPARTMENTS)}"
                    )
        elif step == 3:
            draft = action_data.get("response_draft")
            if draft is not None and not isinstance(draft, str):
                return "'response_draft' must be a string."

    return None  # valid


# ---------------------------------------------------------------------------
# Main reward dispatcher
# ---------------------------------------------------------------------------

INVALID_ACTION_PENALTY: float = 0.05
EMPTY_ACTION_PENALTY: float = 0.10

# Scores must be STRICTLY between 0 and 1 (not 0.0, not 1.0)
SCORE_MIN: float = 0.01
SCORE_MAX: float = 0.99


def compute_reward(
    task_id: str,
    step: int,
    action_data: Dict[str, Any],
    episode_context: Dict[str, Any],
    prior_penalty: float = 0.0,
) -> Tuple[float, str, Optional[str]]:
    """
    Compute the reward for a single step.

    Args:
        task_id:         One of TASK_REGISTRY keys.
        step:            Current step number (1-indexed).
        action_data:     Parsed action dict from the agent.
        episode_context: Dict with episode-specific info (e.g., email_id for Task 1).
        prior_penalty:   Accumulated penalty from prior invalid actions this episode.

    Returns:
        (reward, feedback_str, action_error)
        - reward       : float in [0.0, 1.0]
        - feedback_str : human-readable grader feedback
        - action_error : None if valid, else error message
    """
    # Check for completely empty action
    if not action_data:
        feedback = "Empty action — no fields provided."
        return SCORE_MIN, feedback, "Empty action received."

    # Validate action fields
    action_error = _validate_action(task_id, step, action_data)
    penalty = INVALID_ACTION_PENALTY if action_error else 0.0

    # Grade the action (even if partially invalid — best-effort)
    if task_id == "email_classify":
        email_id = episode_context.get("email_id", "e1_01")
        base_score, feedback = grade_task1(action_data, episode_email_id=email_id)

    elif task_id == "email_rank":
        base_score, feedback = grade_task2(action_data)

    elif task_id == "email_triage":
        if step == 1:
            base_score, feedback = grade_task3_step1(action_data)
        elif step == 2:
            base_score, feedback = grade_task3_step2(action_data)
        elif step == 3:
            base_score, feedback = grade_task3_step3(action_data)
        else:
            base_score, feedback = 0.0, f"Unknown step {step} for email_triage."
    else:
        base_score = 0.0
        feedback = f"Unknown task_id: {task_id}"
        action_error = feedback

    # Apply penalty then clamp strictly to (0.01, 0.99)
    raw = base_score - penalty
    reward = round(max(SCORE_MIN, min(SCORE_MAX, raw)), 4)

    if action_error:
        feedback = f"[Warning: {action_error}] {feedback}"

    return reward, feedback, action_error


# ---------------------------------------------------------------------------
# Episode-level score normalisation
# ---------------------------------------------------------------------------

def compute_episode_score(rewards: list[float], task_id: str) -> float:
    """
    Compute the normalised episode score in [0.0, 1.0].

    For single-step tasks (Task 1, 2): score = the single step reward.
    For multi-step tasks (Task 3):     score = average of step rewards.
    """
    if not rewards:
        return SCORE_MIN
    score = sum(rewards) / len(rewards)
    return round(max(SCORE_MIN, min(SCORE_MAX, score)), 4)
