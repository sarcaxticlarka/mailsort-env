"""
Task registry and deterministic graders for the MailSort environment.

All grading functions are fully deterministic — no LLM, no randomness.
Every grader returns a float in [0.0, 1.0].
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from server.email_data import (
    TASK1_EMAILS,
    TASK1_BY_ID,
    TASK2_EMAILS,
    TASK2_BY_ID,
    TASK2_TRUE_RANKING,
    TASK2_PRESENTATION_ORDER,
    TASK3_EMAILS,
    TASK3_BY_ID,
    TASK3_RESPONSE_TARGET_ID,
    VALID_CATEGORIES,
    VALID_PRIORITIES,
    VALID_DEPARTMENTS,
)

# ---------------------------------------------------------------------------
# Partial-credit tables
# ---------------------------------------------------------------------------

# Category adjacency: pred → {true → partial_score}
# Reflects domain knowledge: security ≈ urgent, spam ≈ routine, etc.
CATEGORY_ADJACENCY: Dict[str, Dict[str, float]] = {
    "urgent":   {"security": 0.4, "billing": 0.2},
    "security": {"urgent": 0.4},
    "billing":  {"urgent": 0.2, "finance": 0.1},
    "spam":     {"routine": 0.2},
    "routine":  {"spam": 0.2, "hr": 0.1},
    "hr":       {"routine": 0.1},
}

# Priority adjacency pairs that receive 0.5 partial credit
PRIORITY_ADJACENT_PAIRS = {
    frozenset({"critical", "high"}),
    frozenset({"high", "medium"}),
    frozenset({"medium", "low"}),
}

# Professional response indicators (deterministic keyword list)
PROFESSIONAL_PHRASES = [
    "thank you", "we will", "i will", "please", "sincerely",
    "best regards", "appreciate", "understand", "acknowledge",
    "we have received", "we are", "our team", "looking into",
    "follow up", "get back", "reach out", "happy to help",
]

# Placeholder / unprofessional indicators (penalize)
BAD_PHRASES = [
    "[your name]", "[name]", "lorem ipsum", "placeholder",
    "insert text", "[insert", "todo", "xxx",
]


# ---------------------------------------------------------------------------
# Core scoring primitives
# ---------------------------------------------------------------------------

def score_category(pred: str, true: str) -> float:
    """Return category score in [0.0, 1.0]."""
    pred = (pred or "").lower().strip()
    true = (true or "").lower().strip()
    if pred == true:
        return 1.0
    return CATEGORY_ADJACENCY.get(true, {}).get(pred, 0.0)


def score_priority(pred: str, true: str) -> float:
    """Return priority score in [0.0, 1.0]."""
    pred = (pred or "").lower().strip()
    true = (true or "").lower().strip()
    if pred == true:
        return 1.0
    if frozenset({pred, true}) in PRIORITY_ADJACENT_PAIRS:
        return 0.5
    return 0.0


def grade_single_classification(
    pred_category: str,
    pred_priority: str,
    true_category: str,
    true_priority: str,
    weight_category: float = 0.6,
    weight_priority: float = 0.4,
) -> float:
    """
    Grade a single email classification.

    Returns weighted score: category (60%) + priority (40%).
    """
    cat = score_category(pred_category, true_category)
    pri = score_priority(pred_priority, true_priority)
    return round(weight_category * cat + weight_priority * pri, 4)


def grade_ranking(pred_ids: List[str], true_ids: List[str]) -> float:
    """
    Kendall Tau correlation between predicted and true ranking,
    normalized from [-1, +1] to [0.0, 1.0].

    Handles missing IDs by placing them at the end.
    """
    n = len(true_ids)
    if n <= 1:
        return 1.0

    # Map each true_id to its predicted position (0-indexed)
    pred_pos: Dict[str, int] = {eid: i for i, eid in enumerate(pred_ids)}

    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            # true order: true_ids[i] should come before true_ids[j]
            pi = pred_pos.get(true_ids[i], n)   # missing → pushed to end
            pj = pred_pos.get(true_ids[j], n)
            if pi < pj:
                concordant += 1
            elif pi > pj:
                discordant += 1
            # ties count as neither

    total = n * (n - 1) // 2
    tau = (concordant - discordant) / total if total > 0 else 0.0
    return round((tau + 1.0) / 2.0, 4)  # normalize to [0, 1]


def grade_response_draft(draft: Optional[str], target_email: Dict[str, Any]) -> float:
    """
    Deterministic quality scoring for an email response draft.
    No LLM involved — uses heuristics only.

    Scoring breakdown (max 1.0):
      0.2 — appropriate length (20–600 chars)
      0.3 — professional tone (keyword match)
      0.3 — relevance (mentions subject keywords)
      0.2 — no placeholder / boilerplate text
    """
    if not draft:
        return 0.0

    score = 0.0
    draft_lower = draft.lower()

    # Length check
    if 20 <= len(draft) <= 600:
        score += 0.2

    # Professional tone
    if any(phrase in draft_lower for phrase in PROFESSIONAL_PHRASES):
        score += 0.3

    # Relevance: at least 1 non-trivial word from the subject
    stop_words = {"the", "a", "an", "of", "in", "to", "for", "and", "or", "is", "are", "be"}
    subject_words = {
        w.lower().strip(".,!?:;")
        for w in target_email.get("subject", "").split()
        if w.lower() not in stop_words and len(w) > 3
    }
    draft_words = set(draft_lower.split())
    if subject_words & draft_words:
        score += 0.3

    # No placeholder text
    if not any(phrase in draft_lower for phrase in BAD_PHRASES):
        score += 0.2

    return round(min(score, 1.0), 4)


def grade_routing(pred_dept: Optional[str], true_dept: Optional[str]) -> float:
    """
    Grade routing department assignment.

    Returns 1.0 for exact match, 0.3 for plausible (but wrong) valid dept,
    0.0 for invalid or very wrong routing.
    """
    if true_dept is None:
        # No routing needed (e.g., spam) — any answer is acceptable
        return 1.0

    pred = (pred_dept or "").lower().strip()
    true = (true_dept or "").lower().strip()

    if pred == true:
        return 1.0
    if pred in VALID_DEPARTMENTS:
        return 0.3  # valid dept but wrong one
    return 0.0


# ---------------------------------------------------------------------------
# Task-level graders
# ---------------------------------------------------------------------------

def grade_task1(
    action_data: Dict[str, Any],
    episode_email_id: str = "e1_01",
) -> Tuple[float, str]:
    """
    Grade Task 1: email_classify (single email, single step).

    action_data must contain:
        classifications: [{"email_id": str, "category": str, "priority": str}]

    Returns (score, feedback_string).
    """
    gt = TASK1_BY_ID[episode_email_id]["ground_truth"]
    classifications = action_data.get("classifications") or []

    if not classifications:
        return 0.0, "No classifications provided. Expected a 'classifications' list."

    # Find the entry for the current email
    pred = None
    for c in classifications:
        if c.get("email_id") == episode_email_id:
            pred = c
            break

    if pred is None:
        # Try first entry regardless of id
        pred = classifications[0]

    pred_cat = (pred.get("category") or "").lower().strip()
    pred_pri = (pred.get("priority") or "").lower().strip()

    score = grade_single_classification(pred_cat, pred_pri, gt["category"], gt["priority"])

    cat_correct = pred_cat == gt["category"]
    pri_correct = pred_pri == gt["priority"]

    feedback_parts = []
    cat_status = "correct" if cat_correct else "incorrect (expected " + gt["category"] + ")"
    pri_status = "correct" if pri_correct else "incorrect (expected " + gt["priority"] + ")"
    feedback_parts.append("Category: " + cat_status)
    feedback_parts.append("Priority: " + pri_status)
    feedback_parts.append(f"Score: {score:.2f}")

    return score, " | ".join(feedback_parts)


def grade_task2(
    action_data: Dict[str, Any],
) -> Tuple[float, str]:
    """
    Grade Task 2: email_rank (5 emails, 1 step — rank + classify all).

    action_data must contain:
        rankings:        [email_id, ...]     # most urgent first
        classifications: [{email_id, category, priority}, ...]

    Returns (score, feedback_string).
    """
    rankings = action_data.get("rankings") or []
    classifications = action_data.get("classifications") or []

    # --- Ranking score ---
    if rankings:
        rank_score = grade_ranking(rankings, TASK2_TRUE_RANKING)
    else:
        rank_score = 0.0

    # --- Classification score (average over all 5 emails) ---
    cls_map: Dict[str, Dict[str, str]] = {}
    for c in classifications:
        eid = c.get("email_id", "")
        cls_map[eid] = c

    per_email_scores: List[float] = []
    for email in TASK2_EMAILS:
        eid = email["id"]
        gt = email["ground_truth"]
        pred = cls_map.get(eid, {})
        pred_cat = (pred.get("category") or "").lower().strip()
        pred_pri = (pred.get("priority") or "").lower().strip()
        s = grade_single_classification(pred_cat, pred_pri, gt["category"], gt["priority"])
        per_email_scores.append(s)

    cls_score = sum(per_email_scores) / len(per_email_scores) if per_email_scores else 0.0
    total_score = round(0.5 * rank_score + 0.5 * cls_score, 4)

    feedback = (
        f"Ranking (Kendall Tau): {rank_score:.2f} | "
        f"Classification avg: {cls_score:.2f} | "
        f"Total: {total_score:.2f}"
    )
    return total_score, feedback


def grade_task3_step1(action_data: Dict[str, Any]) -> Tuple[float, str]:
    """
    Grade Task 3 Step 1: classify 3 emails + detect phishing.

    action_data must contain:
        classifications: [
            {"email_id": str, "category": str, "priority": str, "is_phishing": bool},
            ...
        ]

    Scoring:
        0.6 × avg_classification_score
        0.4 × phishing_accuracy
    """
    classifications = action_data.get("classifications") or []
    cls_map: Dict[str, Dict[str, Any]] = {
        c.get("email_id", ""): c for c in classifications
    }

    per_email_cls: List[float] = []
    phishing_correct = 0
    phishing_total = 0

    for email in TASK3_EMAILS:
        eid = email["id"]
        gt = email["ground_truth"]
        pred = cls_map.get(eid, {})

        pred_cat = (pred.get("category") or "").lower().strip()
        pred_pri = (pred.get("priority") or "").lower().strip()
        s = grade_single_classification(pred_cat, pred_pri, gt["category"], gt["priority"])
        per_email_cls.append(s)

        # Phishing detection
        gt_phish = bool(gt.get("is_phishing", False))
        pred_phish_raw = pred.get("is_phishing")
        if pred_phish_raw is not None:
            if isinstance(pred_phish_raw, str):
                pred_phish = pred_phish_raw.lower() in {"true", "1", "yes"}
            else:
                pred_phish = bool(pred_phish_raw)
            phishing_total += 1
            if pred_phish == gt_phish:
                phishing_correct += 1

    cls_score = sum(per_email_cls) / len(per_email_cls) if per_email_cls else 0.0
    phish_score = phishing_correct / phishing_total if phishing_total > 0 else 0.0

    total = round(0.6 * cls_score + 0.4 * phish_score, 4)
    feedback = (
        f"Classification avg: {cls_score:.2f} | "
        f"Phishing detection: {phish_score:.2f} ({phishing_correct}/{phishing_total}) | "
        f"Step score: {total:.2f}"
    )
    return total, feedback


def grade_task3_step2(action_data: Dict[str, Any]) -> Tuple[float, str]:
    """
    Grade Task 3 Step 2: route each email to the correct department.

    action_data must contain:
        routings: [{"email_id": str, "dept": str}, ...]

    Scoring: average routing accuracy across non-spam emails.
    """
    routings = action_data.get("routings") or []
    routing_map: Dict[str, str] = {r.get("email_id", ""): r.get("dept", "") for r in routings}

    per_email_scores: List[float] = []
    for email in TASK3_EMAILS:
        eid = email["id"]
        gt = email["ground_truth"]
        gt_dept = gt.get("routing_dept")

        if gt_dept is None:
            # spam — no routing required; any response (including omission) is fine
            per_email_scores.append(1.0)
        else:
            pred_dept = routing_map.get(eid, "")
            per_email_scores.append(grade_routing(pred_dept, gt_dept))

    total = round(sum(per_email_scores) / len(per_email_scores), 4) if per_email_scores else 0.0
    feedback = f"Routing accuracy: {total:.2f} (avg across {len(per_email_scores)} emails)"
    return total, feedback


def grade_task3_step3(action_data: Dict[str, Any]) -> Tuple[float, str]:
    """
    Grade Task 3 Step 3: draft acknowledgment response for top-priority email.

    action_data must contain:
        response_draft: str

    Scoring: heuristic quality scoring (see grade_response_draft).
    """
    draft = action_data.get("response_draft") or ""
    target_email = TASK3_BY_ID[TASK3_RESPONSE_TARGET_ID]
    score = grade_response_draft(draft, target_email)
    feedback = f"Response quality: {score:.2f} (length={len(draft)} chars)"
    return score, feedback


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASK_REGISTRY: Dict[str, Dict[str, Any]] = {
    "email_classify": {
        "id": "email_classify",
        "name": "Email Classification",
        "description": (
            "Classify a single enterprise email into one of six categories "
            "(spam, urgent, routine, billing, hr, security) and assign a priority level "
            "(critical, high, medium, low). Clear, unambiguous signals are present."
        ),
        "difficulty": "easy",
        "max_steps": 1,
        "emails": TASK1_EMAILS,
        "default_email_id": "e1_01",
        "step_descriptions": {
            1: (
                "Classify the email. Respond with a JSON action containing:\n"
                "  classifications: [{email_id, category, priority}]\n"
                "Valid categories: spam | urgent | routine | billing | hr | security\n"
                "Valid priorities: critical | high | medium | low"
            )
        },
        "graders": {1: grade_task1},
    },
    "email_rank": {
        "id": "email_rank",
        "name": "Email Priority Ranking",
        "description": (
            "Given five enterprise emails presented in random order, rank them from "
            "most to least urgent and classify each one. Tests the ability to assess "
            "relative urgency across diverse email types."
        ),
        "difficulty": "medium",
        "max_steps": 1,
        "emails": [TASK2_BY_ID[eid] for eid in TASK2_PRESENTATION_ORDER],
        "true_ranking": TASK2_TRUE_RANKING,
        "step_descriptions": {
            1: (
                "Rank all 5 emails by priority (most urgent first) and classify each one.\n"
                "Respond with a JSON action containing:\n"
                "  rankings: [email_id, ...]  (most urgent first)\n"
                "  classifications: [{email_id, category, priority}, ...]"
            )
        },
        "graders": {1: grade_task2},
    },
    "email_triage": {
        "id": "email_triage",
        "name": "Full Email Triage",
        "description": (
            "Perform multi-step triage of three complex emails. One email contains "
            "phishing signals. Step 1: classify and detect threats. "
            "Step 2: route to the correct department. "
            "Step 3: draft a professional acknowledgment for the highest-priority "
            "legitimate email."
        ),
        "difficulty": "hard",
        "max_steps": 3,
        "emails": TASK3_EMAILS,
        "step_descriptions": {
            1: (
                "Classify each email and detect phishing attempts.\n"
                "Respond with a JSON action containing:\n"
                "  classifications: [{email_id, category, priority, is_phishing}, ...]"
            ),
            2: (
                "Route each email to the appropriate department.\n"
                "Valid departments: engineering | support | hr | finance | security | management | sales\n"
                "Respond with a JSON action containing:\n"
                "  routings: [{email_id, dept}, ...]"
            ),
            3: (
                "Draft a professional acknowledgment response (2–5 sentences) for the "
                "highest-priority non-spam email (e3_02).\n"
                "Respond with a JSON action containing:\n"
                "  response_draft: \"Your draft here...\""
            ),
        },
        "graders": {
            1: grade_task3_step1,
            2: grade_task3_step2,
            3: grade_task3_step3,
        },
    },
}

TASK_IDS = list(TASK_REGISTRY.keys())
