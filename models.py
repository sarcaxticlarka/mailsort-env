"""
Typed Pydantic models for the MailSort OpenEnv environment.

These extend the openenv-core base types and are shared between
the server (environment logic) and the client (inference scripts).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    # Fallback for environments without openenv-core installed —
    # defines minimal compatible base classes using plain Pydantic.
    from pydantic import BaseModel, ConfigDict

    class Action(BaseModel):
        model_config = ConfigDict(extra="allow")
        metadata: Dict[str, Any] = {}

    class Observation(BaseModel):
        model_config = ConfigDict(extra="allow")
        done: bool = False
        reward: Optional[float] = None
        metadata: Dict[str, Any] = {}

    class State(BaseModel):
        model_config = ConfigDict(extra="allow")
        episode_id: Optional[str] = None
        step_count: int = 0


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class MailSortAction(Action):
    """
    Action submitted by an agent to the MailSort environment.

    Fields are used selectively based on the current task and step:

    Task 1 — email_classify (step 1):
        classifications: list of {email_id, category, priority}

    Task 2 — email_rank (step 1):
        rankings:        ordered list of email IDs, most urgent first
        classifications: list of {email_id, category, priority}

    Task 3 — email_triage:
        Step 1: classifications with is_phishing flag
        Step 2: routings — list of {email_id, dept}
        Step 3: response_draft — plain text acknowledgment

    All fields are Optional so a single model covers all tasks.
    """

    # Classification entries for one or more emails
    classifications: Optional[List[Dict[str, Any]]] = None
    # Format: [{"email_id": "e1_01", "category": "urgent",
    #            "priority": "critical", "is_phishing": false}]

    # Ordered list of email IDs (Task 2 and Task 3 ranking)
    rankings: Optional[List[str]] = None
    # Format: ["e2_01", "e2_02", ...]  — most urgent first

    # Department routing (Task 3 Step 2)
    routings: Optional[List[Dict[str, str]]] = None
    # Format: [{"email_id": "e3_02", "dept": "support"}]

    # Response draft (Task 3 Step 3)
    response_draft: Optional[str] = None


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class MailSortObservation(Observation):
    """
    Observation returned to the agent after reset() or step().

    Contains the full email context, task instructions, and
    feedback from the previous step (if any).
    """

    # Which task is currently active
    task_name: str = ""

    # Human-readable task description (shown at reset)
    task_description: str = ""

    # Instructions for the CURRENT step
    step_description: str = ""

    # List of email objects the agent must process
    # Each dict has: id, subject, sender, sender_email, body, metadata
    emails: List[Dict[str, Any]] = []

    # Current step number (1-indexed)
    step: int = 0

    # Maximum steps for this episode
    max_steps: int = 1

    # Grader feedback from the previous step (None on reset)
    feedback: Optional[str] = None

    # Error message if the last action was malformed (None if clean)
    last_action_error: Optional[str] = None


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class MailSortState(State):
    """
    Internal episode state — returned by state() / GET /state.

    Tracks metadata about the running episode separate from observations.
    """

    # Active task identifier
    task_name: str = ""

    # Unique episode identifier (UUID)
    episode_id: str = ""

    # Current step within the episode (1-indexed; 0 = not started)
    step: int = 0

    # Maximum steps for this episode
    max_steps: int = 1

    # Whether the episode has terminated
    done: bool = False

    # Sum of rewards collected so far
    cumulative_reward: float = 0.0

    # Task difficulty label
    task_difficulty: str = "easy"

    # Step-by-step reward history
    reward_history: List[float] = []
