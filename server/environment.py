"""
MailSort environment — core logic.

Implements the OpenEnv Environment interface:
  reset(seed, episode_id, task, **kwargs) -> MailSortObservation
  step(action, **kwargs)                 -> MailSortObservation
  state (property)                       -> MailSortState
  close()                                -> None
"""

from __future__ import annotations

import sys
import os
import uuid
from typing import Any, Dict, List, Optional

# Support both in-repo (OpenEnv monorepo) and standalone pip installs
try:
    from openenv.core.env_server import Environment
except ImportError:
    # Minimal fallback base class — exposes the same interface
    class Environment:
        SUPPORTS_CONCURRENT_SESSIONS: bool = False

        def reset(self, *args, **kwargs):
            raise NotImplementedError

        def step(self, action, *args, **kwargs):
            raise NotImplementedError

        @property
        def state(self):
            raise NotImplementedError

        def close(self) -> None:
            pass

# Ensure project root is importable regardless of working directory
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models import MailSortAction, MailSortObservation, MailSortState
from server.tasks import TASK_REGISTRY, TASK_IDS
from server.rewards import compute_reward, compute_episode_score


class MailSortEnvironment(Environment):
    """
    Enterprise email triage environment.

    Three tasks of increasing difficulty:
      easy   — email_classify  : classify a single email (1 step)
      medium — email_rank      : rank + classify 5 emails (1 step)
      hard   — email_triage    : multi-step classify, route, and draft (3 steps)

    Episodes are fully deterministic given the same task name and seed.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # Default task when none is specified
    DEFAULT_TASK: str = "email_classify"

    def __init__(self) -> None:
        self._task_id: str = self.DEFAULT_TASK
        self._episode_id: str = ""
        self._step: int = 0
        self._max_steps: int = 1
        self._done: bool = False
        self._cumulative_reward: float = 0.0
        self._reward_history: List[float] = []
        self._episode_context: Dict[str, Any] = {}
        self._last_feedback: Optional[str] = None
        self._last_action_error: Optional[str] = None
        self._task_config: Dict[str, Any] = TASK_REGISTRY[self.DEFAULT_TASK]

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task: Optional[str] = None,
        **kwargs: Any,
    ) -> MailSortObservation:
        """
        Initialise a new episode.

        Args:
            seed:       Unused (dataset is fully deterministic); kept for API compat.
            episode_id: Optional UUID to assign; auto-generated if None.
            task:       Task ID — one of: email_classify | email_rank | email_triage.
                        Defaults to email_classify.
            **kwargs:   Passed through for forward-compatibility.

        Returns:
            Initial observation with task description and email(s).
        """
        # Select task
        requested = task or kwargs.get("task_name") or self.DEFAULT_TASK
        if requested not in TASK_REGISTRY:
            requested = self.DEFAULT_TASK

        self._task_id = requested
        self._task_config = TASK_REGISTRY[self._task_id]

        # Reset episode counters
        self._episode_id = episode_id or str(uuid.uuid4())
        self._step = 0
        self._max_steps = self._task_config["max_steps"]
        self._done = False
        self._cumulative_reward = 0.0
        self._reward_history = []
        self._last_feedback = None
        self._last_action_error = None

        # Build episode context
        if self._task_id == "email_classify":
            self._episode_context = {"email_id": self._task_config["default_email_id"]}
        else:
            self._episode_context = {}

        # Emails shown to agent (strip ground_truth before sending)
        # For Task 1, show only the single selected email
        all_emails = self._task_config["emails"]
        if self._task_id == "email_classify":
            selected_id = self._episode_context["email_id"]
            all_emails = [e for e in all_emails if e["id"] == selected_id]
        agent_emails = [self._strip_ground_truth(e) for e in all_emails]

        # Cache episode emails for step() calls
        self._episode_emails_cache = all_emails

        # Step 1 description
        step_desc = self._task_config["step_descriptions"].get(1, "")

        obs = MailSortObservation(
            done=False,
            reward=0.0,
            task_name=self._task_id,
            task_description=self._task_config["description"],
            step_description=step_desc,
            emails=agent_emails,
            step=0,
            max_steps=self._max_steps,
            feedback=None,
            last_action_error=None,
        )
        return obs

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(
        self,
        action: MailSortAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> MailSortObservation:
        """
        Execute one agent action.

        Args:
            action: MailSortAction instance from the agent.
            timeout_s: Ignored (kept for API compatibility).

        Returns:
            Observation with reward, feedback, and done flag.
        """
        if self._done:
            # Episode already finished — return terminal observation
            return self._build_terminal_obs(
                reward=0.0,
                feedback="Episode is already done. Call reset() to start a new episode.",
                error="Episode already done.",
            )

        self._step += 1

        # Parse action to dict
        action_data = self._action_to_dict(action)

        # Compute reward via the reward dispatcher
        reward, feedback, action_error = compute_reward(
            task_id=self._task_id,
            step=self._step,
            action_data=action_data,
            episode_context=self._episode_context,
        )

        self._cumulative_reward += reward
        self._reward_history.append(reward)
        self._last_feedback = feedback
        self._last_action_error = action_error

        # Determine if episode is done
        done = self._step >= self._max_steps
        self._done = done

        # Next step description (or empty if done)
        next_step = self._step + 1
        next_step_desc = (
            self._task_config["step_descriptions"].get(next_step, "")
            if not done
            else ""
        )

        cached = getattr(self, "_episode_emails_cache", self._task_config["emails"])
        agent_emails = [self._strip_ground_truth(e) for e in cached]

        obs = MailSortObservation(
            done=done,
            reward=reward,
            task_name=self._task_id,
            task_description=self._task_config["description"],
            step_description=next_step_desc,
            emails=agent_emails,
            step=self._step,
            max_steps=self._max_steps,
            feedback=feedback,
            last_action_error=action_error,
        )
        return obs

    # ------------------------------------------------------------------
    # state (property)
    # ------------------------------------------------------------------

    @property
    def state(self) -> MailSortState:
        """Return current episode state metadata."""
        return MailSortState(
            task_name=self._task_id,
            episode_id=self._episode_id,
            step=self._step,
            step_count=self._step,   # satisfies base State.step_count
            max_steps=self._max_steps,
            done=self._done,
            cumulative_reward=self._cumulative_reward,
            task_difficulty=self._task_config.get("difficulty", "easy"),
            reward_history=list(self._reward_history),
        )

    # ------------------------------------------------------------------
    # close
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Clean up resources. No-op for this environment."""
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_ground_truth(email: Dict[str, Any]) -> Dict[str, Any]:
        """Return a copy of the email dict without ground_truth fields."""
        return {k: v for k, v in email.items() if k != "ground_truth"}

    @staticmethod
    def _action_to_dict(action: MailSortAction) -> Dict[str, Any]:
        """Convert MailSortAction to a plain dict for the grader."""
        if hasattr(action, "model_dump"):
            d = action.model_dump(exclude_none=True)
        elif hasattr(action, "dict"):
            d = {k: v for k, v in action.dict().items() if v is not None}
        else:
            d = {}
        # Remove the inherited 'metadata' key if empty
        d.pop("metadata", None)
        return d

    def _build_terminal_obs(
        self,
        reward: float,
        feedback: str,
        error: Optional[str] = None,
    ) -> MailSortObservation:
        cached = getattr(self, "_episode_emails_cache", self._task_config["emails"])
        agent_emails = [self._strip_ground_truth(e) for e in cached]
        return MailSortObservation(
            done=True,
            reward=reward,
            task_name=self._task_id,
            task_description=self._task_config["description"],
            step_description="",
            emails=agent_emails,
            step=self._step,
            max_steps=self._max_steps,
            feedback=feedback,
            last_action_error=error,
        )
