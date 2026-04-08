"""
EmailTriageEnv – OpenEnv-compliant environment for email triage tasks.

Interface:
    reset(task_id) -> Observation
    step(action)   -> (Observation | None, Reward, done: bool, info: dict)
    state()        -> EnvState
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
from copy import deepcopy

from .models import Action, EnvState, Observation, Reward, Email
from .tasks import ALL_TASKS
from .graders import (
    grade_spam_classification,
    grade_email_prioritization,
    grade_email_response,
)

GRADERS = {
    "spam_classification": grade_spam_classification,
    "email_prioritization": grade_email_prioritization,
    "email_response": grade_email_response,
}

MAX_STEPS_PENALTY = 0.0  # penalty if agent takes more than max steps (N/A here – fixed number)


class EmailTriageEnv:
    """OpenEnv-compliant email triage environment."""

    def __init__(self) -> None:
        self._task_id: Optional[str] = None
        self._emails: list = []
        self._labels: list = []
        self._step_idx: int = 0
        self._done: bool = False
        self._cumulative_score: float = 0.0
        self._actions_taken: list = []

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, task_id: str = "spam_classification") -> Observation:
        """Reset the environment for a given task and return the first observation."""
        if task_id not in ALL_TASKS:
            raise ValueError(f"Unknown task_id '{task_id}'. Available: {list(ALL_TASKS)}")

        task = ALL_TASKS[task_id]
        self._task_id = task_id
        self._emails = deepcopy(task["emails"])
        # Store labels/criteria under a unified key
        self._labels = task.get("labels") or task.get("criteria") or []
        self._step_idx = 0
        self._done = False
        self._cumulative_score = 0.0
        self._actions_taken = []

        return self._build_observation()

    def step(self, action: Action) -> Tuple[Optional[Observation], Reward, bool, Dict[str, Any]]:
        """
        Process an action and advance the environment.

        Returns:
            (observation, reward, done, info)
        """
        if self._done:
            raise RuntimeError("Environment is done. Call reset() to start a new episode.")
        if self._task_id is None:
            raise RuntimeError("Call reset() before step().")

        # Penalise completely empty actions
        if not any([action.spam_label, action.priority, action.category, action.response_text]):
            reward = Reward(
                score=0.0,
                feedback="Empty action – no fields were populated.",
                breakdown={},
            )
            self._actions_taken.append({"step": self._step_idx, "action": action.model_dump(), "reward": 0.0})
            # Still advance to avoid infinite loops
            self._step_idx += 1
            if self._step_idx >= len(self._emails):
                self._done = True
            obs = None if self._done else self._build_observation()
            return obs, reward, self._done, {"warning": "empty_action"}

        grader = GRADERS[self._task_id]
        label = self._labels[self._step_idx]
        total = len(self._emails)

        reward = grader(action, label, self._step_idx, total)

        self._cumulative_score += reward.score
        self._actions_taken.append({
            "step": self._step_idx,
            "action": action.model_dump(),
            "reward": reward.score,
        })

        self._step_idx += 1
        if self._step_idx >= total:
            self._done = True

        obs = None if self._done else self._build_observation()

        info: Dict[str, Any] = {
            "step": self._step_idx,
            "average_score": self._cumulative_score / self._step_idx,
        }
        if self._done:
            info["final_score"] = self._cumulative_score / total
            info["summary"] = self._build_summary(total)

        return obs, reward, self._done, info

    def state(self) -> EnvState:
        """Return the current environment state."""
        return EnvState(
            task_id=self._task_id or "",
            step_number=self._step_idx,
            total_steps=len(self._emails),
            cumulative_score=self._cumulative_score,
            done=self._done,
            actions_taken=self._actions_taken,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_observation(self) -> Observation:
        task = ALL_TASKS[self._task_id]
        meta = task["meta"]
        email_data = self._emails[self._step_idx]

        return Observation(
            task_id=self._task_id,
            task_name=meta["name"],
            task_description=meta["description"],
            instructions=meta["instructions"],
            current_email=Email(**email_data),
            step_number=self._step_idx,
            total_steps=len(self._emails),
            cumulative_score=self._cumulative_score,
            history=list(self._actions_taken),
        )

    def _build_summary(self, total: int) -> str:
        avg = self._cumulative_score / total if total else 0.0
        return (
            f"Episode complete. {total} emails processed. "
            f"Average score: {avg:.3f}. "
            f"Total score: {self._cumulative_score:.3f}/{total:.1f}."
        )

    # ------------------------------------------------------------------
    # Convenience: list available tasks
    # ------------------------------------------------------------------

    @staticmethod
    def available_tasks() -> Dict[str, str]:
        return {tid: ALL_TASKS[tid]["meta"]["name"] for tid in ALL_TASKS}
