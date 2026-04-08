from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Literal


class Email(BaseModel):
    id: str
    subject: str
    body: str
    sender: str
    timestamp: str


class Observation(BaseModel):
    task_id: str
    task_name: str
    task_description: str
    instructions: str
    current_email: Email
    step_number: int
    total_steps: int
    cumulative_score: float
    history: List[Dict[str, Any]] = Field(default_factory=list)


class Action(BaseModel):
    # Task 1 – Spam Classification
    spam_label: Optional[Literal["spam", "not_spam"]] = None
    # Task 2 – Email Prioritization
    priority: Optional[Literal["high", "medium", "low"]] = None
    category: Optional[Literal["work", "personal", "finance", "promotion", "support", "other"]] = None
    # Task 3 – Email Response
    response_text: Optional[str] = None
    # Always encouraged
    reasoning: Optional[str] = None


class Reward(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    feedback: str
    breakdown: Dict[str, float] = Field(default_factory=dict)


class EnvState(BaseModel):
    task_id: str
    step_number: int
    total_steps: int
    cumulative_score: float
    done: bool
    actions_taken: List[Dict[str, Any]] = Field(default_factory=list)
