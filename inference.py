"""
Baseline inference script for EmailTriageEnv.

Uses the OpenAI-compatible API client.
Credentials and endpoints are read from environment variables.

Usage:
    HF_TOKEN=hf_xxx python inference.py
    HF_TOKEN=hf_xxx python inference.py --task spam_classification
    HF_TOKEN=hf_xxx python inference.py --all
"""
from __future__ import annotations
import os
import json
import argparse
import re
from typing import Any, Dict

from openai import OpenAI

from environment import EmailTriageEnv, Action

# ---------------------------------------------------------------------------
# Environment variables — required by OpenEnv submission checklist
# Defaults are set ONLY for API_BASE_URL and MODEL_NAME, NOT for HF_TOKEN
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1/")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional: used when running from a local Docker image
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")


def get_client() -> OpenAI:
    if not HF_TOKEN:
        raise EnvironmentError("HF_TOKEN environment variable is not set.")
    return OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)


# ---------------------------------------------------------------------------
# Structured logging — START / STEP / END format (required by OpenEnv spec)
# ---------------------------------------------------------------------------

def log_start(task_id: str, total_steps: int) -> None:
    print(json.dumps({
        "event": "START",
        "task_id": task_id,
        "model": MODEL_NAME,
        "api_base_url": API_BASE_URL,
        "total_steps": total_steps,
    }), flush=True)


def log_step(task_id: str, step: int, total: int, score: float, feedback: str, action: Dict[str, Any]) -> None:
    print(json.dumps({
        "event": "STEP",
        "task_id": task_id,
        "step": step,
        "total_steps": total,
        "score": round(score, 4),
        "feedback": feedback,
        "action": action,
    }), flush=True)


def log_end(task_id: str, final_score: float, total_steps: int) -> None:
    print(json.dumps({
        "event": "END",
        "task_id": task_id,
        "final_score": round(final_score, 4),
        "total_steps": total_steps,
        "model": MODEL_NAME,
    }), flush=True)


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert email triage assistant.
You will be shown one email at a time and must return a JSON object with your action.

Depending on the task:
- spam_classification: return {"spam_label": "spam" | "not_spam", "reasoning": "..."}
- email_prioritization: return {"priority": "high"|"medium"|"low", "category": "work"|"personal"|"finance"|"promotion"|"support"|"other", "reasoning": "..."}
- email_response: return {"response_text": "<your full email response>", "reasoning": "..."}

IMPORTANT: Return ONLY a valid JSON object. No markdown, no code fences, no extra text.
"""


def build_user_prompt(obs_dict: Dict[str, Any]) -> str:
    email = obs_dict["current_email"]
    return (
        f"Task: {obs_dict['task_name']}\n"
        f"Instructions: {obs_dict['instructions']}\n"
        f"Step: {obs_dict['step_number'] + 1} / {obs_dict['total_steps']}\n\n"
        f"--- EMAIL ---\n"
        f"From: {email['sender']}\n"
        f"Date: {email['timestamp']}\n"
        f"Subject: {email['subject']}\n\n"
        f"{email['body']}\n"
        f"--- END EMAIL ---\n\n"
        f"Provide your action as a JSON object."
    )


def llm_action(client: OpenAI, obs_dict: Dict[str, Any]) -> Action:
    """Call the LLM and parse its response into an Action."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(obs_dict)},
    ]
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=512,
        temperature=0.2,
    )
    raw = response.choices[0].message.content.strip()

    # Strip accidental code fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            print(json.dumps({"event": "WARN", "message": f"Could not parse LLM response: {raw[:200]}"}), flush=True)
            data = {}

    return Action(**data)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, env: EmailTriageEnv, task_id: str) -> float:
    """Run one full episode and return the final average score."""
    obs = env.reset(task_id=task_id)
    obs_dict = obs.model_dump()
    total_steps = obs_dict["total_steps"]

    log_start(task_id, total_steps)

    total_reward = 0.0
    steps = 0

    while True:
        action = llm_action(client, obs_dict)
        next_obs, reward, done, info = env.step(action)

        steps += 1
        total_reward += reward.score

        log_step(
            task_id=task_id,
            step=steps,
            total=total_steps,
            score=reward.score,
            feedback=reward.feedback,
            action=action.model_dump(exclude_none=True),
        )

        if done:
            final_score = info.get("final_score", total_reward / steps)
            log_end(task_id, final_score, total_steps)
            return final_score

        obs_dict = next_obs.model_dump()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EmailTriageEnv baseline inference")
    parser.add_argument(
        "--task",
        default="spam_classification",
        choices=list(EmailTriageEnv.available_tasks().keys()),
        help="Task to evaluate",
    )
    parser.add_argument("--all", action="store_true", help="Run all tasks")
    args = parser.parse_args()

    client = get_client()
    env = EmailTriageEnv()

    tasks = list(EmailTriageEnv.available_tasks().keys()) if args.all else [args.task]

    results: Dict[str, float] = {}
    for task_id in tasks:
        score = run_task(client, env, task_id)
        results[task_id] = score

    # Final summary as structured log
    print(json.dumps({
        "event": "SUMMARY",
        "results": results,
        "average": round(sum(results.values()) / len(results), 4) if results else 0.0,
        "model": MODEL_NAME,
        "api_base_url": API_BASE_URL,
    }), flush=True)


if __name__ == "__main__":
    main()
