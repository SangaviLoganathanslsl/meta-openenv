"""
Baseline inference script for EmailTriageEnv.

Uses the OpenAI-compatible HuggingFace Inference API.
Reads HF_TOKEN from environment variables.

Usage:
    HF_TOKEN=hf_xxx python inference.py
    HF_TOKEN=hf_xxx python inference.py --task spam_classification
    HF_TOKEN=hf_xxx python inference.py --all
"""
from __future__ import annotations
import os
import json
import argparse
from typing import Any, Dict

from openai import OpenAI

from environment import EmailTriageEnv, Action

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN", "")
MODEL = os.environ.get("OPENENV_MODEL", "Qwen/Qwen2.5-7B-Instruct")
HF_BASE_URL = "https://api-inference.huggingface.co/v1/"


def get_client() -> OpenAI:
    if not HF_TOKEN:
        raise EnvironmentError("HF_TOKEN environment variable is not set.")
    return OpenAI(api_key=HF_TOKEN, base_url=HF_BASE_URL)


# ---------------------------------------------------------------------------
# System prompt shared across all tasks
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
    return f"""Task: {obs_dict['task_name']}
Instructions: {obs_dict['instructions']}
Step: {obs_dict['step_number'] + 1} / {obs_dict['total_steps']}

--- EMAIL ---
From: {email['sender']}
Date: {email['timestamp']}
Subject: {email['subject']}

{email['body']}
--- END EMAIL ---

Provide your action as a JSON object."""


def llm_action(client: OpenAI, obs_dict: Dict[str, Any]) -> Action:
    """Call the LLM and parse its response into an Action."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(obs_dict)},
    ]
    response = client.chat.completions.create(
        model=MODEL,
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
        # Attempt lenient extraction
        import re
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            print(f"  [WARN] Could not parse LLM response: {raw[:200]}")
            data = {}

    return Action(**data)


def run_task(client: OpenAI, env: EmailTriageEnv, task_id: str) -> float:
    """Run one full episode and return the final average score."""
    print(f"\n{'='*60}")
    print(f"Task: {task_id}")
    print(f"{'='*60}")

    obs = env.reset(task_id=task_id)
    obs_dict = obs.model_dump()
    total_reward = 0.0
    steps = 0

    while True:
        print(f"\nStep {obs_dict['step_number'] + 1}/{obs_dict['total_steps']}: "
              f"{obs_dict['current_email']['subject'][:60]}")

        action = llm_action(client, obs_dict)
        next_obs, reward, done, info = env.step(action)

        total_reward += reward.score
        steps += 1
        print(f"  Score: {reward.score:.3f} | {reward.feedback[:100]}")

        if done:
            final_score = info.get("final_score", total_reward / steps)
            print(f"\n  {info.get('summary', '')}")
            return final_score

        obs_dict = next_obs.model_dump()


def main():
    parser = argparse.ArgumentParser(description="EmailTriageEnv baseline inference")
    parser.add_argument("--task", default="spam_classification",
                        choices=list(EmailTriageEnv.available_tasks().keys()),
                        help="Task to evaluate")
    parser.add_argument("--all", action="store_true", help="Run all tasks")
    args = parser.parse_args()

    client = get_client()
    env = EmailTriageEnv()

    tasks = list(EmailTriageEnv.available_tasks().keys()) if args.all else [args.task]

    results: Dict[str, float] = {}
    for task_id in tasks:
        score = run_task(client, env, task_id)
        results[task_id] = score
        print(f"\n>>> {task_id} final score: {score:.4f}")

    print(f"\n{'='*60}")
    print("BASELINE RESULTS")
    print(f"{'='*60}")
    for tid, sc in results.items():
        print(f"  {tid:<35} {sc:.4f}")
    if results:
        avg = sum(results.values()) / len(results)
        print(f"  {'AVERAGE':<35} {avg:.4f}")
    print(f"{'='*60}")
    print(f"Model: {MODEL}")


if __name__ == "__main__":
    main()
