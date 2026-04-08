"""
FastAPI app exposing EmailTriageEnv as a REST API for Hugging Face Spaces.
"""
from __future__ import annotations
import os
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from environment import EmailTriageEnv, Action

app = FastAPI(
    title="Email Triage OpenEnv",
    description="Real-world email triage environment compliant with the OpenEnv spec.",
    version="0.1.0",
)

# Single shared environment instance (stateful for demo; use session IDs in production)
_env = EmailTriageEnv()


class ResetRequest(BaseModel):
    task_id: str = "spam_classification"


class StepRequest(BaseModel):
    action: Dict[str, Any]


@app.get("/")
def root():
    return {
        "name": "Email Triage OpenEnv",
        "version": "0.1.0",
        "tasks": EmailTriageEnv.available_tasks(),
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/health"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tasks")
def list_tasks():
    return {"tasks": EmailTriageEnv.available_tasks()}


@app.post("/reset")
def reset(req: Optional[ResetRequest] = Body(default=None)):
    task_id = req.task_id if req else "spam_classification"
    try:
        obs = _env.reset(task_id=task_id)
        return {"observation": obs.model_dump()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(req: StepRequest):
    try:
        action = Action(**req.action)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid action: {e}")

    try:
        obs, reward, done, info = _env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "observation": obs.model_dump() if obs else None,
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    return {"state": _env.state().model_dump()}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
