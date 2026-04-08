# Email Triage OpenEnv

> A real-world email triage environment compliant with the [OpenEnv](https://openenv.dev) specification.  
> Built for the **Meta OpenEnv Hackathon**.

---

## Overview & Motivation

Email is one of the most universal and time-consuming tasks in modern work. Professionals spend hours daily triaging inboxes — identifying spam, prioritising urgent messages, and crafting appropriate responses. This environment benchmarks an AI agent's ability to perform these three real-world sub-tasks with increasing complexity.

The environment is Gym-like but fully typed via Pydantic, and exposes a REST API so any HTTP-capable agent can interact with it.

---

## Action & Observation Spaces

### Observation

| Field              | Type                  | Description                                 |
|--------------------|-----------------------|---------------------------------------------|
| `task_id`          | `str`                 | Current task identifier                     |
| `task_name`        | `str`                 | Human-readable task name                    |
| `task_description` | `str`                 | Task objective description                  |
| `instructions`     | `str`                 | What fields to populate in the action       |
| `current_email`    | `Email`               | The email to process (subject, body, etc.)  |
| `step_number`      | `int`                 | 0-indexed current step                      |
| `total_steps`      | `int`                 | Total emails in the episode                 |
| `cumulative_score` | `float`               | Running sum of rewards so far               |
| `history`          | `list[dict]`          | Past actions and rewards                    |

### Action

| Field           | Type                                              | Used in task(s)              |
|-----------------|---------------------------------------------------|------------------------------|
| `spam_label`    | `"spam" \| "not_spam" \| None`                   | Task 1                       |
| `priority`      | `"high" \| "medium" \| "low" \| None`            | Task 2                       |
| `category`      | `"work" \| "personal" \| "finance" \| "promotion" \| "support" \| "other" \| None` | Task 2 |
| `response_text` | `str \| None`                                    | Task 3                       |
| `reasoning`     | `str \| None`                                    | All tasks (encouraged)       |

### Reward

| Field       | Type               | Description                         |
|-------------|--------------------|-------------------------------------|
| `score`     | `float ∈ [0, 1]`  | Per-step reward                     |
| `feedback`  | `str`              | Explanation of the score            |
| `breakdown` | `dict[str, float]` | Per-criterion scores                |

---

## Tasks

### Task 1 – Spam Classification *(Easy)*

- **Steps:** 5 emails  
- **Action:** Set `spam_label` to `"spam"` or `"not_spam"`  
- **Grading:** Binary correctness. 1.0 for correct, 0.0 for incorrect. Small bonus for including `reasoning`.  
- **Emails include:** Lottery scam, phishing, legitimate work emails, promotional spam

### Task 2 – Email Prioritization & Categorization *(Medium)*

- **Steps:** 5 emails  
- **Action:** Set `priority` and `category`  
- **Grading:** 0.5 weight each. Priority has partial credit for adjacent levels (e.g., medium when correct is high → 0.5 rather than 0.0).  
- **Emails include:** Production outage, invoice due, newsletter, HR review notice, cafeteria menu

### Task 3 – Professional Email Response Generation *(Hard)*

- **Steps:** 3 emails  
- **Action:** Set `response_text` to a full professional reply  
- **Grading:** Multi-criteria deterministic rubric:
  - **Keyword coverage (35%):** Does the response address the key topics?
  - **Structure (25%):** Does it include a greeting and professional closing?
  - **Length (20%):** Is it sufficiently detailed (≥60–80 words)?
  - **Tone (20%):** Does it avoid inappropriate/dismissive phrases?
- **Emails include:** Angry refund demand, enterprise sales inquiry, security vulnerability disclosure

---

## Reward Function

- Rewards are issued **per step**, not only at episode end
- **Partial credit** is given for adjacent priority levels and partial keyword matches
- **Empty actions** receive 0.0 (penalises no-ops)
- **Forbidden phrases** in Task 3 reduce tone score (penalises dismissive responses)
- The final episode score is the **average step reward** ∈ [0.0, 1.0]

---

## Setup & Usage

### Local (Python)

```bash
pip install -r requirements.txt

# Start the REST API
python app.py
# → http://localhost:7860

# Run baseline inference (requires HF_TOKEN)
HF_TOKEN=hf_xxx python inference.py --all
```

### REST API

```bash
# Reset to a task
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "spam_classification"}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"spam_label": "spam", "reasoning": "Lottery scam email"}}'

# Get current state
curl http://localhost:7860/state
```

### Docker

```bash
# Build
docker build -t email-triage-openenv .

# Run
docker run -p 7860:7860 email-triage-openenv

# With inference credentials
docker run -p 7860:7860 -e HF_TOKEN=hf_xxx email-triage-openenv
```

### Python (direct)

```python
from environment import EmailTriageEnv, Action

env = EmailTriageEnv()
obs = env.reset("spam_classification")

while True:
    action = Action(spam_label="spam", reasoning="Looks like phishing")
    obs, reward, done, info = env.step(action)
    print(reward.score, reward.feedback)
    if done:
        print(info["final_score"])
        break
```

---

## Baseline Performance

Model: `Qwen/Qwen2.5-7B-Instruct` via HuggingFace Inference API

| Task                          | Difficulty | Baseline Score |
|-------------------------------|------------|----------------|
| spam_classification           | Easy       | ~0.90          |
| email_prioritization          | Medium     | ~0.68          |
| email_response                | Hard       | ~0.62          |
| **Average**                   |            | **~0.73**      |

> Scores are approximate and depend on model temperature and prompt sensitivity.

---

## Project Structure

```
openEnv/
├── environment/
│   ├── __init__.py       # Package exports
│   ├── env.py            # EmailTriageEnv (reset/step/state)
│   ├── models.py         # Pydantic: Observation, Action, Reward, EnvState
│   ├── tasks.py          # Email data + ground truth labels
│   └── graders.py        # Deterministic per-task graders
├── app.py                # FastAPI REST server (HF Spaces entrypoint)
├── inference.py          # Baseline inference script (OpenAI-compatible)
├── openenv.yaml          # OpenEnv metadata
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Tags

`openenv` `email` `nlp` `real-world` `classification` `generation`
