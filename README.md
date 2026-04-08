---
title: MailSort Enterprise Email Triage
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - email-triage
license: bsd-3-clause
---

# MailSort — Enterprise Email Triage OpenEnv Environment

**MailSort** is a production-ready [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment
that trains and evaluates AI agents on **enterprise email triage** — a genuine real-world task
performed by millions of knowledge workers daily.

Agents must classify, prioritize, detect phishing, route, and draft responses for realistic
enterprise emails across three tasks of increasing difficulty.

---

## Why Email Triage?

Enterprise teams receive hundreds of emails per day. Misclassifying an urgent escalation as
routine costs money and damages relationships. Failing to detect phishing exposes the company
to breaches. A well-trained triage agent provides measurable operational value immediately.

This environment fills a real gap: existing OpenEnv benchmarks focus on code, games, or
web navigation. MailSort provides the first structured environment for **document understanding
+ decision-making under priority pressure**.

---

## Observation Space

The agent receives a `MailSortObservation` JSON object after each `reset()` and `step()`:

| Field | Type | Description |
|-------|------|-------------|
| `task_name` | `string` | Active task ID |
| `task_description` | `string` | Full task description shown at episode start |
| `step_description` | `string` | Instructions for the **current** step |
| `emails` | `list[Email]` | Email objects the agent must process |
| `step` | `int` | Current step number (0 = not started, 1+ = in progress) |
| `max_steps` | `int` | Total steps in this episode |
| `feedback` | `string\|null` | Grader feedback from the previous step |
| `last_action_error` | `string\|null` | Validation error if last action was malformed |
| `done` | `bool` | Whether the episode has ended |
| `reward` | `float` | Reward for the last step |

### Email Object Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | `string` | Unique email ID (e.g. `e1_01`) |
| `subject` | `string` | Email subject line |
| `sender` | `string` | Sender display name |
| `sender_email` | `string` | Sender email address |
| `body` | `string` | Email body text |
| `metadata` | `object` | Extra signals: `x_priority`, `recipient_count` |

---

## Action Space

The agent submits a `MailSortAction` JSON object to `step()`.
All fields are optional — use only what the current task/step requires:

| Field | Type | Used In | Description |
|-------|------|---------|-------------|
| `classifications` | `list[Classification]` | T1 step 1, T2 step 1, T3 step 1 | Per-email category + priority |
| `rankings` | `list[string]` | T2 step 1 | Ordered email IDs (most urgent first) |
| `routings` | `list[Routing]` | T3 step 2 | Per-email department assignment |
| `response_draft` | `string` | T3 step 3 | Acknowledgment response draft |

### Classification Entry

```json
{
  "email_id": "e1_01",
  "category": "urgent",
  "priority": "critical",
  "is_phishing": false
}
```

**Valid categories:** `spam` | `urgent` | `routine` | `billing` | `hr` | `security`

**Valid priorities:** `critical` | `high` | `medium` | `low`

### Routing Entry

```json
{"email_id": "e3_02", "dept": "support"}
```

**Valid departments:** `engineering` | `support` | `hr` | `finance` | `security` | `management` | `sales`

---

## Tasks

### Task 1: `email_classify` — Easy (1 step)

**Objective:** Classify a single enterprise email into one of six categories and assign a priority level.

**Input:** One email with clear, unambiguous signals.

**Output:** `classifications: [{email_id, category, priority}]`

**Scoring:**
- Category correct: **0.6** (with partial credit for adjacent categories)
- Priority correct: **0.4** (with partial credit for adjacent levels)
- Partial credit examples: security ≈ urgent (0.4×), high ≈ critical (0.5×)

**Expected baseline score (Qwen2.5-72B):** 0.70–0.90

---

### Task 2: `email_rank` — Medium (1 step)

**Objective:** Given 5 enterprise emails in random order, rank them from most to least urgent
AND classify each one.

**Input:** Five emails across diverse categories (urgent, security, billing, HR, routine).

**Output:** `rankings: [id1, id2, ...]` + `classifications: [...]`

**Scoring:**
- Ranking quality: **0.5 × Kendall Tau** (normalized to [0,1])
- Classification accuracy: **0.5 × average per-email score**

**Expected baseline score (Qwen2.5-72B):** 0.50–0.70

---

### Task 3: `email_triage` — Hard (3 steps)

**Objective:** Full multi-step triage of three complex emails, including phishing detection,
department routing, and response drafting.

**Emails include:**
- A phishing email disguised as an IT helpdesk alert (typosquatted domain)
- A critical customer escalation ($4.2M ARR account)
- A routine HR compliance reminder

**Step 1 — Classify + Phishing Detection:**
- Output: `classifications` with `is_phishing` field
- Score: `0.6 × avg_classification + 0.4 × phishing_accuracy`

**Step 2 — Route to Department:**
- Output: `routings` for each email
- Score: average routing accuracy (exact match = 1.0, valid but wrong = 0.3)

**Step 3 — Draft Response:**
- Output: `response_draft` (acknowledgment for the critical escalation)
- Score: heuristic quality (length, professional tone, relevance, no placeholders)

**Episode score:** average of 3 step scores

**Expected baseline score (Qwen2.5-72B):** 0.40–0.65

---

## Reward Function

The reward function provides **continuous signal** across the full trajectory:

| Component | Signal Type | Range |
|-----------|-------------|-------|
| Category classification | Partial credit via adjacency table | [0.0, 1.0] |
| Priority assignment | Partial credit for adjacent levels | [0.0, 1.0] |
| Ranking quality | Kendall Tau correlation | [0.0, 1.0] |
| Phishing detection | Binary per-email accuracy | [0.0, 1.0] |
| Routing accuracy | Exact/partial department match | [0.0, 1.0] |
| Response quality | Keyword + length heuristics | [0.0, 1.0] |

**Penalties (applied before flooring at 0.0):**
- Invalid field value (e.g., unknown category): **−0.05**
- Malformed/empty action: **−0.10**

---

## Baseline Scores

Evaluated with **Qwen/Qwen2.5-72B-Instruct** via HuggingFace router, temperature=0:

| Task | Difficulty | Score |
|------|------------|-------|
| `email_classify` | Easy | ~0.80 |
| `email_rank` | Medium | ~0.60 |
| `email_triage` | Hard | ~0.52 |

---

## Setup & Usage

### Prerequisites

- Python 3.10+
- Docker (for containerized deployment)
- An API key with access to an OpenAI-compatible LLM endpoint

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# Verify the server is running
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "email_classify"}'
```

### Docker

```bash
# Build
docker build -t mailsort-env .

# Run
docker run -p 8000:8000 mailsort-env

# Health check
curl http://localhost:8000/health
```

### Run Baseline Inference

```bash
export HF_TOKEN="your-huggingface-token"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export ENV_BASE_URL="http://localhost:8000"

python inference.py
```

Expected output format:
```
[START] task=email_classify env=mailsort model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={...} reward=0.80 done=true error=null
[END] success=true steps=1 score=0.80 rewards=0.80

[START] task=email_rank env=mailsort model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={...} reward=0.60 done=true error=null
[END] success=true steps=1 score=0.60 rewards=0.60

[START] task=email_triage env=mailsort model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={...} reward=0.65 done=false error=null
[STEP] step=2 action={...} reward=0.67 done=false error=null
[STEP] step=3 action={...} reward=0.30 done=true error=null
[END] success=true steps=3 score=0.54 rewards=0.65,0.67,0.30
```

### With Docker Image (inference)

```bash
export IMAGE_NAME="mailsort-env:latest"
python inference.py
```

---

## API Reference

### POST /reset

Start a new episode.

**Request body (optional):**
```json
{"task": "email_classify", "seed": null, "episode_id": null}
```

**Response:**
```json
{
  "observation": {...},
  "reward": 0.0,
  "done": false,
  "state": {...}
}
```

### POST /step

Submit an action.

**Request body:**
```json
{
  "classifications": [{"email_id": "e1_01", "category": "urgent", "priority": "critical"}]
}
```

### GET /state

Return current episode state.

### GET /tasks

List all available tasks with metadata.

### GET /health

Liveness probe — returns `{"status": "ok"}`.

---

## Project Structure

```
mailsort-env/
├── models.py              # Pydantic models: MailSortAction, MailSortObservation, MailSortState
├── client.py              # MailSortEnv client (EnvClient subclass)
├── openenv.yaml           # OpenEnv manifest
├── inference.py           # Baseline inference script
├── requirements.txt       # Dependencies
├── README.md              # This file
├── Dockerfile             # Container definition
└── server/
    ├── __init__.py
    ├── app.py             # FastAPI application (create_app)
    ├── environment.py     # MailSortEnvironment class
    ├── tasks.py           # Task registry + deterministic graders
    ├── email_data.py      # Synthetic email corpus (30+ emails)
    └── rewards.py         # Reward computation helpers
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | Yes (inference) | — | HuggingFace / API key |
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `ENV_BASE_URL` | No | `http://localhost:8000` | MailSort server URL |
| `IMAGE_NAME` | No | — | Docker image name (if using from_docker_image) |

---
 
