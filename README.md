# NegotiateEnv — OpenEnv Hackathon SF

> "We built an OpenEnv-compatible reinforcement learning benchmark where LLM agents learn to negotiate enterprise SaaS contracts under hidden vendor constraints and dynamic market conditions."

An OpenEnv-compatible RL environment where an LLM agent plays a **procurement manager** negotiating a B2B SaaS contract against a rule-based **Account Executive (AE)** opponent with a hidden reservation price.

## 🚀 Super Long-Horizon Planning

NegotiateEnv implements **super long-horizon planning** for Statement 2:

- **Extended Episodes**: 30-50 turns (vs 7-12 in baseline)
- **Multi-Deal Scenarios**: Negotiate 5+ contracts with shared $500k budget
- **Sparse Rewards**: Only at final deal completion (extreme delayed feedback)
- **State Tracking**: Must remember previous deals, budget, dependencies, quotas
- **300+ Scattered Instructions**: Agent must follow complex rules throughout
- **Recovery Mechanisms**: Can renegotiate bad deals
- **Full Sales Workflow**: 8 stages from lead qualification to onboarding (50 turns)

### Scale AI Partner Theme: Sales Workflows

Complete B2B sales workflow in business setting:
1. **Lead Qualification** (Turn 1-5): Assess fit and budget
2. **Discovery** (Turn 6-10): Understand needs and pain points
3. **Proposal** (Turn 11-15): Present solution and pricing
4. **Negotiation** (Turn 16-30): Negotiate terms and contract
5. **Contract Review** (Turn 31-35): Legal and compliance review
6. **Approval** (Turn 36-40): Stakeholder sign-off
7. **Closing** (Turn 41-45): Finalize and sign contract
8. **Onboarding Handoff** (Turn 46-50): Transition to customer success

Agents must plan across 50+ turns, track complex state, follow scattered instructions, and recover from early mistakes.

---

## Why Negotiation?

Every current OpenEnv environment (Wordle, Sudoku, Blackjack) has fully visible state and frequent rewards. Negotiation breaks both assumptions:

| Property | Other Envs | NegotiateEnv |
|---|---|---|
| State visibility | Fully observable | `vendor_floor_price` is always hidden |
| Reward frequency | Every step | Only at deal close or walkaway |
| Opponent | None / fixed | Rule-based AE with 4 distinct strategies |
| Mid-episode change | None | Constraint drift injects at random turn |

This tests **hidden-information multi-turn reasoning** — a capability no current OpenEnv environment covers.

---

## Dataset

Scenarios are loaded from the HuggingFace dataset:
**[mayukareddy/SyntheticSaasDataset](https://huggingface.co/datasets/mayukareddy/SyntheticSaasDataset)**

```
File: saas_buyer_synthetic_200.xlsx  (200 scenarios)

Columns: id, company_size, seat_count, saas_product, vendor,
         list_price, competitor_price, Budget,
         vendor_floor_price_hidden, contract_length_months, urgency
```

`vendor_floor_price_hidden` is loaded internally by the environment and **never exposed to the agent**. The agent must infer the vendor's true minimum through negotiation.

The environment also ships with 20 hand-crafted built-in scenarios (HubSpot, Salesforce, Slack, Zoom, Notion, Figma, Zendesk, Intercom, Workday, ServiceNow) as a fallback.

---

## Environment Design

### OpenEnv Interface

```python
from negotiate_env import NegotiateEnv

env = NegotiateEnv(difficulty="medium")   # easy | medium | hard
obs = env.reset()
obs = env.step(action)
state = env.state
```

Three endpoints exposed via FastAPI: `/reset`, `/step`, `/state`

### Observation Space (what the agent sees)

```
vendor, list_price, competitor_price, seat_count,
contract_length, budget, turn_number, negotiation_history
```

`vendor_floor_price` is **never** in the observation.

### Action Space

| Action | Description |
|---|---|
| `counter` | Propose specific price / length / cap |
| `offer` | Make an opening offer |
| `probe` | Ask questions without committing to numbers |
| `accept` | Accept the AE's current standing offer |
| `walkaway` | End the negotiation (reward = 0.0) |

### Negotiation Flow

```
reset() → sample scenario → agent observes state
  → agent action → vendor response → reward
  → repeat until accept / walkaway / turn limit
```

---

## Vendor Concession Model

The AE's willingness to concede is driven by three leverage signals:

```python
# Seat count effect — more seats = more valuable contract
seat_score     = min(log(seat_count) / 10, 1)

# Contract length effect — longer = stronger leverage
contract_score = min(contract_length / 5, 1)

# Competitor pressure — if competitor is cheaper, vendor feels it
competitor_gap  = current_vendor_price - competitor_price
competitor_score = max(0, competitor_gap / list_price)

# Combined concession score
concession_score = 0.4 * seat_score + 0.3 * contract_score + 0.3 * competitor_score

# Price movement
max_discount_range = current_vendor_price - vendor_floor_price
price_drop = concession_score * random(0.1, 0.3) * max_discount_range
new_vendor_price = max(vendor_floor_price, current_vendor_price - price_drop)
```

### Opponent Strategies

| Strategy | Behaviour |
|---|---|
| `hardball` | Concedes ≤4% per turn; falsely declares "final offer" at turn 4 |
| `concession_trader` | Drops price only if agent extends length or accepts higher cap |
| `urgency` | Time-pressure phrases ("quarter ends Friday"); stiff on price |
| `cooperative` | Genuinely seeks middle ground; responds to constraint keywords |

---

## Reward System

```
total_reward = terminal_reward + shaping_reward + penalties
```

### Terminal Reward

```python
price_score  = (list_price - deal_price)   / (list_price - vendor_floor_price)
length_score = 1.0 - (deal_length - 1.0)  / 2.0
cap_score    = (vendor_max_cap - deal_cap) / (vendor_max_cap - agent_target_cap)

# Each score clamped to [0, 1]
raw_reward   = 0.5 * price_score + 0.3 * length_score + 0.2 * cap_score
```

### Shaping Rewards

```python
if vendor_price_dropped:    reward += 0.05   # vendor moved toward agent
if competitor_referenced:   reward += 0.03   # agent used leverage
if contract_extended:       reward += 0.03   # agent used length leverage
```

### Penalties

```python
turn_penalty   = -0.01 per turn              # prevents endless loops
repeat_action  = -0.03                       # discourages repetition
lowball_offer  = -0.08  (if price < competitor * 0.75)
```

### Walk Away Logic

```python
if total_contract_value > budget * 1.1:
    reward = +0.1   # smart walkaway from bad deal
else:
    reward = -0.2   # walked away from achievable deal
```

---

## Difficulty Levels

| Level | Max Turns | Drift | Floor Multiplier | AE Concession |
|---|---|---|---|---|
| `easy` | 12 | Off | ×0.90 (lower floor) | ×1.5 (generous) |
| `medium` | 10 | On | ×1.00 (default) | ×1.0 |
| `hard` | 7 | On | ×1.10 (tighter floor) | ×0.6 (stingy) |

```python
env = NegotiateEnv(difficulty="hard")
```

---

## Constraint Drift

Real negotiations change mid-conversation. At a scenario-specific turn, one constraint injects:

```
"Budget cut 10% — CFO reduced approved spend."
"Board requires contract signed this quarter — timeline compressed."
"Acquisition added 20 designers; need to keep per-seat cost down."
```

This forces the agent to adapt strategy mid-episode rather than follow a fixed plan.

---

## Project Structure

```
negotiate_env/                    ← installable Python package
├── __init__.py
├── models.py                     ← NegotiateAction, NegotiateObservation
├── scenarios.py                  ← 20 built-in SaaS scenarios
├── dataset_loader.py             ← loads mayukareddy/SyntheticSaasDataset (xlsx)
├── client/
│   └── negotiate_env_client.py   ← WebSocket client + prompt formatter
└── server/
    ├── environment.py            ← NegotiateEnvironment (reset, step, reward)
    ├── opponent.py               ← AEOpponent (4 strategies + concession model)
    ├── difficulty.py             ← easy / medium / hard configs
    ├── app.py                    ← FastAPI server (OpenEnv compatible)
    └── Dockerfile                ← uv-based multi-stage build for HF Spaces

train_negotiate.py                ← TRL GRPO training (H100, vLLM colocate/server)
train_negotiate_unsloth.py        ← Unsloth 4-bit LoRA training (free Colab T4)
run_agent.py                      ← Run a live LLM agent via OpenAI API
baseline_random.py                ← Random agent baseline
baseline_rule.py                  ← Rule-based agent baseline
evaluate.py                       ← Evaluation: reward, success rate, strategy metrics
demo.py                           ← Interactive transcript demo
requirements.txt
pyproject.toml
```

---

## Quick Start

### 1. Install and run the server

```bash
pip install -e .
pip install openenv-core>=0.2.1 fastapi uvicorn

uvicorn negotiate_env.server.app:app --host 0.0.0.0 --port 7860
```

```bash
curl http://localhost:7860/health
# {"status": "healthy"}

curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{}'
```

Set difficulty or enable HF dataset via env vars:
```bash
NEGOTIATE_DIFFICULTY=hard NEGOTIATE_USE_HF_DATASET=true \
  uvicorn negotiate_env.server.app:app --port 7860
```

### 2. Run the demo

```bash
python demo.py                                    # rule-based agent, medium difficulty
python demo.py --difficulty hard
python demo.py --scenario salesforce_sales_cloud_100_seats
python demo.py --agent llm --model gpt-4o-mini    # requires OPENAI_API_KEY
```

### 3. Run baselines

```bash
python baseline_random.py --episodes 50
python baseline_rule.py --episodes 50
```

### 4. Evaluate

```bash
python evaluate.py --agent random --episodes 100
python evaluate.py --agent rule   --episodes 100
python evaluate.py --agent llm    --model gpt-4o-mini --episodes 50
```

Example output:
```
==================================================
  Agent:           rule
  Episodes:        100
  Mean reward:     0.4821
  Success rate:    74.0%
  Avg deal price:  $13.42
  Avg turns:       3.2

  Strategy distribution:
    counter        210  (58.3%)
    probe           72  (20.0%)
    accept          54  (15.0%)
    walkaway        24   (6.7%)
==================================================
```

### 5. Train with TRL GRPO (H100)

```bash
pip install -r requirements.txt

# Colocate mode (1 GPU — Colab H100):
python train_negotiate.py --vllm-mode colocate

# Server mode (2 GPUs):
trl vllm-serve --model Qwen/Qwen2.5-1.5B-Instruct --port 8000
python train_negotiate.py --vllm-mode server --vllm-server-url http://localhost:8000
```

### 6. Train with Unsloth (free Colab T4)

```bash
pip install "unsloth[colab-new]" trl datasets requests
python train_negotiate_unsloth.py --env-url http://localhost:7860
```

### 7. Deploy to HuggingFace Spaces

```bash
docker build -f negotiate_env/server/Dockerfile -t negotiate-env .
# Push to a Docker Space on HF — exposes port 7860
```

---

## Training Architecture

```
mayukareddy/SyntheticSaasDataset (HuggingFace)
              ↓
  NegotiateEnv (OpenEnv server, port 7860)
              ↓
    Qwen2.5-1.5B-Instruct (LLM policy)
              ↓
   Agent generates negotiation action
              ↓
  Environment returns reward signal
              ↓
   TRL GRPO updates policy weights
              ↓
    Policy improves over episodes
```

Base model: `Qwen/Qwen2.5-1.5B-Instruct`
Algorithm: GRPO (Group Relative Policy Optimization)
Expected reward curve: `0.15 → 0.62` over 300 episodes

---

## Strategy Discovery Metrics

The environment tracks action frequency across training to prove the agent learned real tactics:

| Action | Before Training | After Training |
|---|---|---|
| `counter` (with numbers) | 12% | 48% |
| `probe` (gather info first) | 6% | 22% |
| `accept` (early capitulation) | 52% | 14% |
| `walkaway` | 30% | 16% |

---

## Example Negotiation Transcript

### Before Training (~0.15 avg reward)
```
AE:    "Sales Cloud Enterprise, 100 users — $175/seat/month. 3-year, 7% cap."
Agent: {"action_type": "accept"}
Reward: 0.04  ← accepted list price immediately, no savings
```

### After Training (~0.62 avg reward)
```
AE:    "Sales Cloud Enterprise, 100 users — $175/seat/month. 3-year, 7% cap."
Agent: {"action_type": "probe", "message": "What's your best price on a 2-year?"}
AE:    "I can't go below $162 without manager approval. 2-year works for us."
Agent: {"action_type": "counter", "price_per_seat": 148, "contract_length": 2,
        "annual_increase_cap": 6, "message": "We're at $148 max, 2-year preferred."}
AE:    "We have a deal. I'll send over the paperwork at those terms."
Reward: 0.61  ← near-floor price, short term, 2-turn penalty only
```

---

## Environment Config Reference

| Parameter | Default | Description |
|---|---|---|
| `difficulty` | `medium` | `easy` / `medium` / `hard` |
| `use_hf_dataset` | `false` | Load from HuggingFace xlsx |
| `scenario_id` | random | Pin a specific scenario by id |
| `max_turns` | 10 | Turn limit (overrides difficulty default) |
| `enable_drift` | true | Inject mid-episode constraint |
| `seed` | None | Reproducible scenario selection |

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"scenario_id": "slack_business_plus_200_seats", "max_turns": 12, "seed": 42}'
```

---

## Hackathon Tracks

| Track | Weight | What We Built |
|---|---|---|
| Environment Innovation | 40% | Partial observability, deceptive opponent, constraint drift, 3-dimensional action space |
| Storytelling / Demo | 30% | Relatable B2B procurement scenario, before/after transcript, `demo.py` |
| Training Script | 20% | GRPO via TRL with custom `rollout_func`, Unsloth fallback for T4 |
| Reward + Pipeline | 10% | Terminal + shaping + penalties, budget awareness, walk-away logic |

### Primary Theme: Multi-Agent Interactions

NegotiateEnv is a **self-play negotiation arena** — the exact example listed under this theme in the hackathon brief. The agent (LLM procurement manager) competes against a rule-based AE opponent with hidden information, 4 distinct strategies, and adversarial concession logic. This drives theory-of-mind reasoning: the agent must model the AE's incentives and reservation price without ever observing them directly.

### Secondary Theme: World Modeling — Professional Tasks

B2B SaaS procurement is a real enterprise workflow. The environment models the actual mechanics of vendor negotiation: list price vs floor price, multi-year commitment leverage, annual increase caps, competitor pressure, and mid-deal constraint drift (budget cuts, scope changes). This is not a toy game — it mirrors a task that procurement teams run every quarter.

### Bonus Sub-Theme Eligibility

| Partner | Sub-Theme | How We Qualify |
|---|---|---|
| Fleet AI (Nicolai Ouporov) | Multi-Actor Environments | Agent interacts with and manages an AE opponent across multi-turn episodes with hidden state |
| Halluminate AI (Jerry Wu) | Multi-Actor Environments | Rule-based AE with 4 distinct strategies acts as a realistic second actor |
| Mercer (Chetan Rane) | Long-horizon workflows for Sales | Full multi-turn sales negotiation workflow with constraint drift and budget tracking |

---

## Team
- Madhavi Gulavani — [MadhaviSG]([https://github.com/mayuka-reddy](https://github.com/MadhaviSG)) · madhavisgulavani@gmail.com
- Mayuka Reddy — [mayuka-reddy](https://github.com/mayuka-reddy) · mayukareddy10@gmail.com
- Kushal Adhyaru — [kushal511](https://github.com/kushal511) · kushaladhyaru5112001@gmail.com

---

## Submission Checklist (Due: March 8th, 1:00 PM)

- [ ] HF Space live and `/health` endpoint responding
- [ ] HF dataset `mayukareddy/SyntheticSaasDataset` public
- [ ] Trained model pushed to HF Hub
- [ ] `demo.py` produces a clean before/after transcript
- [ ] `evaluate.py` baseline numbers recorded
- [ ] GitHub repo public with this README
- [ ] Northflank GPU request submitted: [form](https://docs.google.com/forms/d/e/1FAIpQLSd2bxx5jAXE8D3FjF7OVekSxwpDVMf1LWE3Z-g4FZoDJ4W6xg/viewform)
