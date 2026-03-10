#!/usr/bin/env python3
"""
NegotiateEnv GRPO Training Script — Unsloth edition
Trains Qwen/Qwen2.5-1.5B-Instruct with 4-bit LoRA to negotiate B2B SaaS contracts.

Advantages over the TRL/vLLM script:
  - Runs on a FREE Colab T4 GPU (no H100 needed)
  - ~2x faster and ~70% less VRAM via Unsloth 4-bit quantisation
  - No separate vLLM server process required

Install (run these in Colab):
    pip install "unsloth[colab-new]" "openenv-core>=0.2.1" requests datasets
    pip install git+https://huggingface.co/spaces/YOUR_HF_USERNAME/negotiate-env

    # OR test locally without HF Spaces:
    pip install -e /path/to/OpenEnv   # installs negotiate_env package
    uvicorn negotiate_env.server.app:app --host 0.0.0.0 --port 7860 &

Run:
    python train_negotiate_unsloth.py --env-url http://localhost:7860
    python train_negotiate_unsloth.py --env-url https://YOUR_USERNAME-negotiate-env.hf.space
"""

import argparse
import json
import re
import time

import requests
import torch
from datasets import Dataset

# Try to import GRPO, fall back to older TRL if not available
try:
    from trl import GRPOConfig, GRPOTrainer
    HAS_GRPO = True
except ImportError:
    HAS_GRPO = False
    print("[WARN] GRPOConfig not found in TRL. Please install TRL >= 0.11.0")
    print("Run: pip install --upgrade 'trl>=0.11.0'")
    import sys
    sys.exit(1)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="NegotiateEnv GRPO training with Unsloth")
parser.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
parser.add_argument("--env-url", default="http://localhost:7860",
                    help="URL of running NegotiateEnv server (local or HF Space)")
parser.add_argument("--max-turns", type=int, default=50,
                    help="Max turns per negotiation episode (long-horizon: 50)")
parser.add_argument("--num-episodes", type=int, default=1000,
                    help="Number of dataset entries for long-horizon training")
parser.add_argument("--output-dir", default="negotiate-unsloth-output")
parser.add_argument("--lora-rank", type=int, default=16)
parser.add_argument("--max-seq-length", type=int, default=1024)
cli_args = parser.parse_args()

# ---------------------------------------------------------------------------
# Module-level globals — set in main() before training begins
# ---------------------------------------------------------------------------

model = None      # FastLanguageModel instance; available to reward function
tokenizer = None  # tokenizer instance; available to reward function

# ---------------------------------------------------------------------------
# System prompt (same as TRL script for consistency)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert procurement manager negotiating a B2B SaaS contract.
Your goal is to get the best deal: lowest price per seat, shortest contract length,
and lowest annual price increase cap.

Respond with a single JSON action:
{"action_type": "counter", "price_per_seat": 85.0, "contract_length": 1.0,
 "annual_increase_cap": 3.0, "message": "Your message to the AE"}

action_type must be one of: "offer", "counter", "probe", "accept", "walkaway"
Always respond with valid JSON only."""

# ---------------------------------------------------------------------------
# Environment helpers (pure HTTP, no dependency on installed negotiate_env pkg)
# ---------------------------------------------------------------------------

def env_reset(env_url: str, scenario_id: str | None = None, difficulty: str = "hard") -> tuple[dict, str]:
    """Reset environment and return (observation, session_id)."""
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            payload = {"difficulty": difficulty}
            if scenario_id:
                payload["scenario_id"] = scenario_id
            r = requests.post(f"{env_url}/reset", json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()
            session_id = data.get("session_id")
            obs = data.get("observation", data)
            return obs, session_id
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[warn] Reset failed (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(retry_delay)
            else:
                raise


def env_step(env_url: str, action: dict, session_id: str) -> dict:
    """Take a step with the given session_id."""
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            payload = {"action": action, "session_id": session_id}
            r = requests.post(f"{env_url}/step", json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()
            return data.get("observation", data)
        except requests.exceptions.HTTPError as e:
            if attempt < max_retries - 1:
                # Log the error details for debugging
                try:
                    error_detail = r.json() if r.text else str(e)
                    print(f"[warn] Step failed (attempt {attempt+1}/{max_retries}): {e}")
                    print(f"[warn] Error detail: {error_detail}")
                    print(f"[warn] Action sent: {action}")
                except:
                    print(f"[warn] Step failed (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(retry_delay)
            else:
                raise
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[warn] Step failed (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(retry_delay)
            else:
                raise


def obs_to_prompt(obs: dict) -> str:
    """Format a raw observation dict as an LLM-readable string."""
    cur = obs.get("current_offer", {})
    hist = obs.get("conversation_history", [])[-4:]
    turns_left = obs.get("max_turns", 10) - obs.get("turn_number", 0)

    lines = [
        "## Negotiation Context",
        obs.get("context", ""),
        "",
        "## AE's Current Offer",
        f"  Price:    ${cur.get('price_per_seat', 0):.2f}/seat/month",
        f"  Length:   {cur.get('contract_length', 0):.0f} year(s)",
        f"  Ann. cap: {cur.get('annual_increase_cap', 0):.1f}%",
        "",
        "## Your Budget Limits",
        f"  Max price:  ${obs.get('your_max_price', 0):.2f}/seat/month",
        f"  Max length: {obs.get('your_max_length', 0):.0f} year(s)",
        f"  Max cap:    {obs.get('your_max_cap', 0):.1f}%",
    ]

    constraints = obs.get("active_constraints", [])
    if constraints:
        lines += ["", "## ⚠ Active Constraints"]
        lines += [f"  - {c}" for c in constraints]

    if hist:
        lines += ["", "## Recent Conversation"]
        lines += [f"  {h}" for h in hist]

    lines += [
        "",
        f"Turn {obs.get('turn_number', 0)} of {obs.get('max_turns', 10)} "
        f"({turns_left} remaining)",
        "",
        f'AE: "{obs.get("ae_message", "")}"',
        "",
        "Respond with valid JSON only:",
        '{"action_type": "counter", "price_per_seat": 0.0, '
        '"contract_length": 1.0, "annual_increase_cap": 5.0, "message": "..."}',
    ]
    return "\n".join(lines)


def parse_to_action(text: str) -> dict:
    """Parse LLM output to an action dict. Never crashes."""
    text = text.strip()
    valid_types = {"offer", "counter", "probe", "accept", "walkaway"}

    try:
        data = json.loads(text)
        if data.get("action_type") in valid_types:
            # Ensure all required fields are present with defaults
            return {
                "action_type": data.get("action_type", "counter"),
                "price_per_seat": float(data.get("price_per_seat", 0.0)),
                "contract_length": float(data.get("contract_length", 0.0)),
                "annual_increase_cap": float(data.get("annual_increase_cap", 0.0)),
                "message": str(data.get("message", "")),
            }
    except Exception:
        pass

    m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            if data.get("action_type") in valid_types:
                return {
                    "action_type": data.get("action_type", "counter"),
                    "price_per_seat": float(data.get("price_per_seat", 0.0)),
                    "contract_length": float(data.get("contract_length", 0.0)),
                    "annual_increase_cap": float(data.get("annual_increase_cap", 0.0)),
                    "message": str(data.get("message", "")),
                }
        except Exception:
            pass

    action_type = "counter"
    for at in ("accept", "walkaway", "probe", "offer", "counter"):
        if at in text.lower():
            action_type = at
            break

    price = 0.0
    dollar = re.search(r"\$\s*([\d]+(?:\.\d+)?)", text)
    if dollar:
        price = float(dollar.group(1))

    return {
        "action_type": action_type,
        "price_per_seat": price,
        "contract_length": 1.0,
        "annual_increase_cap": 5.0,
        "message": text[:200],
    }


# ---------------------------------------------------------------------------
# Reward function — runs a FULL multi-turn episode using model generations
# ---------------------------------------------------------------------------

def reward_negotiate(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """
    For each completion produced by GRPO's rollout:
      1. Reset the environment to a fresh episode.
      2. Turn 0: parse the pre-generated completion as the first action.
      3. Turns 1+: call model.generate for every subsequent turn so ALL turns
         use the model's current policy (not dummy probe actions).
      4. Return the final episode reward as the training signal.

    This ensures the reward reflects the model's full negotiation strategy,
    not just its first move.
    """
    if model is None or tokenizer is None:
        # model not yet loaded (e.g. import-time check)
        return [0.0] * len(completions)

    rewards = []
    was_training = model.training
    model.eval()

    try:
        for completion in completions:
            try:
                obs, session_id = env_reset(cli_args.env_url)
                if obs.get("done", False):
                    rewards.append(0.0)
                    continue

                final_reward = 0.0

                # Turn 0: use GRPO's pre-generated completion so its gradient matters
                action = parse_to_action(completion)
                obs = env_step(cli_args.env_url, action, session_id)
                if obs.get("done", False):
                    final_reward = float(obs.get("reward", 0.0))
                    rewards.append(final_reward)
                    continue

                # Turns 1+: generate fresh completions with model.generate
                for _turn in range(1, cli_args.max_turns):
                    if obs.get("done", False):
                        final_reward = float(obs.get("reward", 0.0))
                        break

                    user_content = obs_to_prompt(obs)
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ]
                    prompt_text = tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                    inputs = tokenizer(
                        prompt_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=cli_args.max_seq_length - 256,
                    ).to(model.device)

                    with torch.no_grad():
                        output_ids = model.generate(
                            **inputs,
                            max_new_tokens=256,
                            do_sample=True,
                            temperature=0.7,
                            pad_token_id=tokenizer.eos_token_id,
                        )

                    new_token_ids = output_ids[0][inputs["input_ids"].shape[1]:]
                    completion_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)

                    action = parse_to_action(completion_text)
                    obs = env_step(cli_args.env_url, action, session_id)

                    if obs.get("done", False):
                        final_reward = float(obs.get("reward", 0.0))

                rewards.append(final_reward)

            except Exception as e:
                print(f"[warn] Episode failed: {e}")
                rewards.append(0.0)

    finally:
        if was_training:
            model.train()

    return rewards


# ---------------------------------------------------------------------------
# Build dataset — each row gives the model its initial negotiation context
# ---------------------------------------------------------------------------

def build_dataset(env_url: str, n: int, tokenizer) -> Dataset:
    """
    Each dataset entry is a pre-formatted chat prompt from a fresh env.reset().
    The model generates its first negotiation action; reward_negotiate() scores it.
    """
    prompts = []
    for _ in range(n):
        try:
            obs, _ = env_reset(env_url)  # Ignore session_id for dataset building
            user_content = obs_to_prompt(obs)
        except Exception:
            user_content = "Negotiate a B2B SaaS contract. Respond with a JSON action."

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        prompts.append(prompt_text)

    return Dataset.from_dict({"prompt": prompts})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global model, tokenizer

    # ------------------------------------------------------------------
    # 0. Check environment server is accessible
    # ------------------------------------------------------------------
    print(f"[NegotiateEnv/Unsloth] Checking environment server at {cli_args.env_url}...")
    try:
        # Test with a reset call instead of health endpoint
        r = requests.post(f"{cli_args.env_url}/reset", json={}, timeout=10)
        r.raise_for_status()
        print(f"[NegotiateEnv/Unsloth] Environment server is accessible!")
    except Exception as e:
        print(f"[ERROR] Cannot reach environment server: {e}")
        print(f"[ERROR] Please check that {cli_args.env_url} is running")
        print(f"[ERROR] You may need to restart your HF Space if it's sleeping")
        import sys
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Load model with Unsloth 4-bit (works on T4 / any GPU >= 8 GB)
    # ------------------------------------------------------------------
    from unsloth import FastLanguageModel  # imported here so the file is importable without unsloth

    print(f"[NegotiateEnv/Unsloth] Loading {cli_args.model_id} with 4-bit quantisation...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cli_args.model_id,
        max_seq_length=cli_args.max_seq_length,
        load_in_4bit=True,
        fast_inference=False,   # Disable vLLM inside Unsloth; TRL handles generation
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # 2. Attach LoRA adapters
    # ------------------------------------------------------------------
    model = FastLanguageModel.get_peft_model(
        model,
        r=cli_args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=cli_args.lora_rank,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # ------------------------------------------------------------------
    # 2.5. Patch model to add warnings_issued attribute (TRL compatibility)
    # ------------------------------------------------------------------
    # TRL's GRPOTrainer expects model.warnings_issued but PEFT models don't have it
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    # Also patch the base model
    if hasattr(model, "base_model") and not hasattr(model.base_model, "warnings_issued"):
        model.base_model.warnings_issued = {}
    if hasattr(model, "base_model") and hasattr(model.base_model, "model") and not hasattr(model.base_model.model, "warnings_issued"):
        model.base_model.model.warnings_issued = {}

    # ------------------------------------------------------------------
    # 3. Build dataset (each entry = one fresh env observation as prompt)
    # ------------------------------------------------------------------
    print(f"[NegotiateEnv/Unsloth] Building dataset ({cli_args.num_episodes} episodes)...")
    dataset = build_dataset(cli_args.env_url, cli_args.num_episodes, tokenizer)

    # ------------------------------------------------------------------
    # 4. GRPO config — no vLLM, Unsloth handles the model efficiency
    # ------------------------------------------------------------------
    grpo_config = GRPOConfig(
        output_dir=cli_args.output_dir,
        use_vllm=False,                     # Unsloth uses its own optimised kernels
        num_train_epochs=1,
        num_generations=4,                  # lower than TRL script to fit T4 VRAM
        max_completion_length=256,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        logging_steps=5,
        save_steps=50,
        bf16=True,                          # Use bf16 to avoid dtype mismatches
        report_to="none",
    )

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_negotiate],
        train_dataset=dataset,
        args=grpo_config,
    )

    print("[NegotiateEnv/Unsloth] Starting GRPO training...")
    trainer.train()

    # ------------------------------------------------------------------
    # 6. Save merged model (optional — good for demo)
    # ------------------------------------------------------------------
    model.save_pretrained_merged(
        cli_args.output_dir + "/merged",
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"[NegotiateEnv/Unsloth] Done. Model saved to {cli_args.output_dir}/")


if __name__ == "__main__":
    main()
