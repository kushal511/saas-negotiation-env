#!/usr/bin/env python3
"""
NegotiateEnv GRPO Training Script - Direct Environment (No HTTP Server)
Trains Qwen/Qwen2.5-1.5B-Instruct with 4-bit LoRA to negotiate B2B SaaS contracts.

This version uses the environment directly without HTTP/WebSocket, making it simpler
and more reliable for Colab.
"""

import argparse
import json
import re

import torch
from datasets import Dataset

# Try to import GRPO
try:
    from trl import GRPOConfig, GRPOTrainer
    HAS_GRPO = True
except ImportError:
    HAS_GRPO = False
    print("[ERROR] GRPOConfig not found in TRL. Please install TRL >= 0.11.0")
    print("Run: pip install --upgrade 'trl>=0.11.0'")
    import sys
    sys.exit(1)

# Import environment directly
from negotiate_env.server.environment import NegotiateEnvironment
from negotiate_env.models import NegotiateAction

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="NegotiateEnv GRPO training (direct)")
parser.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
parser.add_argument("--max-turns", type=int, default=50)
parser.add_argument("--num-episodes", type=int, default=1000)
parser.add_argument("--output-dir", default="negotiate-unsloth-output")
parser.add_argument("--lora-rank", type=int, default=16)
parser.add_argument("--max-seq-length", type=int, default=1024)
parser.add_argument("--difficulty", default="hard", choices=["easy", "medium", "hard"])
cli_args = parser.parse_args()

# ---------------------------------------------------------------------------
# Module-level globals
# ---------------------------------------------------------------------------
model = None
tokenizer = None

# ---------------------------------------------------------------------------
# System prompt
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
# Environment helpers (direct Python calls, no HTTP)
# ---------------------------------------------------------------------------

def obs_to_prompt(obs) -> str:
    """Format observation as an LLM-readable string."""
    cur = obs.current_offer
    hist = obs.conversation_history[-4:]
    turns_left = obs.max_turns - obs.turn_number

    lines = [
        "## Negotiation Context",
        obs.context,
        "",
        "## AE's Current Offer",
        f"  Price:    ${cur.get('price_per_seat', 0):.2f}/seat/month",
        f"  Length:   {cur.get('contract_length', 0):.0f} year(s)",
        f"  Ann. cap: {cur.get('annual_increase_cap', 0):.1f}%",
        "",
        "## Your Budget Limits",
        f"  Max price:  ${obs.your_max_price:.2f}/seat/month",
        f"  Max length: {obs.your_max_length:.0f} year(s)",
        f"  Max cap:    {obs.your_max_cap:.1f}%",
    ]

    if obs.active_constraints:
        lines += ["", "## Active Constraints"]
        lines += [f"  - {c}" for c in obs.active_constraints]

    if hist:
        lines += ["", "## Recent Conversation"]
        lines += [f"  {h}" for h in hist]

    lines += [
        "",
        f"Turn {obs.turn_number} of {obs.max_turns} ({turns_left} remaining)",
        "",
        f'AE: "{obs.ae_message}"',
        "",
        "Respond with valid JSON only:",
        '{"action_type": "counter", "price_per_seat": 0.0, '
        '"contract_length": 1.0, "annual_increase_cap": 5.0, "message": "..."}',
    ]
    return "\n".join(lines)


def parse_to_action(text: str) -> NegotiateAction:
    """Parse LLM output to a NegotiateAction. Never crashes."""
    text = text.strip()
    valid_types = {"offer", "counter", "probe", "accept", "walkaway"}

    try:
        data = json.loads(text)
        if data.get("action_type") in valid_types:
            return NegotiateAction(**data)
    except Exception:
        pass

    m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            if data.get("action_type") in valid_types:
                return NegotiateAction(**data)
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

    return NegotiateAction(
        action_type=action_type,
        price_per_seat=price,
        contract_length=1.0,
        annual_increase_cap=5.0,
        message=text[:200],
    )


# ---------------------------------------------------------------------------
# Reward function - runs a FULL multi-turn episode
# ---------------------------------------------------------------------------

def reward_negotiate(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """
    For each completion produced by GRPO's rollout:
      1. Reset the environment to a fresh episode.
      2. Turn 0: parse the pre-generated completion as the first action.
      3. Turns 1+: call model.generate for every subsequent turn.
      4. Return the final episode reward as the training signal.
    """
    if model is None or tokenizer is None:
        return [0.0] * len(completions)

    rewards = []
    was_training = model.training
    model.eval()

    try:
        for completion in completions:
            try:
                # Create fresh environment for this episode
                env = NegotiateEnvironment(
                    difficulty=cli_args.difficulty,
                    use_hf_dataset=True,
                    enable_instructions=True,
                    enable_workflow=False,
                )
                obs = env.reset()
                
                if obs.done:
                    rewards.append(0.0)
                    continue

                final_reward = 0.0

                # Turn 0: use GRPO's pre-generated completion
                action = parse_to_action(completion)
                obs = env.step(action)
                
                if obs.done:
                    final_reward = float(obs.reward)
                    rewards.append(final_reward)
                    continue

                # Turns 1+: generate fresh completions
                for _turn in range(1, cli_args.max_turns):
                    if obs.done:
                        final_reward = float(obs.reward)
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
                    obs = env.step(action)

                    if obs.done:
                        final_reward = float(obs.reward)

                rewards.append(final_reward)

            except Exception as e:
                print(f"[warn] Episode failed: {e}")
                rewards.append(0.0)

    finally:
        if was_training:
            model.train()

    return rewards


# ---------------------------------------------------------------------------
# Build dataset
# ---------------------------------------------------------------------------

def build_dataset(n: int, tokenizer) -> Dataset:
    """Build dataset by resetting environment n times."""
    prompts = []
    env = NegotiateEnvironment(
        difficulty=cli_args.difficulty,
        use_hf_dataset=True,
        enable_instructions=True,
        enable_workflow=False,
    )
    
    for i in range(n):
        if i % 100 == 0:
            print(f"  Building dataset: {i}/{n}")
        
        try:
            obs = env.reset()
            user_content = obs_to_prompt(obs)
        except Exception as e:
            print(f"[warn] Dataset build failed for episode {i}: {e}")
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

    print(f"[NegotiateEnv/Direct] Using difficulty={cli_args.difficulty} (max_turns={cli_args.max_turns})")
    
    # Load model with Unsloth
    from unsloth import FastLanguageModel

    print(f"[NegotiateEnv/Direct] Loading {cli_args.model_id} with 4-bit quantisation...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cli_args.model_id,
        max_seq_length=cli_args.max_seq_length,
        load_in_4bit=True,
        fast_inference=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Attach LoRA
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

    # Patch for TRL compatibility
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    if hasattr(model, "base_model") and not hasattr(model.base_model, "warnings_issued"):
        model.base_model.warnings_issued = {}
    if hasattr(model, "base_model") and hasattr(model.base_model, "model") and not hasattr(model.base_model.model, "warnings_issued"):
        model.base_model.model.warnings_issued = {}

    # Build dataset
    print(f"[NegotiateEnv/Direct] Building dataset ({cli_args.num_episodes} episodes)...")
    dataset = build_dataset(cli_args.num_episodes, tokenizer)

    # GRPO config
    grpo_config = GRPOConfig(
        output_dir=cli_args.output_dir,
        use_vllm=False,
        num_train_epochs=1,
        num_generations=4,
        max_completion_length=256,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        logging_steps=5,
        save_steps=50,
        bf16=True,
        report_to="none",
    )

    # Train
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_negotiate],
        train_dataset=dataset,
        args=grpo_config,
    )

    print("[NegotiateEnv/Direct] Starting GRPO training...")
    trainer.train()

    # Save merged model
    model.save_pretrained_merged(
        cli_args.output_dir + "/merged",
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"[NegotiateEnv/Direct] Done. Model saved to {cli_args.output_dir}/")


if __name__ == "__main__":
    main()
