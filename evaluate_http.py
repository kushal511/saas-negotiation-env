#!/usr/bin/env python3
"""HTTP-based evaluation script for NegotiateEnv (works with HuggingFace Spaces).

Usage:
    python evaluate_http.py --agent rule --episodes 100 --env-url https://kushaladhyaru-negotiate-env.hf.space
"""

import argparse
import random
import statistics
from collections import Counter

import requests
from negotiate_env.models import NegotiateAction, NegotiateObservation


def random_policy(obs: NegotiateObservation, turn: int) -> NegotiateAction:
    """Random baseline: pick random valid action."""
    action_type = random.choice(["counter", "probe", "accept", "walkaway"])
    
    if action_type == "counter":
        return NegotiateAction(
            action_type="counter",
            price_per_seat=random.uniform(obs.your_max_price * 0.7, obs.your_max_price),
            contract_length=random.uniform(1.0, obs.your_max_length),
            annual_increase_cap=random.uniform(3.0, obs.your_max_cap),
            message="Here's my counter-offer."
        )
    elif action_type == "probe":
        return NegotiateAction(action_type="probe", message="Can you tell me more about your pricing?")
    elif action_type == "accept":
        return NegotiateAction(action_type="accept", message="I accept your offer.")
    else:
        return NegotiateAction(action_type="walkaway", message="I'll pass on this.")


def rule_policy(obs: NegotiateObservation, turn: int) -> NegotiateAction:
    """Rule-based baseline: simple heuristic negotiation."""
    current_price = obs.current_offer.get("price_per_seat", obs.your_max_price)
    current_length = obs.current_offer.get("contract_length", 2.0)
    current_cap = obs.current_offer.get("annual_increase_cap", 7.0)
    
    # Accept if within budget
    if (current_price <= obs.your_max_price and 
        current_length <= obs.your_max_length and 
        current_cap <= obs.your_max_cap):
        return NegotiateAction(action_type="accept", message="This works for us.")
    
    # Probe first
    if turn == 1:
        return NegotiateAction(action_type="probe", message="What's your best price for a 2-year contract?")
    
    # Counter-offer
    target_price = obs.your_max_price * 0.85
    target_length = min(2.0, obs.your_max_length)
    target_cap = min(5.0, obs.your_max_cap)
    
    return NegotiateAction(
        action_type="counter",
        price_per_seat=target_price,
        contract_length=target_length,
        annual_increase_cap=target_cap,
        message=f"Can we do ${target_price:.0f}/seat for {target_length:.0f} years?"
    )


def run_episode_http(env_url: str, policy, max_turns: int, difficulty: str = "hard"):
    """Run one episode using HTTP endpoints."""
    # Reset
    try:
        response = requests.post(
            f"{env_url}/reset",
            json={"difficulty": difficulty},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        obs_data = data.get("observation", data)
        obs = NegotiateObservation(**obs_data)
    except Exception as e:
        print(f"Reset failed: {e}")
        return 0.0, 0, False, 0.0, []
    
    actions_taken = []
    turn = 0
    
    while not obs.done and turn < max_turns:
        turn += 1
        
        # Get action from policy
        action = policy(obs, turn)
        actions_taken.append(action.action_type)
        
        # Step
        try:
            response = requests.post(
                f"{env_url}/step",
                json={"action": action.model_dump()},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            obs_data = data.get("observation", data)
            obs = NegotiateObservation(**obs_data)
        except Exception as e:
            print(f"Step failed: {e}")
            break
    
    # Extract results
    reward = obs.reward
    success = reward > 0.0
    final_price = obs.current_offer.get("price_per_seat", 0.0) if success else 0.0
    
    return reward, turn, success, final_price, actions_taken


def main():
    parser = argparse.ArgumentParser(description="Evaluate NegotiateEnv agent (HTTP)")
    parser.add_argument("--agent", choices=["random", "rule"], default="rule")
    parser.add_argument("--env-url", default="http://127.0.0.1:7860")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-turns", type=int, default=50)
    parser.add_argument("--difficulty", default="hard", choices=["easy", "medium", "hard"])
    args = parser.parse_args()
    
    # Select policy
    if args.agent == "random":
        policy = random_policy
    else:
        policy = rule_policy
    
    print(f"Evaluating '{args.agent}' agent over {args.episodes} episodes (difficulty={args.difficulty})...")
    
    rewards = []
    turns_list = []
    successes = 0
    prices = []
    all_actions = Counter()
    
    for ep in range(args.episodes):
        reward, turns, success, price, actions = run_episode_http(
            args.env_url, policy, args.max_turns, args.difficulty
        )
        
        rewards.append(reward)
        turns_list.append(turns)
        if success:
            successes += 1
            prices.append(price)
        
        for action in actions:
            all_actions[action] += 1
        
        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep + 1}/{args.episodes} complete...")
    
    # Report
    print("\n" + "=" * 60)
    print(f"  Agent:           {args.agent}")
    print(f"  Episodes:        {args.episodes}")
    print(f"  Difficulty:      {args.difficulty}")
    print(f"  Mean reward:     {statistics.mean(rewards):.4f}")
    print(f"  Median reward:   {statistics.median(rewards):.4f}")
    print(f"  Success rate:    {successes / args.episodes * 100:.1f}%")
    
    if prices:
        print(f"  Avg deal price:  ${statistics.mean(prices):.2f}")
    
    print(f"  Avg turns:       {statistics.mean(turns_list):.1f}")
    print()
    print("  Strategy distribution:")
    total_actions = sum(all_actions.values())
    for action, count in all_actions.most_common():
        pct = count / total_actions * 100
        print(f"    {action:12s} {count:4d}  ({pct:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
