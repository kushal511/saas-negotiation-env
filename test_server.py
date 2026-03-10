#!/usr/bin/env python3
"""Test script to debug server issues."""

import requests
import json

ENV_URL = "http://localhost:7860"

print("Testing environment server...")
print("=" * 60)

# Test 1: Reset
print("\n1. Testing /reset with difficulty=hard...")
try:
    response = requests.post(f"{ENV_URL}/reset", json={"difficulty": "hard"}, timeout=10)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        session_id = data.get("session_id")
        obs = data.get("observation", {})
        print(f"Session ID: {session_id}")
        print(f"Max turns: {obs.get('max_turns')}")
        print(f"Turn number: {obs.get('turn_number')}")
        print("✓ Reset successful!")
        
        # Test 2: Step with a simple action
        print("\n2. Testing /step with counter action...")
        action = {
            "action_type": "counter",
            "price_per_seat": 85.0,
            "contract_length": 1.0,
            "annual_increase_cap": 5.0,
            "message": "Can we do $85/seat for 1 year?"
        }
        
        try:
            step_response = requests.post(
                f"{ENV_URL}/step",
                json={"action": action, "session_id": session_id},
                timeout=10
            )
            print(f"Status: {step_response.status_code}")
            
            if step_response.status_code == 200:
                step_data = step_response.json()
                step_obs = step_data.get("observation", {})
                print(f"Turn number: {step_obs.get('turn_number')}")
                print(f"Done: {step_obs.get('done')}")
                print(f"Reward: {step_obs.get('reward')}")
                print("✓ Step successful!")
            else:
                print(f"✗ Step failed!")
                print(f"Response: {step_response.text}")
                
        except Exception as e:
            print(f"✗ Step error: {e}")
            
    else:
        print(f"✗ Reset failed!")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"✗ Reset error: {e}")

print("\n" + "=" * 60)
print("\nIf you see errors above, check server logs:")
print("  !tail -50 /tmp/server.log")
