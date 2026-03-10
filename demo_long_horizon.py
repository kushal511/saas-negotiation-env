#!/usr/bin/env python3
"""
Demo for Long-Horizon Planning Features

Shows:
- Extended episodes (50 turns)
- Instruction following (300+ rules)
- Sales workflow (8 stages)
- Multi-deal negotiation (5 contracts)
"""

import requests
import json

BASE_URL = "https://kushaladhyaru-negotiate-env.hf.space"

def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def demo_extended_episodes():
    """Demo 1: Extended episode length (50 turns)"""
    print_header("DEMO 1: Extended Episodes (50 Turns)")
    
    print("\n🎯 Testing long-horizon planning with 50-turn episodes...")
    
    response = requests.post(
        f"{BASE_URL}/reset",
        json={"difficulty": "hard"},  # Hard = 50 turns
        timeout=10
    )
    
    if response.status_code == 200:
        obs = response.json().get("observation", {})
        print(f"✅ Episode started")
        print(f"   Max turns: {obs.get('max_turns', 0)} (was 7-12, now 50!)")
        print(f"   Difficulty: hard")
        print(f"   Scenario: {obs.get('context', '')[:80]}...")
        print(f"\n   This allows for:")
        print(f"   - Multi-stage negotiations")
        print(f"   - Complex strategy development")
        print(f"   - Recovery from early mistakes")
    else:
        print(f"❌ Failed: {response.status_code}")

def demo_instruction_following():
    """Demo 2: Instruction following system"""
    print_header("DEMO 2: Instruction Following (300+ Rules)")
    
    print("\n📋 Agent must follow 300+ scattered instructions...")
    print("\nExample instructions:")
    print("   🔴 #1: Never accept a deal with annual cap > 6%")
    print("   🟡 #15: When negotiating with Salesforce, mention HubSpot")
    print("   🔴 #47: For deals > $100k, require 3-year minimum")
    print("   🔴 #89: If budget < 20%, walk away from non-critical deals")
    
    print("\n✅ Instructions are:")
    print("   - Scattered across 300+ rules")
    print("   - Prioritized (high/medium/low)")
    print("   - Context-dependent")
    print("   - Enforced with penalties")
    
    print("\n   Agent must remember and follow throughout negotiation!")

def demo_sales_workflow():
    """Demo 3: 8-stage sales workflow"""
    print_header("DEMO 3: Full Sales Workflow (8 Stages)")
    
    print("\n🔄 Complete B2B sales workflow (50 turns):")
    
    stages = [
        ("1. Lead Qualification", "Turn 1-5", "Assess fit and budget"),
        ("2. Discovery", "Turn 6-10", "Understand needs and pain points"),
        ("3. Proposal", "Turn 11-15", "Present solution and pricing"),
        ("4. Negotiation", "Turn 16-30", "Negotiate terms and contract"),
        ("5. Contract Review", "Turn 31-35", "Legal and compliance review"),
        ("6. Approval", "Turn 36-40", "Stakeholder sign-off"),
        ("7. Closing", "Turn 41-45", "Finalize and sign contract"),
        ("8. Onboarding Handoff", "Turn 46-50", "Transition to CS team"),
    ]
    
    for stage, turns, desc in stages:
        print(f"\n   {stage}")
        print(f"      {turns}: {desc}")
    
    print("\n✅ Agent must:")
    print("   - Progress through all 8 stages")
    print("   - Complete required actions per stage")
    print("   - Track state across 50 turns")
    print("   - Adapt strategy per stage")

def demo_multi_deal():
    """Demo 4: Multi-deal negotiation"""
    print_header("DEMO 4: Multi-Deal Negotiation (5 Contracts)")
    
    print("\n💼 Negotiate 5 contracts with shared budget...")
    print("\n   Scenario:")
    print("   - Total budget: $500,000")
    print("   - Deals to close: 5")
    print("   - Quarterly quota: 3 deals minimum")
    
    print("\n   Example deals:")
    print("   1. HubSpot CRM - $100k (critical)")
    print("   2. Salesforce Sales Cloud - $150k (high)")
    print("   3. Slack Business+ - $80k (medium)")
    print("   4. Zoom Enterprise - $70k (medium)")
    print("   5. Notion Team - $50k (low)")
    
    print("\n   Dependencies:")
    print("   - Deal 3 (Slack) requires Deal 1 (HubSpot) to close first")
    print("   - Deal 5 (Notion) requires Deal 2 (Salesforce) to close first")
    
    print("\n✅ Agent must:")
    print("   - Track remaining budget across deals")
    print("   - Remember previous deal terms")
    print("   - Respect dependencies")
    print("   - Meet quarterly quota")
    print("   - Optimize total spend")

def demo_state_tracking():
    """Demo 5: Complex state tracking"""
    print_header("DEMO 5: Complex State Tracking")
    
    print("\n🧠 Agent must track complex state:")
    
    state_example = {
        "current_deal": 3,
        "total_deals": 5,
        "remaining_budget": 270000,
        "total_budget": 500000,
        "closed_deals": [
            {"vendor": "HubSpot", "price": 95000, "length": 2},
            {"vendor": "Salesforce", "price": 135000, "length": 3},
        ],
        "deals_closed_this_quarter": 2,
        "quarterly_quota": 3,
        "dependencies": "Slack requires HubSpot (closed ✓)",
        "active_instructions": [
            "#1: Never accept annual cap > 6%",
            "#89: Budget < 20% remaining - walk from non-critical",
        ],
        "workflow_stage": "negotiation",
    }
    
    print("\n   State information:")
    for key, value in state_example.items():
        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
            print(f"   - {key}:")
            for item in value:
                print(f"       {item}")
        elif isinstance(value, list):
            print(f"   - {key}:")
            for item in value:
                print(f"       {item}")
        else:
            print(f"   - {key}: {value}")
    
    print("\n✅ This tests:")
    print("   - Long-term memory")
    print("   - Multi-step reasoning")
    print("   - Context management")
    print("   - Strategic planning")

def demo_recovery():
    """Demo 6: Recovery from mistakes"""
    print_header("DEMO 6: Recovery from Mistakes")
    
    print("\n🔄 Agent can renegotiate bad deals...")
    
    print("\n   Scenario:")
    print("   - Deal 2: Accepted at $180k (over budget!)")
    print("   - Remaining budget: $220k")
    print("   - Still 3 deals to close")
    
    print("\n   Recovery action:")
    action = {
        "action_type": "renegotiate",
        "deal_id": 2,
        "reason": "Budget constraints - need to reduce by $30k",
        "price_per_seat": 125.0,
        "contract_length": 3.0,
        "message": "Can we revisit the terms? Budget got tighter."
    }
    
    print(f"   {json.dumps(action, indent=6)}")
    
    print("\n✅ This enables:")
    print("   - Learning from mistakes")
    print("   - Mid-course corrections")
    print("   - Adaptive strategies")
    print("   - Long-horizon optimization")

def main():
    print_header("NegotiateEnv: Long-Horizon Planning Demo")
    print("\n🚀 Super Long-Horizon Planning Features")
    print("   Team: Kushal Adhyaru and Mayuka Kothuru")
    print("   OpenEnv Hackathon - Statement 2")
    
    # Run demos
    demo_extended_episodes()
    demo_instruction_following()
    demo_sales_workflow()
    demo_multi_deal()
    demo_state_tracking()
    demo_recovery()
    
    # Summary
    print_header("Summary: Long-Horizon Features")
    
    print("\n✅ Implemented:")
    print("   1. Extended episodes: 30-50 turns (vs 7-12)")
    print("   2. Instruction following: 300+ scattered rules")
    print("   3. Sales workflow: 8 stages, 50 turns")
    print("   4. Multi-deal: 5 contracts, shared budget")
    print("   5. State tracking: Complex multi-deal state")
    print("   6. Recovery: Renegotiate bad deals")
    
    print("\n🎯 Statement 2 Alignment:")
    print("   - Super long-horizon planning (50 turns)")
    print("   - Sparse rewards (only at completion)")
    print("   - Complex state tracking (beyond context)")
    print("   - Instruction following (300+ rules)")
    print("   - Recovery mechanisms (renegotiate)")
    
    print("\n🏢 Scale AI Partner Theme:")
    print("   - Complete B2B sales workflow")
    print("   - Professional business setting")
    print("   - Real-world procurement task")
    print("   - 8 stages from lead to close")
    
    print("\n📊 Training Requirements:")
    print("   - Episodes: 1000+ (vs 300)")
    print("   - Max turns: 50 (vs 6)")
    print("   - Context: Multi-deal state")
    print("   - Memory: Track across deals")
    
    print("\n" + "=" * 70)
    print("  Demo Complete! 🎉")
    print("=" * 70)
    print(f"\n🔗 Environment: {BASE_URL}")
    print("🔗 Repository: https://github.com/kushal511/saas-negotiation-env")
    print("\n")

if __name__ == "__main__":
    main()
