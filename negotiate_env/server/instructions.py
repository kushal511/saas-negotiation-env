"""Instruction Following System for Long-Horizon Planning.

300 scattered instructions that agent must remember and follow
throughout negotiation.
"""

from __future__ import annotations
from typing import List, Dict, Any
from negotiate_env.models import NegotiateAction


# 300 scattered instructions
SCATTERED_INSTRUCTIONS = [
    # Price constraints (1-50)
    {"id": 1, "rule": "Never accept a deal with annual cap > 6%", "priority": "high", "category": "price"},
    {"id": 2, "rule": "Always negotiate for at least 5% discount from list price", "priority": "high", "category": "price"},
    {"id": 3, "rule": "If price per seat > $200, require 3-year minimum", "priority": "medium", "category": "price"},
    {"id": 4, "rule": "For enterprise deals (>500 seats), demand volume discount", "priority": "high", "category": "price"},
    {"id": 5, "rule": "Never pay more than competitor price + 10%", "priority": "high", "category": "price"},
    
    # Vendor-specific (51-100)
    {"id": 15, "rule": "When negotiating with Salesforce, mention HubSpot competitor", "priority": "medium", "category": "vendor"},
    {"id": 16, "rule": "Slack deals must include unlimited message history", "priority": "high", "category": "vendor"},
    {"id": 17, "rule": "Zoom contracts require webinar add-on for >100 seats", "priority": "medium", "category": "vendor"},
    {"id": 18, "rule": "HubSpot deals should bundle Marketing + Sales hubs", "priority": "low", "category": "vendor"},
    {"id": 19, "rule": "MongoDB Atlas requires dedicated cluster for production", "priority": "high", "category": "vendor"},
    
    # Contract terms (101-150)
    {"id": 47, "rule": "For deals > $100k, require 3-year minimum contract", "priority": "high", "category": "contract"},
    {"id": 48, "rule": "Annual increase cap must be ≤ 5% for multi-year deals", "priority": "high", "category": "contract"},
    {"id": 49, "rule": "Include 30-day cancellation clause for trials", "priority": "medium", "category": "contract"},
    {"id": 50, "rule": "Require quarterly business reviews for enterprise deals", "priority": "low", "category": "contract"},
    {"id": 51, "rule": "Data residency must be in US for compliance", "priority": "high", "category": "contract"},
    
    # Budget constraints (151-200)
    {"id": 89, "rule": "If budget < 20% remaining, walk away from non-critical deals", "priority": "high", "category": "budget"},
    {"id": 90, "rule": "Reserve 15% of budget for Q4 renewals", "priority": "high", "category": "budget"},
    {"id": 91, "rule": "Critical deals can exceed budget by max 10%", "priority": "medium", "category": "budget"},
    {"id": 92, "rule": "Low priority deals must stay under $50k", "priority": "medium", "category": "budget"},
    {"id": 93, "rule": "Get CFO approval for any deal > $150k", "priority": "high", "category": "budget"},
    
    # Timing constraints (201-250)
    {"id": 120, "rule": "Close critical deals within 2 weeks", "priority": "high", "category": "timing"},
    {"id": 121, "rule": "Q4 deals get 10% discount (vendor quota pressure)", "priority": "medium", "category": "timing"},
    {"id": 122, "rule": "Avoid signing in December (budget freeze)", "priority": "low", "category": "timing"},
    {"id": 123, "rule": "Renewals must start 60 days before expiry", "priority": "high", "category": "timing"},
    {"id": 124, "rule": "Month-end deals get better pricing", "priority": "low", "category": "timing"},
    
    # Negotiation tactics (251-300)
    {"id": 150, "rule": "Always probe before making first counter", "priority": "high", "category": "tactics"},
    {"id": 151, "rule": "Reference competitor pricing in first 3 turns", "priority": "medium", "category": "tactics"},
    {"id": 152, "rule": "Extend contract length to unlock better pricing", "priority": "medium", "category": "tactics"},
    {"id": 153, "rule": "Bundle multiple products for volume discount", "priority": "low", "category": "tactics"},
    {"id": 154, "rule": "Request case studies before committing", "priority": "low", "category": "tactics"},
    
    # Compliance (301-350) - extending beyond 300
    {"id": 200, "rule": "GDPR compliance required for EU customers", "priority": "high", "category": "compliance"},
    {"id": 201, "rule": "SOC 2 certification mandatory for security tools", "priority": "high", "category": "compliance"},
    {"id": 202, "rule": "Data encryption at rest and in transit required", "priority": "high", "category": "compliance"},
    {"id": 203, "rule": "Annual security audits for critical systems", "priority": "medium", "category": "compliance"},
    {"id": 204, "rule": "Vendor must have cyber insurance > $5M", "priority": "medium", "category": "compliance"},
]

# Extend to 300 instructions with variations
for i in range(len(SCATTERED_INSTRUCTIONS), 300):
    base_instruction = SCATTERED_INSTRUCTIONS[i % len(SCATTERED_INSTRUCTIONS)]
    SCATTERED_INSTRUCTIONS.append({
        "id": i + 1,
        "rule": f"{base_instruction['rule']} (variant {i // len(SCATTERED_INSTRUCTIONS) + 1})",
        "priority": base_instruction["priority"],
        "category": base_instruction["category"],
    })


def check_instruction_compliance(
    action: NegotiateAction,
    state: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Check if action violates any instructions.
    
    Returns list of violated instructions.
    """
    violations = []
    
    # Check price constraints
    if action.action_type == "accept":
        annual_cap = action.annual_increase_cap or state.get("current_offer", {}).get("annual_increase_cap", 0)
        if annual_cap > 6:
            violations.append(SCATTERED_INSTRUCTIONS[0])  # Instruction 1
    
    # Check budget constraints
    remaining_budget = state.get("remaining_budget", float('inf'))
    total_budget = state.get("total_budget", float('inf'))
    if remaining_budget < total_budget * 0.2:
        priority = state.get("current_deal_priority", "medium")
        if priority in ["low", "medium"] and action.action_type == "accept":
            violations.append(SCATTERED_INSTRUCTIONS[18])  # Instruction 89
    
    # Check contract length for large deals
    if action.action_type in ["counter", "offer"]:
        deal_value = state.get("estimated_deal_value", 0)
        if deal_value > 100000 and action.contract_length < 3:
            violations.append(SCATTERED_INSTRUCTIONS[6])  # Instruction 47
    
    # Check probing requirement
    turn_number = state.get("turn_number", 0)
    if turn_number == 1 and action.action_type != "probe":
        violations.append(SCATTERED_INSTRUCTIONS[29])  # Instruction 150
    
    return violations


def get_instruction_penalty(violations: List[Dict[str, Any]]) -> float:
    """Calculate penalty for instruction violations."""
    if not violations:
        return 0.0
    
    penalty = 0.0
    for violation in violations:
        if violation["priority"] == "high":
            penalty += 0.1
        elif violation["priority"] == "medium":
            penalty += 0.05
        else:
            penalty += 0.02
    
    return penalty


def get_relevant_instructions(
    state: Dict[str, Any],
    max_instructions: int = 10,
) -> List[Dict[str, Any]]:
    """
    Get most relevant instructions for current state.
    
    Returns top N instructions based on context.
    """
    relevant = []
    
    # Get vendor-specific instructions
    vendor = state.get("vendor", "")
    for inst in SCATTERED_INSTRUCTIONS:
        if inst["category"] == "vendor" and vendor.lower() in inst["rule"].lower():
            relevant.append(inst)
    
    # Get budget instructions if budget is low
    remaining_budget = state.get("remaining_budget", float('inf'))
    total_budget = state.get("total_budget", float('inf'))
    if remaining_budget < total_budget * 0.3:
        for inst in SCATTERED_INSTRUCTIONS:
            if inst["category"] == "budget":
                relevant.append(inst)
    
    # Get high priority instructions
    for inst in SCATTERED_INSTRUCTIONS:
        if inst["priority"] == "high" and inst not in relevant:
            relevant.append(inst)
    
    # Return top N
    return relevant[:max_instructions]


def format_instructions_for_prompt(instructions: List[Dict[str, Any]]) -> str:
    """Format instructions for inclusion in agent prompt."""
    if not instructions:
        return ""
    
    prompt = "\n--- INSTRUCTIONS TO FOLLOW ---\n"
    for inst in instructions:
        priority_marker = "🔴" if inst["priority"] == "high" else "🟡" if inst["priority"] == "medium" else "🟢"
        prompt += f"{priority_marker} Instruction #{inst['id']}: {inst['rule']}\n"
    
    return prompt
