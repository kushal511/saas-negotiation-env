"""Multi-Deal Negotiation Environment for Super Long-Horizon Planning.

Agent must negotiate multiple contracts in sequence with shared budget,
dependencies, and complex state tracking requirements.
"""

from __future__ import annotations
from typing import Any, Dict, List
import random

from negotiate_env.models import NegotiateAction, NegotiateObservation
from negotiate_env.server.environment import NegotiateEnvironment


class MultiDealNegotiationEnv:
    """
    Super long-horizon environment requiring negotiation of 5+ contracts
    with shared budget, dependencies, and state tracking.
    
    Features:
    - Multiple deals to negotiate (default: 5)
    - Shared budget across all deals
    - Deal dependencies (some deals require others to close first)
    - Quarterly quotas (must close X deals per quarter)
    - State tracking across deals
    - Sparse rewards (only when ALL deals closed)
    """
    
    def __init__(
        self,
        num_deals: int = 5,
        total_budget: float = 500000,
        difficulty: str = "medium",
    ):
        self.num_deals = num_deals
        self.total_budget = total_budget
        self.difficulty = difficulty
        
        # State
        self.current_deal_idx = 0
        self.deals: List[Dict[str, Any]] = []
        self.closed_deals: List[Dict[str, Any]] = []
        self.remaining_budget = total_budget
        self.current_env: NegotiateEnvironment | None = None
        
        # Quarterly quota
        self.quarterly_quota = max(3, num_deals // 2)
        self.deals_closed_this_quarter = 0
        
    def reset(self, **kwargs) -> NegotiateObservation:
        """Reset and generate multiple deals to negotiate."""
        self.current_deal_idx = 0
        self.closed_deals = []
        self.remaining_budget = self.total_budget
        self.deals_closed_this_quarter = 0
        
        # Generate multiple deals
        self.deals = self._generate_deals()
        
        # Start first deal
        self.current_env = NegotiateEnvironment(difficulty=self.difficulty)
        obs = self.current_env.reset(
            scenario_id=self.deals[0].get("scenario_id"),
            **kwargs
        )
        
        # Enhance observation with multi-deal context
        return self._enhance_observation(obs)
    
    def step(self, action: NegotiateAction) -> NegotiateObservation:
        """Execute action in current deal."""
        if self.current_env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        # Execute action in current deal
        obs = self.current_env.step(action)
        
        # Check if current deal is done
        if obs.done:
            self._process_deal_completion(obs)
            
            # Move to next deal if available
            if self.current_deal_idx < len(self.deals) - 1:
                self.current_deal_idx += 1
                obs = self._start_next_deal()
            else:
                # All deals complete - calculate final reward
                obs = self._finalize_all_deals(obs)
        
        # Enhance observation with multi-deal context
        return self._enhance_observation(obs)
    
    def _generate_deals(self) -> List[Dict[str, Any]]:
        """Generate multiple deals with dependencies."""
        vendors = [
            "HubSpot", "Salesforce", "Slack", "Zoom", "Notion",
            "Figma", "Zendesk", "Intercom", "MongoDB", "GitLab"
        ]
        
        deals = []
        for i in range(self.num_deals):
            vendor = random.choice(vendors)
            deal = {
                "deal_id": i + 1,
                "vendor": vendor,
                "scenario_id": None,  # Will use random scenario
                "priority": random.choice(["critical", "high", "medium", "low"]),
                "dependencies": [],
                "estimated_value": random.randint(50000, 150000),
            }
            
            # Add dependencies (some deals require others)
            if i > 0 and random.random() < 0.3:
                deal["dependencies"].append(i)  # Depends on previous deal
            
            deals.append(deal)
        
        return deals
    
    def _process_deal_completion(self, obs: NegotiateObservation) -> Dict[str, Any]:
        """Process completion of current deal."""
        current_deal = self.deals[self.current_deal_idx]
        
        if obs.reward > 0:
            # Deal closed successfully
            deal_info = {
                "deal_id": current_deal["deal_id"],
                "vendor": current_deal["vendor"],
                "final_price": obs.current_offer.get("price_per_seat", 0),
                "contract_length": obs.current_offer.get("contract_length", 0),
                "reward": obs.reward,
            }
            
            # Calculate total contract value (estimate 100 seats)
            seats = 100
            monthly_cost = deal_info["final_price"] * seats
            total_cost = monthly_cost * deal_info["contract_length"] * 12
            deal_info["total_cost"] = total_cost
            
            # Update budget
            self.remaining_budget -= total_cost
            self.closed_deals.append(deal_info)
            self.deals_closed_this_quarter += 1
            
            return deal_info
        else:
            return {
                "deal_id": current_deal["deal_id"],
                "vendor": current_deal["vendor"],
                "status": "failed",
                "reward": obs.reward,
            }
    
    def _start_next_deal(self) -> NegotiateObservation:
        """Start negotiation for next deal."""
        next_deal = self.deals[self.current_deal_idx]
        
        # Check dependencies
        for dep_idx in next_deal.get("dependencies", []):
            if dep_idx not in [d["deal_id"] for d in self.closed_deals]:
                # Dependency not met - skip
                if self.current_deal_idx < len(self.deals) - 1:
                    self.current_deal_idx += 1
                    return self._start_next_deal()
        
        # Start next deal
        self.current_env = NegotiateEnvironment(difficulty=self.difficulty)
        obs = self.current_env.reset(scenario_id=next_deal.get("scenario_id"))
        
        return obs
    
    def _finalize_all_deals(self, last_obs: NegotiateObservation) -> NegotiateObservation:
        """Calculate final reward when all deals are complete."""
        total_reward = sum(d.get("reward", 0) for d in self.closed_deals)
        
        # Bonus for meeting quota
        if self.deals_closed_this_quarter >= self.quarterly_quota:
            total_reward += 0.5
        
        # Penalty for budget overrun
        if self.remaining_budget < 0:
            total_reward -= 0.3
        
        # Bonus for staying under budget
        if self.remaining_budget > self.total_budget * 0.2:
            total_reward += 0.2
        
        last_obs.reward = total_reward
        last_obs.done = True
        
        return last_obs
    
    def _enhance_observation(self, obs: NegotiateObservation) -> NegotiateObservation:
        """Add multi-deal context to observation."""
        current_deal = self.deals[self.current_deal_idx]
        
        multi_deal_context = f"""

--- MULTI-DEAL CONTEXT ---
Deal {self.current_deal_idx + 1} of {self.num_deals}
Vendor: {current_deal['vendor']}
Priority: {current_deal['priority']}
Remaining Budget: ${self.remaining_budget:,.0f} of ${self.total_budget:,.0f}
Deals Closed: {len(self.closed_deals)}
Quarterly Quota: {self.deals_closed_this_quarter}/{self.quarterly_quota}

Previous Deals:
"""
        
        for deal in self.closed_deals[-3:]:
            multi_deal_context += f"  - {deal['vendor']}: ${deal['final_price']:.0f}/seat, {deal['contract_length']:.0f}y\n"
        
        if current_deal.get("dependencies"):
            multi_deal_context += f"\nDependencies: Requires deal #{current_deal['dependencies'][0]} to be closed\n"
        
        obs.context = obs.context + multi_deal_context
        
        return obs
    
    @property
    def state(self) -> Dict[str, Any]:
        """Get current multi-deal state."""
        return {
            "current_deal": self.current_deal_idx + 1,
            "total_deals": self.num_deals,
            "remaining_budget": self.remaining_budget,
            "total_budget": self.total_budget,
            "closed_deals": self.closed_deals,
            "deals_closed_this_quarter": self.deals_closed_this_quarter,
            "quarterly_quota": self.quarterly_quota,
        }
