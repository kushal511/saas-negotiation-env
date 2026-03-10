"""NegotiateEnvironment: OpenEnv Environment for B2B SaaS contract negotiation."""

import random
from collections import Counter
from typing import Any, Optional

from openenv.core.env_server import Environment

from negotiate_env.models import NegotiateAction, NegotiateObservation
from negotiate_env.scenarios import SCENARIOS
from negotiate_env.server.difficulty import get_difficulty
from negotiate_env.server.opponent import AEOpponent

# Long-horizon planning imports
from negotiate_env.server.instructions import (
    check_instruction_compliance,
    get_instruction_penalty,
    get_relevant_instructions,
    format_instructions_for_prompt,
)
from negotiate_env.server.sales_workflow import SalesWorkflowManager

# Multi-app imports for Track 3.1
try:
    from negotiate_env.apps.crm import SalesforceCRM
    from negotiate_env.apps.approval import ApprovalWorkflow
    MULTI_APP_ENABLED = True
except ImportError:
    MULTI_APP_ENABLED = False


class NegotiateEnvironment(Environment):
    """RL environment: agent (procurement manager) negotiates with rule-based AE.

    Args:
        difficulty: "easy" | "medium" (default) | "hard"
        use_hf_dataset: If True, load scenarios from HuggingFace on first reset.
        enable_instructions: If True, enforce 300+ scattered instructions.
        enable_workflow: If True, track 8-stage sales workflow.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        difficulty: str = "medium",
        use_hf_dataset: bool = False,
        enable_instructions: bool = True,
        enable_workflow: bool = False,
    ):
        super().__init__()
        self._difficulty_name = difficulty
        self._difficulty = get_difficulty(difficulty)
        self._use_hf_dataset = use_hf_dataset
        self._enable_instructions = enable_instructions
        self._enable_workflow = enable_workflow
        self._scenarios: list[dict[str, Any]] = []
        self._action_counter: Counter = Counter()  # strategy discovery metrics
        self._step_count: int = 0
        self._scenario: dict[str, Any] = {}
        self._opponent: Optional[AEOpponent] = None
        self._vendor_floor_price: float = 0.0
        self._vendor_list_price: float = 0.0
        self._vendor_preferred_length: float = 0.0
        self._vendor_max_cap: float = 10.0
        self._vendor_min_cap: float = 3.0
        self._agent_max_price: float = 0.0
        self._agent_max_length: float = 0.0
        self._agent_max_cap: float = 0.0
        
        # Long-horizon planning components
        self._workflow_manager: Optional[SalesWorkflowManager] = None
        if self._enable_workflow:
            self._workflow_manager = SalesWorkflowManager()
        self._max_turns: int = self._difficulty.max_turns
        self._enable_drift: bool = self._difficulty.enable_drift
        self._conversation_history: list[str] = []
        self._current_offer: dict[str, Any] = {}
        self._active_constraints: list[str] = []
        self._drift_injected: bool = False
        self._turn_penalties: float = 0.0
        self._shaping_reward: float = 0.0
        self._last_action_type: str = ""

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> NegotiateObservation:
        config = kwargs
        if seed is not None:
            random.seed(seed)

        # Lazy-load scenarios (HF or built-in)
        if not self._scenarios:
            if self._use_hf_dataset:
                from negotiate_env.dataset_loader import load_scenarios
                self._scenarios = load_scenarios(hf=True)
            else:
                self._scenarios = list(SCENARIOS)

        scenario_id = config.get("scenario_id")
        if scenario_id:
            self._scenario = next(
                (s for s in self._scenarios if s["id"] == scenario_id),
                random.choice(self._scenarios),
            )
        else:
            self._scenario = random.choice(self._scenarios)

        # Allow per-reset overrides; fall back to difficulty defaults
        self._max_turns = int(config.get("max_turns", self._difficulty.max_turns))
        self._enable_drift = config.get("enable_drift", self._difficulty.enable_drift)

        # Hidden internal state — NEVER exposed in observations
        self._vendor_floor_price = (
            self._scenario["vendor_floor_price"] * self._difficulty.floor_multiplier
        )
        self._vendor_list_price = self._scenario["vendor_list_price"]
        self._vendor_preferred_length = self._scenario["vendor_preferred_length"]
        self._vendor_max_cap = self._scenario.get("vendor_max_cap", 10.0)
        self._vendor_min_cap = self._scenario.get("vendor_min_cap", 3.0)

        self._agent_max_price = self._scenario["agent_max_price"]
        self._agent_max_length = self._scenario["agent_max_length"]
        self._agent_max_cap = self._scenario["agent_max_cap"]

        self._opponent = AEOpponent(self._scenario, self._scenario["opponent_strategy"])
        self._current_offer = {
            "price_per_seat": self._vendor_list_price,
            "contract_length": self._vendor_preferred_length,
            "annual_increase_cap": self._vendor_max_cap,
        }
        self._conversation_history = []
        self._active_constraints = []
        self._drift_injected = False
        self._turn_penalties = 0.0
        self._shaping_reward = 0.0
        self._last_action_type = ""
        self._step_count = 0

        opening_msg = self._scenario["vendor_opening_message"]
        self._conversation_history.append(f"AE: {opening_msg}")

        return NegotiateObservation(
            context=self._scenario["context"],
            your_max_price=self._agent_max_price,
            your_max_length=self._agent_max_length,
            your_max_cap=self._agent_max_cap,
            ae_message=opening_msg,
            conversation_history=list(self._conversation_history),
            turn_number=0,
            max_turns=self._max_turns,
            active_constraints=[],
            current_offer=dict(self._current_offer),
            reward=0.0,
            done=False,
        )

    def step(
        self,
        action: NegotiateAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> NegotiateObservation:
        self._step_count += 1
        turn = self._step_count
        # Proportional turn penalty: scales with episode length (total max = 0.10)
        penalty_per_turn = 0.10 / self._max_turns
        self._turn_penalties += penalty_per_turn

        # Track strategy discovery metrics
        self._action_counter[action.action_type] += 1
        
        # Long-horizon: Check instruction compliance
        instruction_violations = []
        if self._enable_instructions:
            state = {
                "current_offer": self._current_offer,
                "remaining_budget": getattr(self, "_remaining_budget", None),
                "total_budget": getattr(self, "_total_budget", None),
                "turn_number": turn,
                "vendor": self._scenario.get("vendor", ""),
                "estimated_deal_value": self._scenario.get("budget", 0),
                "current_deal_priority": getattr(self, "_current_deal_priority", "medium"),
            }
            instruction_violations = check_instruction_compliance(action, state)
            if instruction_violations:
                penalty = get_instruction_penalty(instruction_violations)
                self._turn_penalties += penalty
        
        # Long-horizon: Update sales workflow stage
        if self._enable_workflow and self._workflow_manager:
            self._workflow_manager.check_stage_transition(turn, action.action_type)

        # Inject drift exactly once at drift_turn
        if (
            self._enable_drift
            and not self._drift_injected
            and turn >= self._scenario.get("drift_turn", 999)
        ):
            self._active_constraints.append(self._scenario["drift_event"])
            self._drift_injected = True

        # Record agent's move in history
        if action.action_type in ("offer", "counter") and (
            action.price_per_seat or action.contract_length or action.annual_increase_cap
        ):
            self._conversation_history.append(
                f"Agent: [Offer] ${action.price_per_seat:.0f}/seat, "
                f"{action.contract_length:.0f}y, {action.annual_increase_cap:.1f}% cap"
            )
        elif action.message:
            self._conversation_history.append(f"Agent: {action.message}")

        # Task 22 — Repeat action penalty
        if action.action_type == self._last_action_type and self._last_action_type != "":
            self._turn_penalties += 0.03
        self._last_action_type = action.action_type

        # Task 23 — Lowball penalty (price < competitor × 0.75)
        competitor_price = self._scenario.get("competitor_price", 0.0)
        if (
            competitor_price > 0
            and action.price_per_seat > 0
            and action.price_per_seat < competitor_price * 0.75
        ):
            self._turn_penalties += 0.08

        # Task 25 — Walk-away reward: smart (+0.1) if standing offer > 110% of budget, else bad (-0.2)
        if action.action_type == "walkaway":
            budget = self._scenario.get("budget", float("inf"))
            seat_count = self._scenario.get("seat_count", 1)
            standing_price = self._current_offer.get("price_per_seat", self._vendor_list_price)
            standing_length = self._current_offer.get("contract_length", 1.0)
            total_contract_value = standing_price * seat_count * standing_length * 12
            walkaway_reward = 0.1 if total_contract_value > budget * 1.1 else -0.2
            return self._obs(
                ae_message="I'm sorry we couldn't find common ground. Feel free to reach out if things change.",
                done=True,
                reward=walkaway_reward,
            )

        if action.action_type == "accept":
            if self._current_offer.get("price_per_seat", 0) >= self._vendor_floor_price:
                reward = self._compute_reward(self._current_offer)
                return self._obs(
                    ae_message="Perfect. We're aligned. I'll get the contract over to you.",
                    done=True,
                    reward=reward,
                )
            # Invalid accept — AE counter-offers because their standing offer is below floor threshold
            ae_msg, self._current_offer = self._opponent.respond(
                action, turn, self._conversation_history, self._current_offer
            )
            self._conversation_history.append(f"AE: {ae_msg}")
            return self._obs(ae_message=ae_msg, done=False, reward=0.0)

        if action.action_type == "probe":
            ae_msg, _ = self._opponent._probe_response(self._current_offer)
            self._conversation_history.append(f"AE: {ae_msg}")
            return self._obs(ae_message=ae_msg, done=False, reward=0.0)

        # "offer" or "counter" — check if agent's numbers clear the floor
        agent_price = action.price_per_seat if action.price_per_seat > 0 else self._current_offer.get("price_per_seat", 0)
        agent_length = action.contract_length if action.contract_length > 0 else self._current_offer.get("contract_length", 2.0)
        agent_cap = action.annual_increase_cap if action.annual_increase_cap > 0 else self._current_offer.get("annual_increase_cap", 7.0)

        # AE accepts if: price >= floor, length >= 1yr, cap >= AE's minimum cap
        if (
            agent_price >= self._vendor_floor_price
            and agent_length >= 1.0
            and agent_cap >= self._vendor_min_cap
        ):
            deal = {
                "price_per_seat": agent_price,
                "contract_length": max(1.0, min(3.0, agent_length)),
                "annual_increase_cap": agent_cap,
            }
            self._current_offer = deal
            reward = self._compute_reward(deal)
            return self._obs(
                ae_message="We have a deal. I'll send over the paperwork at those terms.",
                done=True,
                reward=reward,
            )

        # Offer doesn't clear floor — AE counter-offers
        prev_vendor_price = self._current_offer.get("price_per_seat", self._vendor_list_price)
        ae_msg, new_offer = self._opponent.respond(
            action, turn, self._conversation_history, self._current_offer
        )
        # Task 21 — Shaping rewards
        if new_offer.get("price_per_seat", prev_vendor_price) < prev_vendor_price:
            self._shaping_reward += 0.05  # vendor moved toward agent
        if action.action_type == "probe" and action.message and "competitor" in action.message.lower():
            self._shaping_reward += 0.03  # competitor reference used
        if action.contract_length >= 3.0:
            self._shaping_reward += 0.03  # agent offered long-term commitment
        self._current_offer = new_offer
        self._conversation_history.append(f"AE: {ae_msg}")

        # Safety net: opponent may also accept in edge cases
        if any(kw in ae_msg.lower() for kw in ("we have a deal", "paperwork", "we're aligned")):
            reward = self._compute_reward(self._current_offer)
            return self._obs(ae_message=ae_msg, done=True, reward=reward)

        if turn >= self._max_turns:
            return self._obs(ae_message=ae_msg, done=True, reward=0.0)

        return self._obs(ae_message=ae_msg, done=False, reward=0.0)

    def _obs(
        self,
        ae_message: str,
        done: bool,
        reward: float,
    ) -> NegotiateObservation:
        # Get relevant instructions for current state
        active_instructions_list = []
        if self._enable_instructions:
            state = {
                "vendor": self._scenario.get("vendor", ""),
                "remaining_budget": getattr(self, "_remaining_budget", None),
                "total_budget": getattr(self, "_total_budget", None),
            }
            relevant_insts = get_relevant_instructions(state, max_instructions=5)
            active_instructions_list = [f"#{inst['id']}: {inst['rule']}" for inst in relevant_insts]
        
        # Get workflow stage info
        workflow_stage = "negotiation"
        if self._enable_workflow and self._workflow_manager:
            workflow_stage = self._workflow_manager.current_stage.value
        
        return NegotiateObservation(
            context=self._scenario["context"],
            your_max_price=self._agent_max_price,
            your_max_length=self._agent_max_length,
            your_max_cap=self._agent_max_cap,
            ae_message=ae_message,
            conversation_history=list(self._conversation_history),
            turn_number=self._step_count,
            max_turns=self._max_turns,
            active_constraints=list(self._active_constraints),
            current_offer=dict(self._current_offer),
            reward=reward,
            done=done,
            # Long-horizon fields
            competitor_price=self._scenario.get("competitor_price", 0.0),
            active_instructions=active_instructions_list,
            workflow_stage=workflow_stage,
        )

    def _compute_reward(self, deal: dict[str, Any]) -> float:
        vendor_list = self._vendor_list_price
        vendor_floor = self._vendor_floor_price
        agent_target_cap = self._agent_max_cap
        vendor_max_cap = self._vendor_max_cap

        deal_price = deal.get("price_per_seat", vendor_list)
        deal_length = max(1.0, min(3.0, deal.get("contract_length", 2.0)))
        deal_cap = deal.get("annual_increase_cap", vendor_max_cap)

        price_range = vendor_list - vendor_floor
        price_score = (vendor_list - deal_price) / price_range if price_range > 0 else 1.0
        price_score = max(0.0, min(1.0, price_score))

        length_score = 1.0 - (deal_length - 1.0) / 2.0
        length_score = max(0.0, min(1.0, length_score))

        cap_range = vendor_max_cap - agent_target_cap
        cap_score = (vendor_max_cap - deal_cap) / cap_range if cap_range > 0 else 1.0
        cap_score = max(0.0, min(1.0, cap_score))

        raw = 0.5 * price_score + 0.3 * length_score + 0.2 * cap_score

        # Task 24 — Budget awareness: total contract value vs approved budget
        budget = self._scenario.get("budget", float("inf"))
        seat_count = self._scenario.get("seat_count", 1)
        total_contract_value = deal_price * seat_count * deal_length * 12
        if budget < float("inf"):
            if total_contract_value <= budget:
                raw += 0.3
            else:
                raw -= 0.4

        # Task 21 — Add accumulated shaping reward
        raw += self._shaping_reward

        return round(max(0.0, raw - self._turn_penalties), 4)

    @property
    def state(self) -> dict[str, Any]:
        """Minimal state dict; openenv base class marks this abstract."""
        return {
            "step_count": self._step_count,
            "difficulty": self._difficulty_name,
            "action_counter": dict(self._action_counter),
        }
