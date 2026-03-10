"""Pydantic models for NegotiateEnv: Action and Observation."""

from typing import Any, Optional

from openenv.core.env_server import Action, Observation


class NegotiateAction(Action):
    """Action from the agent (procurement manager) to the environment."""

    action_type: str  # "offer" | "counter" | "probe" | "accept" | "walkaway" | "renegotiate" | "use_crm" | "check_approval"
    price_per_seat: float = 0.0
    contract_length: float = 0.0
    annual_increase_cap: float = 0.0
    message: str = ""
    
    # Renegotiation fields (for long-horizon recovery)
    deal_id: Optional[int] = None  # Which deal to renegotiate (for multi-deal scenarios)
    reason: Optional[str] = None  # Reason for renegotiation
    
    # Multi-app workflow fields
    app_name: Optional[str] = None  # "crm" | "approval"
    app_action: Optional[str] = None  # CRM: "get_account" | "get_deals" | "get_intel" | Approval: "check_required" | "submit"
    app_params: Optional[dict] = None  # Parameters for app action


class NegotiateObservation(Observation):
    """Observation returned to the agent after reset or step."""

    context: str = ""
    your_max_price: float = 0.0
    your_max_length: float = 0.0
    your_max_cap: float = 0.0
    ae_message: str = ""
    conversation_history: list[str] = []
    turn_number: int = 0
    max_turns: int = 0
    active_constraints: list[str] = []
    current_offer: dict[str, Any] = {}
    reward: float = 0.0
    done: bool = False
    
    # Long-horizon planning fields
    competitor_price: float = 0.0  # For instruction following
    remaining_budget: Optional[float] = None  # For multi-deal tracking
    total_budget: Optional[float] = None  # For multi-deal tracking
    current_deal: Optional[int] = None  # Which deal (1-5) in multi-deal scenario
    total_deals: Optional[int] = None  # Total deals to negotiate
    closed_deals: list[dict] = []  # Previously closed deals
    quarterly_quota: Optional[int] = None  # Deals needed this quarter
    deals_closed_this_quarter: int = 0  # Progress toward quota
    
    # Instruction following
    active_instructions: list[str] = []  # Current relevant instructions
    instruction_violations: list[str] = []  # Any violated instructions
    
    # Multi-app workflow fields
    available_apps: list[str] = []  # ["crm", "approval"]
    app_result: Optional[dict] = None  # Latest app interaction result
    workflow_stage: str = "negotiation"  # Current stage in workflow
    crm_data: Optional[dict] = None  # Cached CRM data
    approval_status: Optional[dict] = None  # Approval status if submitted
