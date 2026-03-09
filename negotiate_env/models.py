"""Pydantic models for NegotiateEnv: Action and Observation."""

from typing import Any, Optional

from openenv.core.env_server import Action, Observation


class NegotiateAction(Action):
    """Action from the agent (procurement manager) to the environment."""

    action_type: str  # "offer" | "counter" | "probe" | "accept" | "walkaway" | "use_crm" | "check_approval"
    price_per_seat: float = 0.0
    contract_length: float = 0.0
    annual_increase_cap: float = 0.0
    message: str = ""
    
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
    
    # Multi-app workflow fields
    available_apps: list[str] = []  # ["crm", "approval"]
    app_result: Optional[dict] = None  # Latest app interaction result
    workflow_stage: str = "negotiation"  # Current stage in workflow
    crm_data: Optional[dict] = None  # Cached CRM data
    approval_status: Optional[dict] = None  # Approval status if submitted
