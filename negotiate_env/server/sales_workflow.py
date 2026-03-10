"""Full B2B Sales Workflow for Long-Horizon Planning.

8-stage sales workflow from lead qualification to onboarding handoff.
Total: 50 turns across all stages.
"""

from __future__ import annotations
from typing import Dict, Any, List
from enum import Enum


class WorkflowStage(Enum):
    """8 stages of B2B sales workflow."""
    LEAD_QUALIFICATION = "lead_qualification"
    DISCOVERY = "discovery"
    PROPOSAL = "proposal"
    NEGOTIATION = "negotiation"
    CONTRACT_REVIEW = "contract_review"
    APPROVAL = "approval"
    CLOSING = "closing"
    ONBOARDING_HANDOFF = "onboarding_handoff"


# Stage configuration: (start_turn, end_turn, required_actions)
WORKFLOW_STAGES = {
    WorkflowStage.LEAD_QUALIFICATION: {
        "turns": (1, 5),
        "description": "Qualify the lead and assess fit",
        "required_actions": ["probe", "probe"],
        "success_criteria": "Understand company size, budget, timeline",
        "next_stage": WorkflowStage.DISCOVERY,
    },
    WorkflowStage.DISCOVERY: {
        "turns": (6, 10),
        "description": "Deep dive into needs and pain points",
        "required_actions": ["probe", "probe", "probe"],
        "success_criteria": "Identify 3+ pain points and requirements",
        "next_stage": WorkflowStage.PROPOSAL,
    },
    WorkflowStage.PROPOSAL: {
        "turns": (11, 15),
        "description": "Present solution and initial pricing",
        "required_actions": ["offer"],
        "success_criteria": "Present tailored solution with pricing",
        "next_stage": WorkflowStage.NEGOTIATION,
    },
    WorkflowStage.NEGOTIATION: {
        "turns": (16, 30),
        "description": "Negotiate terms, pricing, and contract",
        "required_actions": ["counter", "counter"],
        "success_criteria": "Reach agreement on price and terms",
        "next_stage": WorkflowStage.CONTRACT_REVIEW,
    },
    WorkflowStage.CONTRACT_REVIEW: {
        "turns": (31, 35),
        "description": "Legal and procurement review",
        "required_actions": ["probe"],
        "success_criteria": "Address legal concerns and compliance",
        "next_stage": WorkflowStage.APPROVAL,
    },
    WorkflowStage.APPROVAL: {
        "turns": (36, 40),
        "description": "Get internal stakeholder approval",
        "required_actions": ["probe"],
        "success_criteria": "Secure approval from decision makers",
        "next_stage": WorkflowStage.CLOSING,
    },
    WorkflowStage.CLOSING: {
        "turns": (41, 45),
        "description": "Finalize and sign contract",
        "required_actions": ["accept"],
        "success_criteria": "Contract signed by both parties",
        "next_stage": WorkflowStage.ONBOARDING_HANDOFF,
    },
    WorkflowStage.ONBOARDING_HANDOFF: {
        "turns": (46, 50),
        "description": "Hand off to customer success team",
        "required_actions": [],
        "success_criteria": "Smooth transition to CS team",
        "next_stage": None,
    },
}


class SalesWorkflowManager:
    """Manages progression through B2B sales workflow stages."""
    
    def __init__(self):
        self.current_stage = WorkflowStage.LEAD_QUALIFICATION
        self.stage_history: List[WorkflowStage] = []
        self.actions_taken: List[str] = []
        self.stage_completion: Dict[WorkflowStage, bool] = {}
        
    def reset(self):
        """Reset workflow to beginning."""
        self.current_stage = WorkflowStage.LEAD_QUALIFICATION
        self.stage_history = [self.current_stage]
        self.actions_taken = []
        self.stage_completion = {}
        
    def get_current_stage_info(self) -> Dict[str, Any]:
        """Get information about current stage."""
        stage_config = WORKFLOW_STAGES[self.current_stage]
        return {
            "stage": self.current_stage.value,
            "description": stage_config["description"],
            "turn_range": stage_config["turns"],
            "required_actions": stage_config["required_actions"],
            "success_criteria": stage_config["success_criteria"],
        }
    
    def check_stage_transition(self, turn_number: int, action_type: str) -> bool:
        """
        Check if we should transition to next stage.
        
        Returns True if transitioned.
        """
        stage_config = WORKFLOW_STAGES[self.current_stage]
        start_turn, end_turn = stage_config["turns"]
        
        # Record action
        self.actions_taken.append(action_type)
        
        # Check if we've completed required actions
        required_actions = stage_config["required_actions"]
        actions_completed = all(
            action in self.actions_taken for action in required_actions
        )
        
        # Check if we're past the stage's turn range
        past_turn_range = turn_number > end_turn
        
        # Transition if requirements met or past turn range
        if actions_completed or past_turn_range:
            next_stage = stage_config["next_stage"]
            if next_stage:
                self.stage_completion[self.current_stage] = actions_completed
                self.current_stage = next_stage
                self.stage_history.append(self.current_stage)
                self.actions_taken = []  # Reset for new stage
                return True
        
        return False
    
    def get_stage_reward_bonus(self) -> float:
        """Calculate reward bonus based on stage completion."""
        bonus = 0.0
        
        # Bonus for completing stages properly
        for stage, completed in self.stage_completion.items():
            if completed:
                bonus += 0.05
        
        # Bonus for reaching later stages
        stage_order = list(WorkflowStage)
        current_idx = stage_order.index(self.current_stage)
        bonus += current_idx * 0.02
        
        return bonus
    
    def format_workflow_context(self, turn_number: int) -> str:
        """Format workflow information for agent prompt."""
        stage_info = self.get_current_stage_info()
        start_turn, end_turn = stage_info["turn_range"]
        
        context = f"""

--- SALES WORKFLOW STAGE ---
Current Stage: {stage_info['stage'].replace('_', ' ').title()}
Description: {stage_info['description']}
Turn Range: {start_turn}-{end_turn} (Current: {turn_number})
Success Criteria: {stage_info['success_criteria']}

Required Actions: {', '.join(stage_info['required_actions']) if stage_info['required_actions'] else 'None'}
Actions Taken: {', '.join(self.actions_taken) if self.actions_taken else 'None'}

Stage Progress:
"""
        
        # Show completed stages
        for stage in self.stage_history[:-1]:
            completed = self.stage_completion.get(stage, False)
            status = "✅" if completed else "⚠️"
            context += f"  {status} {stage.value.replace('_', ' ').title()}\n"
        
        # Show current stage
        context += f"  ▶️  {self.current_stage.value.replace('_', ' ').title()} (In Progress)\n"
        
        return context
    
    def get_state(self) -> Dict[str, Any]:
        """Get current workflow state."""
        return {
            "current_stage": self.current_stage.value,
            "stage_history": [s.value for s in self.stage_history],
            "actions_taken": self.actions_taken,
            "stage_completion": {k.value: v for k, v in self.stage_completion.items()},
            "stages_completed": len(self.stage_completion),
            "total_stages": len(WorkflowStage),
        }
