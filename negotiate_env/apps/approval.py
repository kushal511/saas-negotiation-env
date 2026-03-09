"""Approval Workflow System (Simulated)"""
import random
import time
from datetime import datetime
from typing import List, Dict


class ApprovalWorkflow:
    """Simulated enterprise approval workflow system"""
    
    # Approval hierarchy based on deal size
    APPROVAL_RULES = {
        "0-50000": ["manager"],
        "50000-200000": ["manager", "director"],
        "200000-500000": ["manager", "director", "vp"],
        "500000+": ["manager", "director", "vp", "cfo"]
    }
    
    # Discount authority by role
    DISCOUNT_AUTHORITY = {
        "sales_rep": 0.10,    # 10% max discount
        "manager": 0.20,      # 20% max discount
        "director": 0.30,     # 30% max discount
        "vp": 0.40,           # 40% max discount
        "cfo": 0.50           # 50% max discount
    }
    
    def __init__(self):
        self.api_delay = 0.3
        self.approval_history = []
    
    def check_approval_required(self, deal_value: float, discount_percent: float) -> dict:
        """Check if approval is required and from whom"""
        time.sleep(self.api_delay)
        
        # Determine required approvers based on deal size
        required_approvers = self._get_required_approvers(deal_value)
        
        # Check if discount exceeds authority
        discount_approver = self._get_discount_approver(discount_percent)
        
        # Combine requirements
        all_approvers = list(set(required_approvers + [discount_approver]))
        
        return {
            "approval_required": len(all_approvers) > 0,
            "required_approvers": all_approvers,
            "deal_value": deal_value,
            "discount_percent": discount_percent,
            "reason": self._get_approval_reason(deal_value, discount_percent),
            "estimated_time": f"{len(all_approvers) * 2} hours",
            "source": "Approval Workflow System"
        }
    
    def submit_for_approval(self, deal_details: dict) -> dict:
        """Submit deal for approval"""
        time.sleep(self.api_delay)
        
        deal_value = deal_details.get("deal_value", 0)
        discount = deal_details.get("discount_percent", 0)
        
        required_approvers = self._get_required_approvers(deal_value)
        
        # Simulate approval process
        approval_id = f"APR-{random.randint(10000, 99999)}"
        
        approvals = []
        for approver in required_approvers:
            # Simulate approval decision (90% approval rate)
            approved = random.random() < 0.9
            approvals.append({
                "approver_role": approver,
                "approver_name": self._get_approver_name(approver),
                "status": "approved" if approved else "pending",
                "timestamp": datetime.now().isoformat(),
                "comments": self._get_approval_comment(approver, approved)
            })
        
        overall_status = "approved" if all(a["status"] == "approved" for a in approvals) else "pending"
        
        result = {
            "approval_id": approval_id,
            "status": overall_status,
            "deal_value": deal_value,
            "discount_percent": discount,
            "approvals": approvals,
            "submitted_at": datetime.now().isoformat(),
            "source": "Approval Workflow System"
        }
        
        self.approval_history.append(result)
        return result
    
    def get_approval_status(self, approval_id: str) -> dict:
        """Check status of approval request"""
        time.sleep(self.api_delay)
        
        # Find in history
        for approval in self.approval_history:
            if approval["approval_id"] == approval_id:
                return approval
        
        return {
            "error": "Approval not found",
            "approval_id": approval_id
        }
    
    def _get_required_approvers(self, deal_value: float) -> List[str]:
        """Determine required approvers based on deal size"""
        if deal_value >= 500000:
            return ["manager", "director", "vp", "cfo"]
        elif deal_value >= 200000:
            return ["manager", "director", "vp"]
        elif deal_value >= 50000:
            return ["manager", "director"]
        elif deal_value > 0:
            return ["manager"]
        return []
    
    def _get_discount_approver(self, discount_percent: float) -> str:
        """Determine who can approve this discount level"""
        discount_decimal = discount_percent / 100.0
        
        if discount_decimal > 0.40:
            return "cfo"
        elif discount_decimal > 0.30:
            return "vp"
        elif discount_decimal > 0.20:
            return "director"
        elif discount_decimal > 0.10:
            return "manager"
        return "sales_rep"
    
    def _get_approval_reason(self, deal_value: float, discount_percent: float) -> str:
        """Generate reason for approval requirement"""
        reasons = []
        
        if deal_value >= 200000:
            reasons.append(f"Deal value ${deal_value:,.0f} exceeds threshold")
        
        if discount_percent > 20:
            reasons.append(f"Discount {discount_percent}% exceeds standard authority")
        
        return "; ".join(reasons) if reasons else "Standard approval process"
    
    def _get_approver_name(self, role: str) -> str:
        """Get simulated approver name"""
        names = {
            "manager": "Michael Rodriguez",
            "director": "Jennifer Park",
            "vp": "David Thompson",
            "cfo": "Lisa Anderson"
        }
        return names.get(role, "Unknown")
    
    def _get_approval_comment(self, role: str, approved: bool) -> str:
        """Generate approval comment"""
        if approved:
            comments = [
                "Approved - good strategic fit",
                "Approved - within budget",
                "Approved - aligns with Q4 goals",
                "Approved - customer is high priority"
            ]
        else:
            comments = [
                "Pending - need more justification",
                "Pending - awaiting budget confirmation",
                "Pending - requires additional review"
            ]
        return random.choice(comments)
