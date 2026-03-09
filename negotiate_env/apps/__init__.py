"""Enterprise application integrations"""
from .crm import SalesforceCRM
from .approval import ApprovalWorkflow

__all__ = ["SalesforceCRM", "ApprovalWorkflow"]
