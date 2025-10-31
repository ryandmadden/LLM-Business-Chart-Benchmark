"""
Cost tracking and budget enforcement utilities.
"""
import logging
from typing import Dict, Any
from dataclasses import dataclass
from src.config import BUDGET_LIMITS


@dataclass
class CostEntry:
    """Data class for tracking individual cost entries."""
    model_name: str
    input_tokens: int
    output_tokens: int
    cost: float
    timestamp: str


class BudgetExceededError(Exception):
    """Exception raised when budget limits are exceeded."""
    pass


class CostTracker:
    """Track costs and enforce budget limits."""
    
    def __init__(self):
        """Initialize the cost tracker."""
        self.costs: Dict[str, float] = {}  # model_name -> total_cost
        self.total_cost: float = 0.0
        self.entries: list[CostEntry] = []
        self.logger = logging.getLogger(__name__)
    
    def add_cost(self, model_name: str, input_tokens: int, output_tokens: int, 
                 cost: float, timestamp: str) -> None:
        """
        Add a cost entry and check budget limits.
        
        Args:
            model_name: Name of the model
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cost: Cost in USD
            timestamp: Timestamp of the request
            
        Raises:
            BudgetExceededError: If budget limits would be exceeded
        """
        # Check per-model budget
        current_model_cost = self.costs.get(model_name, 0.0)
        if current_model_cost + cost > BUDGET_LIMITS["per_model_budget"]:
            raise BudgetExceededError(
                f"Per-model budget exceeded for {model_name}. "
                f"Current: ${current_model_cost:.2f}, "
                f"Request: ${cost:.2f}, "
                f"Limit: ${BUDGET_LIMITS['per_model_budget']:.2f}"
            )
        
        # Check total budget
        if self.total_cost + cost > BUDGET_LIMITS["total_budget"]:
            raise BudgetExceededError(
                f"Total budget exceeded. "
                f"Current: ${self.total_cost:.2f}, "
                f"Request: ${cost:.2f}, "
                f"Limit: ${BUDGET_LIMITS['total_budget']:.2f}"
            )
        
        # Add the cost
        self.costs[model_name] = current_model_cost + cost
        self.total_cost += cost
        
        # Add entry
        entry = CostEntry(model_name, input_tokens, output_tokens, cost, timestamp)
        self.entries.append(entry)
        
        # Log warning if approaching threshold
        if self.total_cost >= BUDGET_LIMITS["total_budget"] * BUDGET_LIMITS["warning_threshold"]:
            self.logger.warning(
                f"Budget warning: ${self.total_cost:.2f} used "
                f"({self.total_cost/BUDGET_LIMITS['total_budget']*100:.1f}% of total budget)"
            )
        
        self.logger.info(f"Cost added: {model_name} - ${cost:.4f} (Total: ${self.total_cost:.2f})")
    
    def get_model_cost(self, model_name: str) -> float:
        """
        Get total cost for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Total cost for the model
        """
        return self.costs.get(model_name, 0.0)
    
    def get_total_cost(self) -> float:
        """
        Get total cost across all models.
        
        Returns:
            Total cost in USD
        """
        return self.total_cost
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all costs.
        
        Returns:
            Dictionary with cost summary
        """
        return {
            "total_cost": self.total_cost,
            "per_model_costs": self.costs.copy(),
            "budget_limits": BUDGET_LIMITS.copy(),
            "remaining_budget": BUDGET_LIMITS["total_budget"] - self.total_cost,
            "budget_utilization": self.total_cost / BUDGET_LIMITS["total_budget"] * 100,
            "num_requests": len(self.entries)
        }
    
    def reset(self) -> None:
        """Reset all cost tracking."""
        self.costs.clear()
        self.total_cost = 0.0
        self.entries.clear()
        self.logger.info("Cost tracker reset")
