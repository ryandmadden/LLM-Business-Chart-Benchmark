"""
Abstract base class for vision model implementations.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any


class VisionModel(ABC):
    """Abstract base class for vision model implementations."""
    
    def __init__(self, model_name: str):
        """
        Initialize the vision model.
        
        Args:
            model_name: Name of the model for identification
        """
        self.model_name = model_name
    
    @abstractmethod
    def evaluate_chart(self, image_path: str, prompt: str) -> Dict[str, Any]:
        """
        Evaluate a chart image with the given prompt.
        
        Args:
            image_path: Path to the chart image file
            prompt: Text prompt for the evaluation
            
        Returns:
            Dictionary containing:
                - response: str - Model's text response
                - input_tokens: int - Number of input tokens used
                - output_tokens: int - Number of output tokens generated
                - cost: float - Cost in USD for this request
                - latency_seconds: float - Time taken for the request
                - error: str | None - Error message if request failed
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and configuration.
        
        Returns:
            Dictionary with model metadata
        """
        return {
            "model_name": self.model_name,
            "provider": getattr(self, "provider", "unknown"),
            "model_id": getattr(self, "model_id", "unknown")
        }
