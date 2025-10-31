"""
Gemini 2.5 Pro implementation using Google Cloud Vertex AI.
"""
import os
import time
import logging
from typing import Dict, Any
import vertexai
from vertexai.generative_models import GenerativeModel, Part

from src.models.base import VisionModel
from src.config import MODEL_CONFIGS


class GeminiModel(VisionModel):
    """Gemini 2.5 Pro implementation using Google Cloud Vertex AI."""
    
    def __init__(self):
        """Initialize Gemini model with Vertex AI client."""
        super().__init__("gemini-2.5-pro")
        self.provider = "gcp_vertex"
        self.model_id = MODEL_CONFIGS["gemini-2.5-pro"]["model_id"]
        
        # Initialize Vertex AI
        project_id = os.getenv('GCP_PROJECT_ID')
        location = os.getenv('GCP_LOCATION', 'global')
        
        vertexai.init(project=project_id, location=location)
        self.model = GenerativeModel(self.model_id)
        
        self.logger = logging.getLogger(__name__)
    
    def evaluate_chart(self, image_path: str, prompt: str) -> Dict[str, Any]:
        """
        Evaluate a chart using Gemini 2.5 Pro.
        
        Args:
            image_path: Path to the chart image
            prompt: Text prompt for evaluation
            
        Returns:
            Dictionary with response, tokens, cost, latency, and error info
        """
        start_time = time.perf_counter()
        
        try:
            # Read image file
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()
            
            # Create image part
            image_part = Part.from_data(image_data, mime_type="image/png")
            
            # Prepare content
            contents = [image_part, prompt]
            
            # Make API call with retry logic
            response = self._make_request_with_retry(contents)
            
            # Extract response data
            response_text = response.text if hasattr(response, 'text') else str(response)
            
            # Extract token usage from response metadata
            # Note: Gemini token counting may vary by implementation
            usage_metadata = getattr(response, 'usage_metadata', None)
            if usage_metadata:
                input_tokens = getattr(usage_metadata, 'prompt_token_count', 0)
                output_tokens = getattr(usage_metadata, 'candidates_token_count', 0)
            else:
                input_tokens = 0
                output_tokens = 0
            
            # If usage metadata not available, estimate based on text length
            if input_tokens == 0:
                input_tokens = len(prompt.split()) + 100  # Rough estimate for image tokens
            if output_tokens == 0:
                output_tokens = len(response_text.split())
            
            # Calculate cost
            config = MODEL_CONFIGS["gemini-2.5-pro"]
            cost = (input_tokens / 1000000 * config["input_price_per_mtok"] + 
                   output_tokens / 1000000 * config["output_price_per_mtok"])
            
            latency = time.perf_counter() - start_time
            
            return {
                "response": response_text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost,
                "latency_seconds": latency,
                "error": None
            }
            
        except Exception as e:
            latency = time.perf_counter() - start_time
            error_msg = f"Gemini API error: {str(e)}"
            self.logger.error(error_msg)
            
            return {
                "response": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "cost": 0.0,
                "latency_seconds": latency,
                "error": error_msg
            }
    
    def _make_request_with_retry(self, contents: list, max_retries: int = 3) -> Any:
        """
        Make API request with exponential backoff retry logic.
        
        Args:
            contents: List of content parts for Gemini API
            max_retries: Maximum number of retry attempts
            
        Returns:
            API response
            
        Raises:
            Exception: If all retry attempts fail
        """
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    contents,
                    generation_config={
                        "max_output_tokens": MODEL_CONFIGS["gemini-2.5-pro"]["max_tokens"],
                        "temperature": 0.1,
                    }
                )
                
                return response
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                
                # Exponential backoff
                wait_time = 2 ** attempt
                self.logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {str(e)}")
                time.sleep(wait_time)
        
        raise Exception("All retry attempts failed")
