"""
GPT-5 implementation using OpenAI API.
"""
import os
import time
import logging
from typing import Dict, Any
import openai
from openai import OpenAI

from src.models.base import VisionModel
from src.config import MODEL_CONFIGS
from src.utils.helpers import encode_image_to_base64


class GPTModel(VisionModel):
    """GPT-5 implementation using OpenAI API."""
    
    def __init__(self):
        """Initialize GPT-5 model with OpenAI client."""
        super().__init__("gpt-5")
        self.provider = "openai"
        self.model_id = MODEL_CONFIGS["gpt-5"]["model_id"]
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        self.logger = logging.getLogger(__name__)
    
    def evaluate_chart(self, image_path: str, prompt: str) -> Dict[str, Any]:
        """
        Evaluate a chart using GPT-5.
        
        Args:
            image_path: Path to the chart image
            prompt: Text prompt for evaluation
            
        Returns:
            Dictionary with response, tokens, cost, latency, and error info
        """
        start_time = time.perf_counter()
        
        try:
            # Encode image to base64
            image_base64 = encode_image_to_base64(image_path)
            
            # Prepare messages with data URI format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ]
            
            # Make API call with retry logic
            response = self._make_request_with_retry(messages)
            
            # Extract response data
            response_text = response.choices[0].message.content
            usage = response.usage
            
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens
            
            # Calculate cost
            config = MODEL_CONFIGS["gpt-5"]
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
            error_msg = f"GPT-5 API error: {str(e)}"
            self.logger.error(error_msg)
            
            return {
                "response": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "cost": 0.0,
                "latency_seconds": latency,
                "error": error_msg
            }
    
    def _make_request_with_retry(self, messages: list, max_retries: int = 3) -> Any:
        """
        Make API request with exponential backoff retry logic.
        
        Args:
            messages: List of messages for OpenAI API
            max_retries: Maximum number of retry attempts
            
        Returns:
            API response
            
        Raises:
            Exception: If all retry attempts fail
        """
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    max_completion_tokens=MODEL_CONFIGS["gpt-5"]["max_tokens"],
                    temperature=1.0
                )
                
                return response
                
            except openai.RateLimitError as e:
                if attempt == max_retries - 1:
                    raise e
                
                # Exponential backoff for rate limits
                wait_time = 2 ** attempt
                self.logger.warning(f"Rate limit hit, retrying in {wait_time}s: {str(e)}")
                time.sleep(wait_time)
            
            except openai.APIError as e:
                if attempt == max_retries - 1:
                    raise e
                
                wait_time = 2 ** attempt
                self.logger.warning(f"API error, retrying in {wait_time}s: {str(e)}")
                time.sleep(wait_time)
            
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                
                wait_time = 2 ** attempt
                self.logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {str(e)}")
                time.sleep(wait_time)
        
        raise Exception("All retry attempts failed")
