"""
Claude Sonnet 4.5 implementation using AWS Bedrock.
"""
import os
import json
import base64
import time
import logging
from typing import Dict, Any
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, BotoCoreError

from src.models.base import VisionModel
from src.config import MODEL_CONFIGS
from src.utils.helpers import encode_image_to_base64


class ClaudeModel(VisionModel):
    """Claude Sonnet 4.5 implementation using AWS Bedrock."""
    
    def __init__(self):
        """Initialize Claude model with AWS Bedrock client."""
        super().__init__("claude-sonnet-4.5")
        self.provider = "aws_bedrock"
        self.model_id = MODEL_CONFIGS["claude-sonnet-4.5"]["model_id"]
        
        # Initialize Bedrock client with timeout configuration
        self.client = boto3.client(
            'bedrock-runtime',
            region_name=os.getenv('AWS_REGION', 'us-east-1'),
            config=Config(read_timeout=3600)
        )
        
        self.logger = logging.getLogger(__name__)
    
    def evaluate_chart(self, image_path: str, prompt: str) -> Dict[str, Any]:
        """
        Evaluate a chart using Claude Sonnet 4.5.
        
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
            
            # Prepare request body
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": MODEL_CONFIGS["claude-sonnet-4.5"]["max_tokens"],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            }
            
            # Make API call with retry logic
            response = self._make_request_with_retry(request_body)
            
            # Extract response data
            response_text = response['content'][0]['text']
            usage = response['usage']
            
            input_tokens = usage['input_tokens']
            output_tokens = usage['output_tokens']
            
            # Calculate cost
            config = MODEL_CONFIGS["claude-sonnet-4.5"]
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
            error_msg = f"Claude API error: {str(e)}"
            self.logger.error(error_msg)
            
            return {
                "response": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "cost": 0.0,
                "latency_seconds": latency,
                "error": error_msg
            }
    
    def _make_request_with_retry(self, request_body: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
        """
        Make API request with exponential backoff retry logic.
        
        Args:
            request_body: Request body for Bedrock API
            max_retries: Maximum number of retry attempts
            
        Returns:
            API response
            
        Raises:
            Exception: If all retry attempts fail
        """
        for attempt in range(max_retries):
            try:
                response = self.client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(request_body),
                    contentType="application/json", 
                    accept="application/json" 
                )
                
                # Parse response
                response_body = json.loads(response['body'].read())
                return response_body
                
            except (ClientError, BotoCoreError) as e:
                if attempt == max_retries - 1:
                    raise e
                
                # Exponential backoff
                wait_time = 2 ** attempt
                self.logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {str(e)}")
                time.sleep(wait_time)
            
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                
                wait_time = 2 ** attempt
                self.logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {str(e)}")
                time.sleep(wait_time)
        
        raise Exception("All retry attempts failed")
