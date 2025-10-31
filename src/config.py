"""
Configuration settings for the Chart Evaluation Dashboard.
"""
import os
from typing import Dict, Any

# Model configurations with pricing and settings
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "claude-sonnet-4.5": {
        "provider": "aws_bedrock",
        "model_id": "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "input_price_per_mtok": 3.75,
        "output_price_per_mtok": 18.75,
        "max_tokens": 4096
    },
    "gpt-5": {
        "provider": "openai",
        "model_id": "gpt-5",
        "input_price_per_mtok": 1.25,
        "output_price_per_mtok": 10.0,
        "max_tokens": 4096
    },
    "gemini-2.5-pro": {
        "provider": "gcp_vertex",
        "model_id": "gemini-2.5-pro",
        "input_price_per_mtok": 1.25,
        "output_price_per_mtok": 5.0,
        "max_tokens": 4096
    }
}

# Budget limits with hard enforcement
BUDGET_LIMITS: Dict[str, float] = {
    "total_budget": 50.0,  # USD - hard limit
    "per_model_budget": 20.0,
    "warning_threshold": 0.8
}

# S3 configuration
S3_CONFIG: Dict[str, str] = {
    "bucket_name": "llm-chart-bench",
    "region": "us-east-2",
    "results_prefix": "evaluations/"
}

# Chart generation settings
CHART_CONFIG: Dict[str, Any] = {
    "width": 1092,
    "height": 1092,
    "dpi": 110,
    "format": "png"
}

# Evaluation settings
EVALUATION_CONFIG: Dict[str, Any] = {
    "num_charts": 10,
    "cache_interval": 10,  # Save intermediate results every N evaluations
    "retry_attempts": 3,
    "timeout_seconds": 3600
}

# Logging configuration
LOGGING_CONFIG: Dict[str, str] = {
    "log_file": "logs/evaluation.log",
    "log_level": "INFO",
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
}
