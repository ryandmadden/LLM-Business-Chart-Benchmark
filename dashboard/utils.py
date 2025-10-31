"""
Dashboard utility functions for data loading and visualization.
"""
import os
import sys
import json
import pandas as pd
import numpy as np
import streamlit as st
import warnings
from typing import Optional, Dict, Any, List

# Suppress Plotly warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="plotly")
warnings.filterwarnings("ignore", category=FutureWarning, module="plotly")

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from storage.s3_handler import S3Handler

MODEL_COLORS = {
    # Full formatted names (what appears after get_full_model_name())
    'Claude Sonnet 4.5': '#FF6B6B',  # Red
    'GPT-5': '#4A90E2',              # Blue
    'Gemini 2.5 Pro': '#50C878',     # Green
    
    # Also include short names as fallback
    'claude-sonnet-4.5': '#FF6B6B',
    'gpt-5': '#4A90E2',
    'gemini-2.5-pro': '#50C878',
}

def get_full_model_name(model_name: str) -> str:
    """Convert short model names to full display names."""
    model_mapping = {
        'claude': 'Claude Sonnet 4.5',
        'gpt': 'GPT-5',
        'gemini': 'Gemini 2.5 Pro'
    }
    return model_mapping.get(model_name, model_name)

def format_model_names_in_df(df: pd.DataFrame) -> pd.DataFrame:
    """Format model names in a dataframe."""
    df = df.copy()
    if 'model_name' in df.columns:
        df['model_name'] = df['model_name'].apply(get_full_model_name)
    if 'Model' in df.columns:
        df['Model'] = df['Model'].apply(get_full_model_name)  
    return df

@st.cache_data
def load_evaluation_results() -> Optional[pd.DataFrame]:
    """Load evaluation results with caching, preferring rescored versions."""
    import json
    
    try:
        # Try S3 first
        s3_handler = S3Handler()
        df = s3_handler.download_latest_results()
        if df is not None:
            # Parse error_flags if needed
            if 'error_flags' in df.columns:
                df['error_flags'] = df['error_flags'].apply(parse_error_flags_json)
            return df
    except Exception:
        pass
    
    # Fallback to local files
    results_dir = "results"
    if os.path.exists(results_dir):
        # Find all scored results files (including rescored), excluding backups
        parquet_files = [
            f for f in os.listdir(results_dir) 
            if f.endswith('.parquet') 
            and 'scored_results' in f 
            and 'backup' not in f
        ]
        
        if parquet_files:
            # Get most recent file by modification time
            latest_file = max(parquet_files, key=lambda f: os.path.getmtime(os.path.join(results_dir, f)))
            filepath = os.path.join(results_dir, latest_file)
            df = pd.read_parquet(filepath)
            
            # Show which file was loaded
            st.info(f"📁 Loaded: {latest_file}")
            
            # Parse error_flags JSON strings back to dicts
            if 'error_flags' in df.columns:
                df['error_flags'] = df['error_flags'].apply(parse_error_flags_json)
            
            return df
    
    return None


def parse_error_flags_json(error_flags):
    """
    Parse error_flags from JSON string to dict.
    
    Args:
        error_flags: Either a JSON string or already a dict
        
    Returns:
        Dictionary of error flags
    """
    if isinstance(error_flags, str):
        try:
            return json.loads(error_flags)
        except (json.JSONDecodeError, TypeError):
            return {}
    
    return error_flags if isinstance(error_flags, dict) else {}


@st.cache_data
def load_ground_truth() -> Optional[Dict[str, Any]]:
    """Load ground truth data with caching."""
    import json
    
    ground_truth_path = "data/ground_truth.json"
    if os.path.exists(ground_truth_path):
        with open(ground_truth_path, 'r') as f:
            return json.load(f)
    return None


def get_model_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate summary statistics for each model."""
    stats = []
    
    for model_name in df['model_name'].unique():
        model_df = df[df['model_name'] == model_name]
        
        # Basic metrics
        total_questions = len(model_df)
        accuracy = (model_df['score'] >= 0.5).sum() / total_questions
        
        # Tier-specific accuracy
        tier1_df = model_df[model_df['question_tier'] == 'tier1_factual']
        tier2_df = model_df[model_df['question_tier'] == 'tier2_pattern']
        
        tier1_accuracy = (tier1_df['score'] >= 0.5).sum() / len(tier1_df) if len(tier1_df) > 0 else 0
        tier2_accuracy = (tier2_df['score'] >= 0.5).sum() / len(tier2_df) if len(tier2_df) > 0 else 0
        
        # Cost and latency
        avg_cost = model_df['cost'].mean()
        avg_latency = model_df['latency_seconds'].mean()
        
        
        
        stats.append({
            'Model': get_full_model_name(model_name),
            'Accuracy': accuracy,
            'Tier 1 Accuracy': tier1_accuracy,
            'Tier 2 Accuracy': tier2_accuracy,
            'Avg Cost': avg_cost,
            'Avg Latency (s)': avg_latency,
        })
    
    return pd.DataFrame(stats)


def get_chart_image_path(chart_id: str) -> Optional[str]:
    """Get the path to a chart image."""
    image_path = f"data/charts/{chart_id}.png"
    if os.path.exists(image_path):
        return image_path
    return None



def create_performance_comparison_chart(df: pd.DataFrame):
    """Create a performance comparison chart."""
    import plotly.express as px
    
    # Check which column name is being used
    if 'Model' in df.columns:
        input_model_col = 'Model'
    elif 'model_name' in df.columns:
        input_model_col = 'model_name'
    else:
        raise ValueError("DataFrame must contain either 'Model' or 'model_name' column")
    
    # Calculate average scores by model and tier
    tier1_scores = df[df['question_tier'] == 'tier1_factual'].groupby(input_model_col)['score'].mean()
    tier2_scores = df[df['question_tier'] == 'tier2_pattern'].groupby(input_model_col)['score'].mean()
    
    # Combine into a single dataframe
    model_scores = pd.DataFrame({
        'Task 1 Performance': tier1_scores,
        'Task 2 Performance': tier2_scores
    }).reset_index()
    
    # Apply full model names if not already formatted
    if input_model_col == 'model_name':
        model_scores[input_model_col] = model_scores[input_model_col].apply(get_full_model_name)
    
    # Sort by average of both tiers
    model_scores['avg_score'] = (model_scores['Task 1 Performance'] + model_scores['Task 2 Performance']) / 2
    model_scores = model_scores.sort_values('avg_score', ascending=False)

    # Debug: Print what we have
    print(f"Model scores dataframe:\n{model_scores}")
    print(f"MODEL_COLORS keys: {list(MODEL_COLORS.keys())}")
    print(f"Models in data: {list(model_scores[input_model_col].unique())}")
    
    # Create comparison chart with color mapping
    fig = px.bar(
        model_scores, 
        x=input_model_col, 
        y=['Task 1 Performance', 'Task 2 Performance'],
        title='Model Performance Comparison',
        barmode='group',
        color=input_model_col,
        color_discrete_map=MODEL_COLORS
    )

    fig.update_layout(
        yaxis_title='Performance',
        yaxis_range=[0, 1],
        xaxis_title='Model',
        height=400,
        showlegend=True
    )
    
    return fig


def create_cost_performance_scatter(df: pd.DataFrame):
    """Create cost vs performance scatter plot."""
    import plotly.express as px
    
    # Check which column name is being used
    if 'Model' in df.columns:
        input_model_col = 'Model'
    elif 'model_name' in df.columns:
        input_model_col = 'model_name'
    else:
        raise ValueError("DataFrame must contain either 'Model' or 'model_name' column")
    
    # Calculate average metrics per model
    model_metrics = df.groupby(input_model_col).agg({
        'score': 'mean',
        'cost': 'mean',
        'latency_seconds': 'mean'
    }).reset_index()
    
    # Rename columns for display
    model_metrics = model_metrics.rename(columns={
        'score': 'Accuracy',
        'cost': 'Avg Cost',
        'latency_seconds': 'Avg Latency (s)'
    })
    
    # Apply full model names if not already formatted
    if input_model_col == 'model_name':
        model_metrics[input_model_col] = model_metrics[input_model_col].apply(get_full_model_name)
    
    # Debug: Print what we have
    print(f"Model metrics dataframe:\n{model_metrics}")
    print(f"Models in data: {list(model_metrics[input_model_col].unique())}")
    

    fig = px.scatter(
        model_metrics,
        x='Avg Cost',
        y='Accuracy',
        size='Avg Latency (s)',
        title='Cost vs Performance Trade-off',
        color=input_model_col,
        color_discrete_map=MODEL_COLORS
    )

    fig.update_layout(
        height=400,
        showlegend=True
    )
    return fig




def get_chart_image_path(chart_id: str) -> str:
    """Get the file path for a chart image."""
    return f"data/charts/{chart_id}.png"


def format_error_flags(error_flags) -> str:
    """Format error flags for display."""
    if not error_flags:
        return "No errors detected"
    
    # Parse error_flags if it's a JSON string
    if isinstance(error_flags, str):
        try:
            import json
            error_flags = json.loads(error_flags)
        except (json.JSONDecodeError, TypeError):
            return "Error parsing flags"
    
    if not isinstance(error_flags, dict):
        return "No errors detected"
    
    errors = []
    
