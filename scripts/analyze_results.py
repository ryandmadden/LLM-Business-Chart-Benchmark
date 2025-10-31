"""
Results analysis script for generating summary statistics and insights.
"""
import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.storage.s3_handler import S3Handler


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_latest_results() -> pd.DataFrame:
    """Load the latest evaluation results."""
    logger = logging.getLogger(__name__)
    
    # Try to load from S3 first
    try:
        s3_handler = S3Handler()
        df = s3_handler.download_latest_results()
        if df is not None:
            logger.info("Loaded results from S3")
            return df
    except Exception as e:
        logger.warning(f"Failed to load from S3: {str(e)}")
    
    # Fallback to local files
    results_dir = "results"
    if os.path.exists(results_dir):
        parquet_files = [f for f in os.listdir(results_dir) if f.endswith('.parquet')]
        if parquet_files:
            # Get the most recent file
            latest_file = sorted(parquet_files)[-1]
            df = pd.read_parquet(os.path.join(results_dir, latest_file))
            logger.info(f"Loaded results from local file: {latest_file}")
            return df
    
    raise FileNotFoundError("No evaluation results found")


def calculate_model_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive metrics for each model."""
    metrics = []
    
    for model_name in df['model_name'].unique():
        model_df = df[df['model_name'] == model_name]
        
        # Basic metrics
        total_questions = len(model_df)
        correct_answers = (model_df['score'] >= 0.5).sum()
        accuracy = correct_answers / total_questions if total_questions > 0 else 0
        
        # Cost metrics
        total_cost = model_df['cost'].sum()
        avg_cost_per_question = model_df['cost'].mean()
        
        # Latency metrics
        avg_latency = model_df['latency_seconds'].mean()
        median_latency = model_df['latency_seconds'].median()
        
        # Tier-specific accuracy
        tier1_df = model_df[model_df['question_tier'] == 'tier1_factual']
        tier2_df = model_df[model_df['question_tier'] == 'tier2_pattern']
        
        tier1_accuracy = (tier1_df['score'] >= 0.5).sum() / len(tier1_df) if len(tier1_df) > 0 else 0
        tier2_accuracy = (tier2_df['score'] >= 0.5).sum() / len(tier2_df) if len(tier2_df) > 0 else 0
        
        # Error analysis
        hallucination_count = 0
        axis_error_count = 0
        trend_reversal_count = 0
        overgeneralization_count = 0
        
        for _, row in model_df.iterrows():
            error_flags = row.get('error_flags', {})
            if isinstance(error_flags, dict):
                if error_flags.get('number_hallucinations', {}).get('has_hallucinations', False):
                    hallucination_count += 1
                if error_flags.get('axis_errors', {}).get('has_axis_errors', False):
                    axis_error_count += 1
                if error_flags.get('trend_reversals', {}).get('has_trend_reversals', False):
                    trend_reversal_count += 1
                if error_flags.get('overgeneralization', {}).get('has_overgeneralization', False):
                    overgeneralization_count += 1
        
        hallucination_rate = hallucination_count / total_questions if total_questions > 0 else 0
        
        metrics.append({
            'model_name': model_name,
            'total_questions': total_questions,
            'accuracy': accuracy,
            'tier1_accuracy': tier1_accuracy,
            'tier2_accuracy': tier2_accuracy,
            'total_cost': total_cost,
            'avg_cost_per_question': avg_cost_per_question,
            'avg_latency_seconds': avg_latency,
            'median_latency_seconds': median_latency,
            'hallucination_count': hallucination_count,
            'hallucination_rate': hallucination_rate,
            'axis_error_count': axis_error_count,
            'trend_reversal_count': trend_reversal_count,
            'overgeneralization_count': overgeneralization_count
        })
    
    return pd.DataFrame(metrics)


def calculate_chart_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate metrics for each chart."""
    metrics = []
    
    for chart_id in df['chart_id'].unique():
        chart_df = df[df['chart_id'] == chart_id]
        
        # Get chart metadata (assuming it's in the first row)
        first_row = chart_df.iloc[0]
        chart_type = first_row.get('chart_type', 'unknown')
        question_tier = first_row.get('question_tier', 'unknown')
        
        # Calculate metrics
        total_questions = len(chart_df)
        avg_score = chart_df['score'].mean()
        avg_cost = chart_df['cost'].mean()
        avg_latency = chart_df['latency_seconds'].mean()
        
        # Model performance
        model_scores = {}
        for model_name in chart_df['model_name'].unique():
            model_chart_df = chart_df[chart_df['model_name'] == model_name]
            model_scores[model_name] = model_chart_df['score'].mean()
        
        metrics.append({
            'chart_id': chart_id,
            'chart_type': chart_type,
            'total_questions': total_questions,
            'avg_score': avg_score,
            'avg_cost': avg_cost,
            'avg_latency': avg_latency,
            'model_scores': model_scores
        })
    
    return pd.DataFrame(metrics)


def calculate_tier_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate metrics by question tier."""
    metrics = []
    
    for tier in df['question_tier'].unique():
        tier_df = df[df['question_tier'] == tier]
        
        total_questions = len(tier_df)
        avg_score = tier_df['score'].mean()
        avg_cost = tier_df['cost'].mean()
        avg_latency = tier_df['latency_seconds'].mean()
        
        # Model performance by tier
        model_performance = {}
        for model_name in tier_df['model_name'].unique():
            model_tier_df = tier_df[tier_df['model_name'] == model_name]
            model_performance[model_name] = {
                'accuracy': (model_tier_df['score'] >= 0.5).sum() / len(model_tier_df),
                'avg_score': model_tier_df['score'].mean(),
                'avg_cost': model_tier_df['cost'].mean()
            }
        
        metrics.append({
            'tier': tier,
            'total_questions': total_questions,
            'avg_score': avg_score,
            'avg_cost': avg_cost,
            'avg_latency': avg_latency,
            'model_performance': model_performance
        })
    
    return pd.DataFrame(metrics)


def generate_insights(model_metrics: pd.DataFrame, chart_metrics: pd.DataFrame, 
                     tier_metrics: pd.DataFrame) -> List[str]:
    """Generate insights from the analysis."""
    insights = []
    
    # Best performing model
    best_model = model_metrics.loc[model_metrics['accuracy'].idxmax()]
    insights.append(f"Best performing model: {best_model['model_name']} "
                   f"(Accuracy: {best_model['accuracy']:.3f})")
    
    # Most cost-effective model
    cost_effective_model = model_metrics.loc[model_metrics['avg_cost_per_question'].idxmin()]
    insights.append(f"Most cost-effective model: {cost_effective_model['model_name']} "
                   f"(Avg cost per question: ${cost_effective_model['avg_cost_per_question']:.4f})")
    
    # Fastest model
    fastest_model = model_metrics.loc[model_metrics['avg_latency_seconds'].idxmin()]
    insights.append(f"Fastest model: {fastest_model['model_name']} "
                   f"(Avg latency: {fastest_model['avg_latency_seconds']:.2f}s)")
    
    # Tier performance
    tier1_best = tier_metrics[tier_metrics['tier'] == 'tier1_factual']
    tier2_best = tier_metrics[tier_metrics['tier'] == 'tier2_pattern']
    
    if not tier1_best.empty:
        insights.append(f"Tier 1 (Factual) average score: {tier1_best.iloc[0]['avg_score']:.3f}")
    if not tier2_best.empty:
        insights.append(f"Tier 2 (Pattern) average score: {tier2_best.iloc[0]['avg_score']:.3f}")
    
    # Hallucination analysis
    hallucination_leader = model_metrics.loc[model_metrics['hallucination_rate'].idxmax()]
    insights.append(f"Model with highest hallucination rate: {hallucination_leader['model_name']} "
                   f"({hallucination_leader['hallucination_rate']:.3f})")
    
    return insights


def main():
    """Main analysis function."""
    logger = logging.getLogger(__name__)
    setup_logging()
    
    logger.info("Starting results analysis...")
    
    try:
        # Load results
        df = load_latest_results()
        logger.info(f"Loaded {len(df)} evaluation results")
        
        # Calculate metrics
        logger.info("Calculating model metrics...")
        model_metrics = calculate_model_metrics(df)
        
        logger.info("Calculating chart metrics...")
        chart_metrics = calculate_chart_metrics(df)
        
        logger.info("Calculating tier metrics...")
        tier_metrics = calculate_tier_metrics(df)
        
        # Save metrics
        os.makedirs('results', exist_ok=True)
        
        model_metrics.to_csv('results/model_metrics.csv', index=False)
        chart_metrics.to_csv('results/chart_metrics.csv', index=False)
        tier_metrics.to_csv('results/tier_metrics.csv', index=False)
        
        logger.info("Metrics saved to results/ directory")
        
        # Generate insights
        insights = generate_insights(model_metrics, chart_metrics, tier_metrics)
        
        # Print summary
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        
        print("\nModel Performance:")
        print(model_metrics[['model_name', 'accuracy', 'tier1_accuracy', 'tier2_accuracy', 
                           'total_cost', 'avg_latency_seconds']].to_string(index=False))
        
        print("\nKey Insights:")
        for insight in insights:
            print(f"• {insight}")
        
        # Save insights
        with open('results/insights.txt', 'w') as f:
            f.write("Chart Evaluation Analysis Insights\n")
            f.write("="*40 + "\n\n")
            for insight in insights:
                f.write(f"• {insight}\n")
        
        logger.info("Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
