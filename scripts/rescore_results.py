"""
Re-score existing evaluation results without calling APIs.
Useful when scoring logic is updated.
"""
import os
import sys
import json
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.evaluation.questions import load_ground_truth, get_all_questions
from src.evaluation.scoring import score_answer
from src.evaluation.error_taxonomy import run_all_error_detections


def setup_logging():
    """Setup logging configuration."""
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/rescoring.log'),
            logging.StreamHandler()
        ]
    )


def load_latest_results() -> pd.DataFrame:
    """Load the most recent results file."""
    results_dir = 'results'
    
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    # Find all scored results files (excluding backups and rescored)
    parquet_files = [
        f for f in os.listdir(results_dir) 
        if f.endswith('.parquet') 
        and f.startswith('scored_results')
        and 'backup' not in f
        and 'rescored' not in f
    ]
    
    if not parquet_files:
        raise FileNotFoundError("No scored_results parquet files found in results/")
    
    # Sort by modification time to get the most recent
    parquet_files_with_time = [
        (f, os.path.getmtime(os.path.join(results_dir, f))) 
        for f in parquet_files
    ]
    latest_file = max(parquet_files_with_time, key=lambda x: x[1])[0]
    filepath = os.path.join(results_dir, latest_file)
    
    print(f"Loading results from: {filepath}")
    df = pd.read_parquet(filepath)
    print(f"  Charts in file: {sorted(df['chart_id'].unique())}")
    print(f"  Total rows: {len(df)}")
    return df


def rescore_results(df: pd.DataFrame, ground_truth: Dict[str, Any]) -> pd.DataFrame:
    """
    Re-score all results using updated scoring functions.
    
    Args:
        df: DataFrame with existing results
        ground_truth: Ground truth data
        
    Returns:
        DataFrame with updated scores and error flags
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Re-scoring {len(df)} results...")
    
    # Create a copy to avoid modifying original
    updated_df = df.copy()
    
    # Build question lookup for faster access
    all_questions = get_all_questions(ground_truth)
    question_lookup = {
        (q['chart_id'], q['question_id']): q 
        for q in all_questions
    }
    
    # Log which charts are in ground truth vs results
    charts_in_results = set(df['chart_id'].unique())
    charts_in_ground_truth = set(ground_truth.keys())
    
    logger.info(f"\nCharts in results file: {sorted(charts_in_results)}")
    logger.info(f"Charts in ground truth: {sorted(charts_in_ground_truth)}")
    
    missing_in_results = charts_in_ground_truth - charts_in_results
    if missing_in_results:
        logger.warning(f"Charts in ground truth but not in results: {sorted(missing_in_results)}")
        logger.warning("These charts will be skipped during rescoring.")
    
    # Re-score each row
    for idx, row in updated_df.iterrows():
        chart_id = row['chart_id']
        question_id = row['question_id']
        response = row['model_response']
        
        # Get question data
        question_key = (chart_id, question_id)
        if question_key not in question_lookup:
            logger.warning(f"Question not found: {chart_id} / {question_id}")
            continue
        
        question = question_lookup[question_key]

        # In the rescore loop:
        if question_id == 'q4' and chart_id == 'chart_007':
            logger.info(f"\n🔍 DEBUG Chart 7 Q4:")
            logger.info(f"  Question: {question}")
            logger.info(f"  Ground truth: '{question.get('ground_truth')}'")
            logger.info(f"  Ground truth type: {type(question.get('ground_truth'))}")
            logger.info(f"  Answer: '{question.get('answer')}'")
            logger.info(f"  Question type: {question.get('type')}")
            logger.info(f"  Response: '{response[:150]}'")
            
            # Test the scoring directly
            from src.evaluation.scoring import score_categorical_answer
            test_score = score_categorical_answer(response, question.get('ground_truth'))
            logger.info(f"  Direct categorical score: {test_score}")
        
        # ✅ ADD THIS DEBUG FOR Q3
        if question_id == 'q3':
            logger.info(f"\n🔍 DEBUG Q3:")
            logger.info(f"  Chart: {chart_id}")
            logger.info(f"  Question dict: {question}")
            logger.info(f"  Type: {question.get('type')}")
            logger.info(f"  Keywords: {question.get('keywords')}")
            logger.info(f"  Response (first 100 chars): {response[:100]}")
        
        # Re-score
        new_score = score_answer(response, question)
        updated_df.at[idx, 'score'] = new_score
        
        # Update ground truth in DataFrame with the new value from ground_truth.json
        new_ground_truth = question.get('answer') or question.get('ground_truth')
        if new_ground_truth is not None:
            # Always convert to string for Parquet compatibility (handles all types)
            updated_df.at[idx, 'ground_truth'] = str(new_ground_truth)
        
        # Debug for chart_012 q2
        if chart_id == 'chart_012' and question_id == 'q2':
            logger.info(f"\n🔍 DEBUG Chart 12 Q2:")
            logger.info(f"  Question type: {question.get('type')}")
            logger.info(f"  Ground truth: {question.get('answer')}")
            logger.info(f"  Tolerance: {question.get('tolerance')}")
            logger.info(f"  Response: {response[:200]}")
            logger.info(f"  Old score: {row['score']}")
            logger.info(f"  New score: {new_score}")
            logger.info(f"  Old ground truth: {row['ground_truth']}")
            logger.info(f"  New ground truth: {new_ground_truth}\n")
        
        # ✅ ADD THIS DEBUG TOO
        if question_id == 'q3':
            logger.info(f"  New score: {new_score}\n")
        
        # Re-run error detection with full ground truth
        chart_ground_truth = ground_truth.get(chart_id, {})
        ground_truth_data = {
            'data_points': chart_ground_truth.get('data_points', {}),
            'key_facts': chart_ground_truth.get('key_facts', {}),
            'all_valid_numbers': chart_ground_truth.get('all_valid_numbers', [])
        }
        
        new_error_flags = run_all_error_detections(response, ground_truth_data)
        
        # Convert to JSON string for Parquet compatibility
        error_flags_json = json.dumps(new_error_flags) if new_error_flags else "{}"
        updated_df.at[idx, 'error_flags'] = error_flags_json
        
        # Log significant changes
        old_score = row['score']
        if abs(new_score - old_score) > 0.1:
            logger.info(f"Score change for {chart_id}/{question_id}: {old_score:.3f} → {new_score:.3f}")
    
    return updated_df



def print_comparison_statistics(old_df: pd.DataFrame, new_df: pd.DataFrame):
    """Print comparison statistics between old and new scores."""
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*60)
    logger.info("RE-SCORING COMPARISON")
    logger.info("="*60)
    
    # Overall statistics
    old_mean = old_df['score'].mean()
    new_mean = new_df['score'].mean()
    
    logger.info(f"\nOverall Average Score:")
    logger.info(f"  Before: {old_mean:.3f}")
    logger.info(f"  After:  {new_mean:.3f}")
    logger.info(f"  Change: {(new_mean - old_mean):+.3f}")
    
    # By model
    logger.info("\nAverage Score by Model:")
    for model_name in old_df['model_name'].unique():
        old_model_mean = old_df[old_df['model_name'] == model_name]['score'].mean()
        new_model_mean = new_df[new_df['model_name'] == model_name]['score'].mean()
        
        logger.info(f"  {model_name}:")
        logger.info(f"    Before: {old_model_mean:.3f}")
        logger.info(f"    After:  {new_model_mean:.3f}")
        logger.info(f"    Change: {(new_model_mean - old_model_mean):+.3f}")
    
    # By tier
    logger.info("\nAverage Score by Tier:")
    for tier in old_df['question_tier'].unique():
        old_tier_mean = old_df[old_df['question_tier'] == tier]['score'].mean()
        new_tier_mean = new_df[new_df['question_tier'] == tier]['score'].mean()
        
        logger.info(f"  {tier}:")
        logger.info(f"    Before: {old_tier_mean:.3f}")
        logger.info(f"    After:  {new_tier_mean:.3f}")
        logger.info(f"    Change: {(new_tier_mean - old_tier_mean):+.3f}")
    
    # Count significant changes
    score_diff = (new_df['score'] - old_df['score']).abs()
    significant_changes = (score_diff > 0.1).sum()
    
    logger.info(f"\nResults with significant score changes (>0.1): {significant_changes}/{len(old_df)}")


def main():
    """Main re-scoring pipeline."""
    logger = logging.getLogger(__name__)
    
    # Setup logging
    setup_logging()
    
    logger.info("Starting re-scoring pipeline...")
    
    try:
        # Load existing results
        logger.info("Loading existing results...")
        old_df = load_latest_results()
        
        # Load ground truth
        logger.info("Loading ground truth data...")
        ground_truth = load_ground_truth()
        
        # Re-score
        logger.info("Re-scoring all results...")
        new_df = rescore_results(old_df, ground_truth)
        
        # Print comparison
        print_comparison_statistics(old_df, new_df)
        
        # Save updated results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"results/scored_results_rescored_{timestamp}.parquet"
        new_df.to_parquet(output_filename, index=False)
        logger.info(f"\nUpdated results saved to: {output_filename}")
        
        # Also create a backup of the original
        backup_filename = f"results/scored_results_backup_{timestamp}.parquet"
        old_df.to_parquet(backup_filename, index=False)
        logger.info(f"Original results backed up to: {backup_filename}")
        
        logger.info("\n✅ Re-scoring completed successfully!")
        
    except Exception as e:
        logger.error(f"Re-scoring failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()