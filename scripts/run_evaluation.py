"""
Main evaluation pipeline with budget enforcement and comprehensive logging.
"""
import os
import sys
import json
import logging
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
from tqdm import tqdm
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.models.claude import ClaudeModel
from src.models.gpt import GPTModel
from src.models.gemini import GeminiModel
from src.evaluation.questions import load_ground_truth, get_all_questions, generate_prompt
from src.evaluation.scoring import score_answer
from src.evaluation.error_taxonomy import run_all_error_detections
from src.utils.cost_tracker import CostTracker, BudgetExceededError
from src.storage.s3_handler import S3Handler


def setup_logging():
    """Setup logging configuration."""
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/evaluation.log'),
            logging.StreamHandler()
        ]
    )


def initialize_models():
    """Initialize all model instances."""
    logger = logging.getLogger(__name__)
    
    models = {}
    
    try:
        models['claude'] = ClaudeModel()
        logger.info("Claude model initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Claude: {str(e)}")
    
    try:
        models['gpt'] = GPTModel()
        logger.info("GPT-5 model initialized")
    except Exception as e:
        logger.error(f"Failed to initialize GPT-5: {str(e)}")
    
    try:
        models['gemini'] = GeminiModel()
        logger.info("Gemini model initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini: {str(e)}")
    
    return models


def evaluate_single_question(model, model_name: str, chart_id: str, question: Dict[str, Any], 
                            cost_tracker: CostTracker, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate a single question with a single model.
    
    Args:
        model: Model instance
        model_name: Name of the model
        chart_id: Chart identifier
        question: Question dictionary
        cost_tracker: Cost tracking instance
        
    Returns:
        Dictionary with evaluation results
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Generate prompt
        prompt = generate_prompt(chart_id, question)
        
        # Get image path
        image_path = f"data/charts/{chart_id}.png"
        
        # Call model
        result = model.evaluate_chart(image_path, prompt)
        
        if result['error']:
            logger.error(f"Model {model_name} error for {chart_id}: {result['error']}")
            return {
                'chart_id': chart_id,
                'model_name': model_name,
                'question_id': question['question_id'],
                'question_text': question['question_text'],
                'question_tier': question['question_tier'],
                'model_response': '',
                'ground_truth': str(question['ground_truth']),
                'score': 0.0,
                'input_tokens': 0,
                'output_tokens': 0,
                'cost': 0.0,
                'latency_seconds': result['latency_seconds'],
                'error': result['error'],
                'error_flags': "{}"
            }
        
        # Add cost to tracker
        timestamp = datetime.now().isoformat()
        cost_tracker.add_cost(
            model_name, 
            result['input_tokens'], 
            result['output_tokens'], 
            result['cost'], 
            timestamp
        )
        
        # Score the response
        score = score_answer(result['response'], question)

        # Run error detection - need to get full ground truth for the chart
        chart_ground_truth = ground_truth.get(chart_id, {})
        ground_truth_data = {
            'data_points': chart_ground_truth.get('data_points', {}),
            'key_facts': chart_ground_truth.get('key_facts', {}),
            'all_valid_numbers': chart_ground_truth.get('all_valid_numbers', [])
        }
        error_flags = run_all_error_detections(result['response'], ground_truth_data)
        
        # Convert error_flags to JSON string for Parquet compatibility
        import json
        error_flags_json = json.dumps(error_flags) if error_flags else "{}"
        
        return {
            'chart_id': chart_id,
            'model_name': model_name,
            'question_id': question['question_id'],
            'question_text': question['question_text'],
            'question_tier': question['question_tier'],
            'model_response': result['response'],
            'ground_truth': str(question['ground_truth']),
            'score': score,
            'input_tokens': result['input_tokens'],
            'output_tokens': result['output_tokens'],
            'cost': result['cost'],
            'latency_seconds': result['latency_seconds'],
            'error': None,
            'error_flags': error_flags_json
        }
        
    except BudgetExceededError as e:
        logger.error(f"Budget exceeded for {model_name}: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error evaluating {model_name} on {chart_id}: {str(e)}")
        return {
            'chart_id': chart_id,
            'model_name': model_name,
            'question_id': question['question_id'],
            'question_text': question['question_text'],
            'question_tier': question['question_tier'],
            'model_response': '',
            'ground_truth': str(question['ground_truth']),
            'score': 0.0,
            'input_tokens': 0,
            'output_tokens': 0,
            'cost': 0.0,
            'latency_seconds': 0.0,
            'error': str(e),
            'error_flags': "{}"
        }


def save_intermediate_results(results: List[Dict[str, Any]], iteration: int):
    """Save intermediate results for resumption."""
    logger = logging.getLogger(__name__)
    
    try:
        df = pd.DataFrame(results)
        filename = f"results/intermediate_results_{iteration}.parquet"
        os.makedirs('results', exist_ok=True)
        df.to_parquet(filename, index=False)
        logger.info(f"Saved intermediate results: {filename}")
    except Exception as e:
        logger.error(f"Failed to save intermediate results: {str(e)}")


def run_evaluation():
    """Main evaluation pipeline."""
    logger = logging.getLogger(__name__)
    
    # Load environment variables
    load_dotenv()
    
    # Setup logging
    setup_logging()
    
    logger.info("Starting chart evaluation pipeline...")
    
    # Initialize components
    cost_tracker = CostTracker()
    s3_handler = S3Handler()
    
    # Load ground truth
    logger.info("Loading ground truth data...")
    ground_truth = load_ground_truth()
    all_questions = get_all_questions(ground_truth)
    
    logger.info(f"Loaded {len(ground_truth)} charts with {len(all_questions)} total questions")
    
    # Initialize models
    logger.info("Initializing models...")
    models = initialize_models()
    
    if not models:
        logger.error("No models initialized successfully. Exiting.")
        return
    
    logger.info(f"Initialized {len(models)} models: {list(models.keys())}")
    
    # Run evaluation
    results = []
    total_evaluations = len(all_questions) * len(models)
    
    logger.info(f"Starting evaluation of {total_evaluations} model-question pairs...")
    
    with tqdm(total=total_evaluations, desc="Evaluating") as pbar:
        for question in all_questions:
            chart_id = question['chart_id']
            
            for model_name, model in models.items():
                try:
                    # Check budget before proceeding
                    cost_summary = cost_tracker.get_cost_summary()
                    if cost_summary['remaining_budget'] <= 0:
                        logger.error("Budget exhausted. Stopping evaluation.")
                        break
                    
                    # Evaluate question
                    result = evaluate_single_question(
                        model, model_name, chart_id, question, cost_tracker, ground_truth
                    )
                    results.append(result)
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'Cost': f"${cost_tracker.get_total_cost():.2f}",
                        'Chart': chart_id,
                        'Model': model_name
                    })
                    
                    # Save intermediate results every 10 evaluations
                    if len(results) % 10 == 0:
                        save_intermediate_results(results, len(results))
                    
                except BudgetExceededError:
                    logger.error(f"Budget exceeded. Stopping evaluation.")
                    break
                except Exception as e:
                    logger.error(f"Error evaluating {model_name} on {chart_id}: {str(e)}")
                    pbar.update(1)
                    continue
            
            # Break outer loop if budget exceeded
            if cost_summary['remaining_budget'] <= 0:
                break
    
    # Convert results to DataFrame
    logger.info("Converting results to DataFrame...")
    df = pd.DataFrame(results)
    
    # Save results locally
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_filename = f"results/scored_results_{timestamp}.parquet"
    os.makedirs('results', exist_ok=True)
    df.to_parquet(local_filename, index=False)
    logger.info(f"Results saved locally: {local_filename}")
    
    # Upload to S3
    try:
        s3_key = s3_handler.upload_results(df, timestamp)
        logger.info(f"Results uploaded to S3: {s3_key}")
    except Exception as e:
        logger.error(f"Failed to upload to S3: {str(e)}")
    
    # Print summary statistics
    print_summary_statistics(df, cost_tracker)
    
    logger.info("Evaluation pipeline completed!")


def print_summary_statistics(df: pd.DataFrame, cost_tracker: CostTracker):
    """Print summary statistics."""
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*60)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*60)
    
    # Cost summary
    cost_summary = cost_tracker.get_cost_summary()
    logger.info(f"Total Cost: ${cost_summary['total_cost']:.2f}")
    logger.info(f"Budget Utilization: {cost_summary['budget_utilization']:.1f}%")
    logger.info(f"Number of Requests: {cost_summary['num_requests']}")
    
    # Model performance
    logger.info("\nModel Performance:")
    for model_name in df['model_name'].unique():
        model_df = df[df['model_name'] == model_name]
        avg_score = model_df['score'].mean()
        avg_cost = model_df['cost'].mean()
        avg_latency = model_df['latency_seconds'].mean()
        
        logger.info(f"  {model_name}:")
        logger.info(f"    Average Score: {avg_score:.3f}")
        logger.info(f"    Average Cost: ${avg_cost:.4f}")
        logger.info(f"    Average Latency: {avg_latency:.2f}s")
    
    # Tier performance
    logger.info("\nTier Performance:")
    for tier in df['question_tier'].unique():
        tier_df = df[df['question_tier'] == tier]
        avg_score = tier_df['score'].mean()
        logger.info(f"  {tier}: {avg_score:.3f}")
    
    # Error analysis
    logger.info("\nError Analysis:")
    error_columns = [col for col in df.columns if col.startswith('error_flags')]
    for col in error_columns:
        error_count = df[col].apply(lambda x: x.get('has_hallucinations', False) if isinstance(x, dict) else False).sum()
        logger.info(f"  {col}: {error_count} occurrences")


if __name__ == "__main__":
    run_evaluation()
