"""
Question generation and prompt creation for chart evaluation.
"""
import json
from typing import Dict, Any, List


def load_ground_truth(ground_truth_path: str = "data/ground_truth.json") -> Dict[str, Any]:
    """
    Load ground truth data from JSON file.
    
    Args:
        ground_truth_path: Path to ground truth JSON file
        
    Returns:
        Dictionary containing ground truth data for all charts
    """
    with open(ground_truth_path, 'r') as f:
        return json.load(f)


def generate_prompt(chart_id: str, question: Dict[str, Any]) -> str:
    """
    Generate a prompt for chart evaluation.
    
    Args:
        chart_id: Identifier for the chart
        question: Question dictionary with text and metadata
        
    Returns:
        Formatted prompt string
    """
    return f"""Analyze this business chart and answer the following question precisely.

Question: {question['question_text']}

Provide a direct, specific answer based only on what you see in the chart. Include numbers where relevant. Be concise and factual."""


def get_all_questions(ground_truth: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract all questions from ground truth data.
    
    Args:
        ground_truth: Ground truth dictionary
        
    Returns:
        List of question dictionaries with chart_id and question info
    """
    all_questions = []
    
    for chart_id, chart_data in ground_truth.items():
        # Add Tier 1 (factual) questions
        for question in chart_data['questions']['tier1_factual']:
            all_questions.append({
                'chart_id': chart_id,
                'question_id': question['id'],
                'question_text': question['text'],
                'question_tier': 'tier1_factual',
                'ground_truth': question['answer'],
                'answer' : question['answer'],
                'type' : question.get('type', 'categorical'),
                'question_type': question.get('type', 'categorical'),
                'tolerance': question.get('tolerance', 0.03),
                'keywords': question.get('keywords', [])
            })
        
        # Add Tier 2 (pattern) questions
        for question in chart_data['questions']['tier2_pattern']:
            all_questions.append({
                'chart_id': chart_id,
                'question_id': question['id'],
                'question_text': question['text'],
                'question_tier': 'tier2_pattern',
                'ground_truth': question['answer'],
                'answer' : question['answer'],
                'type' : question.get('type', 'categorical'),
                'question_type': question.get('type', 'categorical'),
                'tolerance': question.get('tolerance', 0.03),
                'keywords': question.get('keywords', [])
            })
    
    return all_questions


def get_questions_for_chart(chart_id: str, ground_truth: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Get all questions for a specific chart.
    
    Args:
        chart_id: Chart identifier
        ground_truth: Ground truth dictionary
        
    Returns:
        List of questions for the specified chart
    """
    if chart_id not in ground_truth:
        return []
    
    chart_data = ground_truth[chart_id]
    questions = []
    
    # Add Tier 1 questions
    for question in chart_data['questions']['tier1_factual']:
        questions.append({
            'chart_id': chart_id,
            'question_id': question['id'],
            'question_text': question['text'],
            'question_tier': 'tier1_factual',
            'ground_truth': question['answer'],
            'question_type': question['type'],
            'tolerance': question.get('tolerance', 0.01),
            'keywords': question.get('keywords', [])
        })
    
    # Add Tier 2 questions
    for question in chart_data['questions']['tier2_pattern']:
        questions.append({
            'chart_id': chart_id,
            'question_id': question['id'],
            'question_text': question['text'],
            'question_tier': 'tier2_pattern',
            'ground_truth': question['answer'],
            'question_type': question.get('type', 'categorical'),
            'tolerance': question.get('tolerance', 0.01),
            'keywords': question.get('keywords', [])
        })
    
    return questions


def get_chart_metadata(chart_id: str, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get metadata for a specific chart.
    
    Args:
        chart_id: Chart identifier
        ground_truth: Ground truth dictionary
        
    Returns:
        Chart metadata dictionary
    """
    if chart_id not in ground_truth:
        return {}
    
    chart_data = ground_truth[chart_id]
    return {
        'chart_id': chart_id,
        'chart_type': chart_data['chart_type'],
        'category': chart_data['category'],
        'difficulty': chart_data['difficulty'],
        'data_points': chart_data['data_points'],
        'key_facts': chart_data['key_facts']
    }


def validate_question_format(question: Dict[str, Any]) -> bool:
    """
    Validate that a question has the required format.
    
    Args:
        question: Question dictionary to validate
        
    Returns:
        True if question format is valid, False otherwise
    """
    required_fields = ['chart_id', 'question_id', 'question_text', 'question_tier', 
                      'ground_truth', 'question_type']
    
    return all(field in question for field in required_fields)
