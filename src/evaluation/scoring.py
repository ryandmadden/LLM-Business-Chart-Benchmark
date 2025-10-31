"""
Automated scoring functions for chart evaluation responses.
"""
import re
from typing import Union, List, Dict, Any


def extract_numbers_from_response(text: str) -> List[float]:
    """Extract numbers from text, handling currency formatting."""
    if not text:
        return []
    
    # Remove common context words first
    cleaned = text
    cleaned = cleaned.replace('\u2212', '-')
    cleaned = re.sub(r'\bQ[1-4]\b', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\b(19|20)\d{2}\b', '', cleaned)  # Remove years
    
    # Extract numbers with proper grouping, including negative numbers
    # Matches: -0.99, $167,000, 167000, 167.5, etc.
    pattern = r'-?\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?|-?\$?\d+(?:\.\d+)?%?'
    matches = re.findall(pattern, cleaned)
    
    numbers = []
    for match in matches:
        clean = match.replace('$', '').replace(',', '').replace('%', '')
        try:
            num = float(clean)
            # Keep reasonable numbers (avoid stray decimals, but include negatives)
            if (num >= 0.01 or num == 0 or num <= -0.01):
                if -1_000_000_000 < num < 1_000_000_000:
                    numbers.append(num)
        except ValueError:
            continue
    
    return numbers


def is_number_match(extracted: float, expected: float, tolerance_percent: float = 3.0) -> bool:
    """
    Check if extracted number matches expected within tolerance.
    Handles scaled versions (e.g., 167 vs 167000).
    """
    # Exact match
    if extracted == expected:
        return True
    
    # Percentage difference
    if expected != 0:
        percent_diff = abs((extracted - expected) / expected) * 100
        if percent_diff <= tolerance_percent:
            return True
    
    # Check scaled versions
    scale_factors = [0.001, 0.01, 0.1, 10, 100, 1000, 10000, 100000, 1_000_000]
    for scale in scale_factors:
        scaled = expected * scale
        if scaled == 0:
            continue
        
        percent_diff = abs((extracted - scaled) / scaled) * 100
        if percent_diff <= tolerance_percent:
            return True
    
    return False


def score_numeric_answer(response: str, expected: Union[int, float], tolerance: float = 0.03) -> float:
    """
    Score a numeric answer by extracting numbers from response and checking tolerance.
    
    Args:
        response: Model's text response
        expected: Expected numeric answer
        tolerance: Tolerance as decimal (0.03 = 3%)
        
    Returns:
        Score between 0 and 1 (1 = correct, 0 = incorrect)
    """
    if not response or not isinstance(expected, (int, float)):
        return 0.0
    
    # Extract all numbers from response
    numbers = extract_numbers_from_response(response)
    
    if not numbers:
        return 0.0
    
    # Convert tolerance to percentage
    tolerance_percent = tolerance * 100 if tolerance < 1 else tolerance
    
    # Check if any extracted number matches expected
    for number in numbers:
        if is_number_match(number, float(expected), tolerance_percent):
            return 1.0
    
    return 0.0


def score_categorical_answer(response: str, expected: str) -> float:
    """
    Score a categorical answer using case-insensitive substring matching.
    
    Args:
        response: Model's text response
        expected: Expected categorical answer
        
    Returns:
        Score between 0 and 1 (1 = correct, 0 = incorrect)
    """
    if not response or not expected:
        return 0.0
    
    # Convert to lowercase for case-insensitive matching
    response_lower = response.lower().strip()
    expected_lower = expected.lower().strip()
    
    # Check for exact match
    if expected_lower in response_lower:
        return 1.0
        
    if expected_lower.startswith('h') and expected_lower[1:].isdigit():
        hour_num = expected_lower[1:]  # Extract "23" from "H23"
        
        # Check for various formats:
        # "hour 23", "23rd hour", "at 23", "hour23"
        hour_patterns = [
            f"hour {hour_num}",
            f"hour{hour_num}",
            f"at {hour_num}",
            f"h{hour_num}",
            f"{hour_num}:00",
            f"at hour {hour_num}"
        ]
        
        for pattern in hour_patterns:
            if pattern in response_lower:
                return 1.0

    # Check for partial matches (useful for longer expected answers)
    words_expected = expected_lower.split()
    words_response = response_lower.split()
    
    # If all words from expected are in response, consider it correct
    if all(word in response_lower for word in words_expected if len(word) > 2):
        return 1.0
    
    return 0.0


def score_trend_detection(response: str, keywords: List[str]) -> float:
    """
    Score trend detection based on keyword presence.
    
    Args:
        response: Model's text response
        keywords: List of keywords that indicate correct trend
        
    Returns:
        Score between 0 and 1 based on keyword presence
    """
    if not response or not keywords:
        return 0.0
    
    response_lower = response.lower()
    response_words = response_lower.split()
    
    # Check if ANY keyword is present (flexible matching)
    for keyword in keywords:
        keyword_lower = keyword.lower()
        
        # Method 1: Exact substring match (original)
        if keyword_lower in response_lower:
            return 1.0
        
        # Method 2: Check if keyword is stem of any word in response
        # e.g., "growth" matches "growing", "grow", "grows"
        for word in response_words:
            # Remove common suffixes
            word_stem = word.rstrip('s').rstrip('ing').rstrip('ed').rstrip('es')
            keyword_stem = keyword_lower.rstrip('s').rstrip('ing').rstrip('ed').rstrip('es')
            
            if word_stem == keyword_stem:
                return 1.0
            
            # Also check reverse: if word contains keyword stem
            if len(keyword_stem) >= 4 and keyword_stem in word_stem:
                return 1.0
    
    # Special case: For "uneven" distribution, check if response lists multiple different percentage values
    # This indicates understanding of uneven distribution even without explicit keywords
    if any(kw in ['uneven', 'varied', 'imbalanced', 'mixed'] for kw in keywords):
        percentages = re.findall(r'(\d+)%', response)
        if len(percentages) >= 3:
            # Convert to integers and check if they're actually different
            pct_values = [int(p) for p in percentages]
            unique_values = set(pct_values)
            # If we have at least 3 different percentage values, it shows uneven distribution
            if len(unique_values) >= 3:
                # Check if there's significant variance (not all close together)
                if max(pct_values) - min(pct_values) >= 20:
                    return 1.0
    
    return 0.0


def score_answer(response: str, question: Dict[str, Any]) -> float:
    """
    Score a response based on question type and ground truth.
    
    Args:
        response: Model's response text
        question: Question dictionary with ground truth and metadata
        
    Returns:
        Score between 0 and 1
    """
    if not response:
        return 0.0
    
    # Try multiple possible field names for question type
    question_type = question.get('type') or question.get('question_type', 'categorical')
    
    # Try multiple possible field names for ground truth
    ground_truth = question.get('answer') or question.get('ground_truth')
    
    tolerance = question.get('tolerance', 0.03)
    keywords = question.get('keywords', [])
    
    # ✅ Check keywords FIRST (even if type is wrong)
    if keywords:  # If keywords present, use keyword matching
        return score_trend_detection(response, keywords)
    elif question_type == 'numeric':
        return score_numeric_answer(response, ground_truth, tolerance)
    elif question_type == 'trend':
        # Shouldn't reach here if keywords were provided, but fallback
        return score_categorical_answer(response, ground_truth)
    else:
        return score_categorical_answer(response, ground_truth)


def extract_answer_from_response(response: str, question_type: str) -> Union[str, float, None]:
    """
    Extract the most likely answer from a response based on question type.
    
    Args:
        response: Model's response text
        question_type: Type of question (numeric, categorical, trend)
        
    Returns:
        Extracted answer or None if not found
    """
    if not response:
        return None
    
    if question_type == 'numeric':
        numbers = extract_numbers_from_response(response)
        return numbers[0] if numbers else None
    else:
        return response.strip()


def calculate_accuracy_score(scores: List[float]) -> Dict[str, float]:
    """
    Calculate accuracy metrics from a list of scores.
    
    Args:
        scores: List of scores (0-1)
        
    Returns:
        Dictionary with accuracy metrics
    """
    if not scores:
        return {
            'accuracy': 0.0,
            'num_correct': 0,
            'num_total': 0,
            'average_score': 0.0
        }
    
    num_correct = sum(1 for score in scores if score >= 0.5)
    num_total = len(scores)
    accuracy = num_correct / num_total
    average_score = sum(scores) / num_total
    
    return {
        'accuracy': accuracy,
        'num_correct': num_correct,
        'num_total': num_total,
        'average_score': average_score
    }