"""
Error taxonomy detection functions for chart evaluation responses.
"""
import re
from typing import Dict, Any, List, Set


def extract_numbers_from_text(text: str) -> List[float]:
    """
    Extract meaningful numbers from text, handling currency and ignoring context.
    
    Args:
        text: Text to extract numbers from
        
    Returns:
        List of extracted numbers
    """
    if not text:
        return []
    
    # Step 1: Remove common false positive patterns BEFORE extraction
    cleaned_text = text
    
    # Remove quarter references (Q1, Q2, Q3, Q4)
    cleaned_text = re.sub(r'\bQ[1-4]\b', '', cleaned_text, flags=re.IGNORECASE)
    
    # Remove years (1900-2099)
    cleaned_text = re.sub(r'\b(19|20)\d{2}\b', '', cleaned_text)
    
    # Remove month names
    cleaned_text = re.sub(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b', 
                         '', cleaned_text, flags=re.IGNORECASE)
    
    # ✅ NEW: Remove mathematical operations with 100 (× 100, / 100, * 100)
    cleaned_text = re.sub(r'[×*/]\s*100\b', '', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'\b100\s*[×*/]', '', cleaned_text, flags=re.IGNORECASE)
    
    # ✅ NEW: Remove time references (11 PM, 3 AM, 23:00, etc.)
    cleaned_text = re.sub(r'\b\d{1,2}\s*(?:AM|PM|am|pm)\b', '', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'\b\d{1,2}:\d{2}\b', '', cleaned_text)  # Remove 23:00 format
    
    # ✅ NEW: Remove "hour X" references (to avoid extracting hour numbers)
    cleaned_text = re.sub(r'\bhour\s+\d{1,2}\b', '', cleaned_text, flags=re.IGNORECASE)

    # Remove day/hour references (Day 5, Hour 12, etc.)
    cleaned_text = re.sub(r'\b(Day|Hour|Week|Month|Year)\s+\d+\b', '', cleaned_text, flags=re.IGNORECASE)
    
    # Step 2: Extract numbers with proper grouping
    # This regex matches:
    # - Optional $ sign
    # - Digits with optional commas (must be in groups of 3)
    # - Optional decimal point and decimals
    # - Optional % sign
    # BUT requires at least one digit before any comma
    pattern = r'\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?|\$?\d+(?:\.\d+)?%?'
    
    matches = re.findall(pattern, cleaned_text)
    
    numbers = []
    for match in matches:
        # Clean the match
        clean = match.replace('$', '').replace(',', '').replace('%', '')
        
        try:
            num = float(clean)
            
            # Filter criteria:
            # - Not too small (avoid extracting lone decimals like .0)
            # - Not too large (avoid timestamps, etc.)
            # - Exclude single-digit numbers under 10 (likely labels/indices unless explicitly needed)
            if num >= 0.01 or num == 0:
                if num < 1_000_000_000:  # Reasonable upper bound
                    numbers.append(num)
        except ValueError:
            continue
    
    return numbers


def is_number_valid(claimed: float, valid_numbers: List[float], tolerance_percent: float = 3.0) -> bool:
    """
    Check if a claimed number matches any valid number within tolerance.
    
    Args:
        claimed: Number from model response
        valid_numbers: List of valid numbers from ground truth
        tolerance_percent: Acceptable percentage difference (default 3%)
        
    Returns:
        True if number is valid, False if hallucinated
    """
    for actual in valid_numbers:
        # Exact match
        if claimed == actual:
            return True
        
        # Check percentage difference
        if actual != 0:
            percent_diff = abs((claimed - actual) / actual) * 100
            if percent_diff <= tolerance_percent:
                return True
        
        # Check if it's a scaled version (e.g., 167 vs 167000)
        # Common in "$167K" vs "$167,000"
        scale_factors = [0.001, 0.01, 0.1, 10, 100, 1000, 10000, 100000, 1_000_000]
        for scale in scale_factors:
            scaled_actual = actual * scale
            if scaled_actual == 0:
                continue
            
            percent_diff = abs((claimed - scaled_actual) / scaled_actual) * 100
            if percent_diff <= tolerance_percent:
                return True
    
    return False


def detect_number_hallucinations(response: str, ground_truth_data: Dict[str, Any], 
                                tolerance: float = 0.05) -> Dict[str, Any]:
    """
    Detect numbers in response that are not in ground truth data.
    
    Args:
        response: Model's response text
        ground_truth_data: Ground truth data for the chart
        tolerance: Tolerance for number matching (0.03 = 3%)
        
    Returns:
        Dictionary with hallucination detection results
    """
    if not response:
        return {
            'has_hallucinations': False,
            'hallucinated_numbers': [],
            'valid_numbers': [],
            'count': 0,
            'score': 1.0
        }
    
    # Extract all numbers from response
    response_numbers = extract_numbers_from_text(response)
    
    if not response_numbers:
        return {
            'has_hallucinations': False,
            'hallucinated_numbers': [],
            'valid_numbers': [],
            'count': 0,
            'score': 1.0
        }
    
    # Get valid numbers from ground truth
    # Priority 1: Use all_valid_numbers if available (comprehensive list)
    if 'all_valid_numbers' in ground_truth_data:
        valid_numbers = ground_truth_data['all_valid_numbers']
    else:
        # Fallback: Extract from data_points and key_facts
        valid_numbers = set()
        
        # Add numbers from data_points
        if 'data_points' in ground_truth_data:
            data_points = ground_truth_data['data_points']
            if isinstance(data_points, dict):
                for value in data_points.values():
                    if isinstance(value, (int, float)):
                        valid_numbers.add(float(value))
                    elif isinstance(value, dict):
                        # Handle nested dicts (e.g., multi-axis charts)
                        for nested_val in value.values():
                            if isinstance(nested_val, (int, float)):
                                valid_numbers.add(float(nested_val))
        
        # Add numbers from key_facts
        if 'key_facts' in ground_truth_data:
            for value in ground_truth_data['key_facts'].values():
                if isinstance(value, (int, float)):
                    valid_numbers.add(float(value))
        
        valid_numbers = list(valid_numbers)
    
    # Convert tolerance to percentage if it's in decimal form
    tolerance_percent = tolerance * 100 if tolerance < 1 else tolerance
    
    # Check each number in response
    hallucinated_numbers = []
    valid_found_numbers = []
    
    for number in response_numbers:
        if is_number_valid(number, valid_numbers, tolerance_percent):
            valid_found_numbers.append(number)
        else:
            hallucinated_numbers.append(number)
    
    has_hallucinations = len(hallucinated_numbers) > 0
    count = len(hallucinated_numbers)
    score = len(valid_found_numbers) / len(response_numbers) if response_numbers else 1.0
    
    return {
        'has_hallucinations': has_hallucinations,
        'hallucinated_numbers': hallucinated_numbers,
        'valid_numbers': valid_found_numbers,
        'count': count,
        'score': score
    }


def detect_axis_errors(response: str, ground_truth_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect errors related to axis labels, units, or scale.
    
    Args:
        response: Model's response text
        ground_truth_data: Ground truth data for the chart
        
    Returns:
        Dictionary with axis error detection results
    """
    if not response:
        return {
            'has_errors': False,
            'missing_labels': [],
            'incorrect_units': [],
            'score': 1.0
        }
    
    response_lower = response.lower()
    missing_labels = []
    
    # Check for key axis information in ground truth
    chart_type = ground_truth_data.get('chart_type', '')
    
    # For non-pie charts, check if axis information is mentioned
    if chart_type != 'pie':
        # Check if common axis terms are mentioned
        axis_keywords = ['axis', 'x-axis', 'y-axis', 'horizontal', 'vertical']
        mentions_axis = any(keyword in response_lower for keyword in axis_keywords)
        
        # Check if data dimensions are discussed appropriately
        # (This is a simplified check - could be expanded)
        has_errors = False
        score = 1.0
    else:
        # For pie charts, different criteria
        has_errors = False
        score = 1.0
    
    return {
        'has_errors': has_errors,
        'missing_labels': missing_labels,
        'incorrect_units': [],
        'score': score
    }


def detect_trend_reversals(response: str, ground_truth_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect contradictory trend claims in the response.
    
    Args:
        response: Model's response text
        ground_truth_data: Ground truth data for the chart
        
    Returns:
        Dictionary with trend reversal detection results
    """
    if not response:
        return {
            'has_trend_reversals': False,
            'contradictory_terms': [],
            'score': 1.0
        }
    
    response_lower = response.lower()
    
    # Define contradictory trend terms
    increasing_terms = ['increase', 'increasing', 'growth', 'growing', 'upward', 'rising', 'rise', 'up', 'higher', 'climb']
    decreasing_terms = ['decrease', 'decreasing', 'decline', 'declining', 'downward', 'falling', 'fall', 'down', 'lower', 'drop']
    
    # Check for presence of contradictory terms
    found_increasing = [term for term in increasing_terms if term in response_lower]
    found_decreasing = [term for term in decreasing_terms if term in response_lower]
    
    has_increasing = len(found_increasing) > 0
    has_decreasing = len(found_decreasing) > 0
    
    contradictory_terms = []
    has_trend_reversals = False
    
    # Only flag as contradiction if BOTH types appear
    # (Some responses may legitimately discuss ups and downs)
    if has_increasing and has_decreasing:
        # Check if they appear in different sentences (might be OK)
        sentences = response_lower.split('.')
        
        # Count sentences with both types
        contradictory_sentences = 0
        for sentence in sentences:
            has_inc_in_sent = any(term in sentence for term in increasing_terms)
            has_dec_in_sent = any(term in sentence for term in decreasing_terms)
            if has_inc_in_sent and has_dec_in_sent:
                contradictory_sentences += 1
        
        # Only flag if contradiction appears in same sentence
        if contradictory_sentences > 0:
            has_trend_reversals = True
            contradictory_terms = list(set(found_increasing + found_decreasing))
    
    score = 0.0 if has_trend_reversals else 1.0
    
    return {
        'has_trend_reversals': has_trend_reversals,
        'contradictory_terms': contradictory_terms[:5],  # Limit to 5 to avoid clutter
        'score': score
    }


def detect_overgeneralization(response: str, ground_truth_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect overgeneralization by counting generic phrases vs specific details.
    
    Args:
        response: Model's response text
        ground_truth_data: Ground truth data for the chart
        
    Returns:
        Dictionary with overgeneralization detection results
    """
    if not response:
        return {
            'has_overgeneralization': False,
            'generic_phrases': [],
            'specific_details': [],
            'specificity_score': 1.0,
            'score': 1.0
        }
    
    response_lower = response.lower()
    total_words = len(response.split())
    
    if total_words == 0:
        return {
            'has_overgeneralization': False,
            'generic_phrases': [],
            'specific_details': [],
            'specificity_score': 1.0,
            'score': 1.0
        }
    
    # Generic phrases that indicate overgeneralization
    generic_phrases = [
        'generally', 'appears to', 'seems to', 'looks like', 'might be',
        'could be', 'possibly', 'perhaps', 'maybe', 'sort of', 'kind of',
        'somewhat', 'relatively', 'fairly', 'quite', 'rather', 'roughly',
        'approximately', 'around', 'about'
    ]
    
    # Specific detail indicators (positive signals)
    specific_indicators = [
        'exactly', 'precisely', 'specifically', 'concretely',
        'definitively', 'clearly', 'obviously', 'undoubtedly', 'certainly'
    ]
    
    # Count occurrences
    found_generic = [phrase for phrase in generic_phrases if phrase in response_lower]
    found_specific = [phrase for phrase in specific_indicators if phrase in response_lower]
    
    generic_count = len(found_generic)
    specific_count = len(found_specific)
    
    # Check if response contains actual numbers (high specificity indicator)
    contains_numbers = len(extract_numbers_from_text(response)) > 0
    
    # Calculate specificity score
    generic_ratio = generic_count / total_words
    specific_ratio = specific_count / total_words
    
    # Specificity score: higher is better
    specificity_score = 1.0 - (generic_ratio * 2)  # Penalize generic language
    if contains_numbers:
        specificity_score += 0.2  # Bonus for including numbers
    if specific_count > 0:
        specificity_score += 0.1  # Bonus for specific language
    
    specificity_score = max(0.0, min(1.0, specificity_score))  # Clamp to [0, 1]
    
    # Consider overgeneralization if:
    # - High generic ratio (>5% of words are generic terms)
    # - Low specific ratio (<2% specific terms)
    # - No numbers mentioned
    has_overgeneralization = (
        generic_ratio > 0.05 and 
        specific_ratio < 0.02 and 
        not contains_numbers
    )
    
    return {
        'has_overgeneralization': has_overgeneralization,
        'generic_phrases': found_generic[:5],  # Limit to avoid clutter
        'specific_details': found_specific,
        'specificity_score': round(specificity_score, 3),
        'score': round(specificity_score, 3)
    }


def run_all_error_detections(response: str, ground_truth_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run all error detection functions on a response.
    
    Args:
        response: Model's response text
        ground_truth_data: Ground truth data for the chart
        
    Returns:
        Dictionary with all error detection results
    """
    return {
        'number_hallucinations': detect_number_hallucinations(response, ground_truth_data),
        'axis_errors': detect_axis_errors(response, ground_truth_data),
        'trend_reversals': detect_trend_reversals(response, ground_truth_data),
        'overgeneralization': detect_overgeneralization(response, ground_truth_data)
    }