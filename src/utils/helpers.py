"""
Utility functions for image processing and data handling.
"""
import base64
import os
from typing import Union
from PIL import Image
import numpy as np


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode an image file to base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string of the image
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image format is not supported
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    except Exception as e:
        raise ValueError(f"Failed to encode image: {str(e)}")


def validate_image_format(image_path: str) -> bool:
    """
    Validate that the image is in a supported format.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        True if image format is supported, False otherwise
    """
    try:
        with Image.open(image_path) as img:
            return img.format.lower() in ['png', 'jpg', 'jpeg', 'gif', 'bmp']
    except Exception:
        return False


def resize_image_if_needed(image_path: str, max_width: int = 2048, max_height: int = 2048) -> str:
    """
    Resize image if it exceeds maximum dimensions.
    
    Args:
        image_path: Path to the image file
        max_width: Maximum allowed width
        max_height: Maximum allowed height
        
    Returns:
        Path to the resized image (may be same as input if no resize needed)
    """
    try:
        with Image.open(image_path) as img:
            if img.width <= max_width and img.height <= max_height:
                return image_path
            
            # Calculate new dimensions maintaining aspect ratio
            ratio = min(max_width / img.width, max_height / img.height)
            new_width = int(img.width * ratio)
            new_height = int(img.height * ratio)
            
            # Resize image
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Save resized image
            resized_path = image_path.replace('.png', '_resized.png')
            resized_img.save(resized_path)
            
            return resized_path
    except Exception as e:
        print(f"Warning: Could not resize image {image_path}: {str(e)}")
        return image_path


def extract_numbers_from_text(text: str) -> list[float]:
    """
    Extract all numeric values from text.
    
    Args:
        text: Input text to extract numbers from
        
    Returns:
        List of extracted numbers as floats
    """
    import re
    
    # Pattern to match numbers (including decimals and scientific notation)
    number_pattern = r'-?\d+\.?\d*(?:[eE][+-]?\d+)?'
    matches = re.findall(number_pattern, text)
    
    numbers = []
    for match in matches:
        try:
            numbers.append(float(match))
        except ValueError:
            continue
    
    return numbers


def calculate_percentage_difference(value1: float, value2: float) -> float:
    """
    Calculate percentage difference between two values.
    
    Args:
        value1: First value
        value2: Second value
        
    Returns:
        Percentage difference
    """
    if value2 == 0:
        return float('inf') if value1 != 0 else 0.0
    
    return abs((value1 - value2) / value2) * 100
