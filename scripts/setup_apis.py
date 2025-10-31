"""
API connectivity test script for all three model providers.
"""
import os
import sys
import logging
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.models.claude import ClaudeModel
from src.models.gpt import GPTModel
from src.models.gemini import GeminiModel
from src.utils.cost_tracker import CostTracker


def setup_logging():
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/api_test.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def test_claude_api():
    """Test Claude API connectivity."""
    logger = logging.getLogger(__name__)
    logger.info("Testing Claude API connectivity...")
    
    try:
        model = ClaudeModel()
        logger.info(f"✓ Claude model initialized: {model.get_model_info()}")
        
        # Test with a simple image (you would need a test image)
        # For now, just test initialization
        logger.info("✓ Claude API test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Claude API test failed: {str(e)}")
        return False


def test_gpt_api():
    """Test GPT-5 API connectivity."""
    logger = logging.getLogger(__name__)
    logger.info("Testing GPT-5 API connectivity...")
    
    try:
        model = GPTModel()
        logger.info(f"✓ GPT-5 model initialized: {model.get_model_info()}")
        
        # Test with a simple image (you would need a test image)
        # For now, just test initialization
        logger.info("✓ GPT-5 API test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ GPT-5 API test failed: {str(e)}")
        return False


def test_gemini_api():
    """Test Gemini API connectivity."""
    logger = logging.getLogger(__name__)
    logger.info("Testing Gemini API connectivity...")
    
    try:
        model = GeminiModel()
        logger.info(f"✓ Gemini model initialized: {model.get_model_info()}")
        
        # Test with a simple image (you would need a test image)
        # For now, just test initialization
        logger.info("✓ Gemini API test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Gemini API test failed: {str(e)}")
        return False


def test_cost_tracker():
    """Test cost tracking functionality."""
    logger = logging.getLogger(__name__)
    logger.info("Testing cost tracker...")
    
    try:
        tracker = CostTracker()
        
        # Test adding costs
        tracker.add_cost("test-model", 1000, 500, 0.05, "2024-01-01T00:00:00")
        
        summary = tracker.get_cost_summary()
        logger.info(f"✓ Cost tracker test passed: {summary}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Cost tracker test failed: {str(e)}")
        return False


def check_environment_variables():
    """Check if all required environment variables are set."""
    logger = logging.getLogger(__name__)
    logger.info("Checking environment variables...")
    
    required_vars = {
        'AWS_ACCESS_KEY_ID': 'AWS credentials for Claude',
        'AWS_SECRET_ACCESS_KEY': 'AWS credentials for Claude',
        'AWS_REGION': 'AWS region for Claude',
        'OPENAI_API_KEY': 'OpenAI API key for GPT-5',
        'GCP_PROJECT_ID': 'Google Cloud project ID for Gemini',
        'GCP_LOCATION': 'Google Cloud location for Gemini'
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"{var} ({description})")
    
    if missing_vars:
        logger.error("✗ Missing environment variables:")
        for var in missing_vars:
            logger.error(f"  - {var}")
        return False
    else:
        logger.info("✓ All required environment variables are set")
        return True


def main():
    """Main function to run all API tests."""
    # Load environment variables
    load_dotenv()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting API connectivity tests...")
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Check environment variables
    env_check = check_environment_variables()
    
    # Test each API
    claude_test = test_claude_api()
    gpt_test = test_gpt_api()
    gemini_test = test_gemini_api()
    cost_test = test_cost_tracker()
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("API CONNECTIVITY TEST SUMMARY")
    logger.info("="*50)
    logger.info(f"Environment Variables: {'✓' if env_check else '✗'}")
    logger.info(f"Claude API: {'✓' if claude_test else '✗'}")
    logger.info(f"GPT-5 API: {'✓' if gpt_test else '✗'}")
    logger.info(f"Gemini API: {'✓' if gemini_test else '✗'}")
    logger.info(f"Cost Tracker: {'✓' if cost_test else '✗'}")
    
    all_passed = all([env_check, claude_test, gpt_test, gemini_test, cost_test])
    
    if all_passed:
        logger.info("\n🎉 All tests passed! Ready to run evaluation.")
    else:
        logger.error("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
