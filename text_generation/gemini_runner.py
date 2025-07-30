"""
Gemini 2.0 Flash text generation runner.
"""
from utils.logging_utils import setup_logging, get_logger
setup_logging()

import os
from dotenv import load_dotenv
import google.generativeai as genai
from config.prompts import GENERALIZE_PROMPT_TEMPLATE

logger = get_logger(__name__)

# Load environment variables
if "GENAI_API_KEY" not in os.environ:
    load_dotenv()
    logger.info("Loaded .env for Gemini credentials")

GENAI_API_KEY = os.getenv("GENAI_API_KEY")
if not GENAI_API_KEY:
    logger.error("GENAI_API_KEY not found in environment")
else:
    genai.configure(api_key=GENAI_API_KEY)

def generate_caption_with_gemini(input_data: dict) -> str:
    """
    Generate a response using Gemini 2.0 Flash.
    
    Args:
        input_data: dict with keys 'query', 'firm_summary_context', 'patent_context'
        
    Returns:
        str: Generated response
    """
    query = input_data.get("query", "")
    firm_summary_context = input_data.get("firm_summary_context", "")
    patent_context = input_data.get("patent_context", "")
    
    logger.info(f"[Gemini] Received query: {query}")
    logger.info(f"[Gemini] Firm context length: {len(firm_summary_context)}")
    logger.info(f"[Gemini] Patent context length: {len(patent_context)}")

    # Format the prompt with correct parameter names
    prompt = GENERALIZE_PROMPT_TEMPLATE.format(
        query=query,
        firm_summary_context=firm_summary_context,
        patent_context=patent_context
    )

    try:
        logger.info("[Gemini] Sending request to Gemini 2.0 Flash")
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0,
                "max_output_tokens": 1024,
            }
        )
        
        result = response.text.strip()
        logger.info(f"[Gemini] Generated response length: {len(result)}")
        return result
        
    except Exception as e:
        logger.error(f"[Gemini] Error generating response: {e}")
        return f"Error generating response: {str(e)}"


