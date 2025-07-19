"""
OpenAI GPT-4o-mini text generation runner.
"""
from utils.logging_utils import setup_logging, get_logger
setup_logging()

from openai import OpenAI
from config.prompts import GENERALIZE_PROMPT_TEMPLATE
import os
from dotenv import load_dotenv

logger = get_logger(__name__)

# Load environment variables
if "OPENAI_API_KEY" not in os.environ:
    load_dotenv()
    logger.info("Loaded .env for OpenAI credentials")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_caption_with_openai(input_data: dict) -> str:
    """
    Generate a response using OpenAI GPT-4o-mini.
    
    Args:
        input_data: dict with keys 'query', 'firm_summary_context', 'patent_context'
        
    Returns:
        str: Generated response
    """
    query = input_data.get("query", "")
    firm_summary_context = input_data.get("firm_summary_context", "")
    patent_context = input_data.get("patent_context", "")
    
    logger.info(f"[OpenAI] Received query: {query}")
    logger.info(f"[OpenAI] Firm context length: {len(firm_summary_context)}")
    logger.info(f"[OpenAI] Patent context length: {len(patent_context)}")

    # Format the prompt with correct parameter names
    prompt = GENERALIZE_PROMPT_TEMPLATE.format(
        query=query,
        firm_summary_context=firm_summary_context,
        patent_context=patent_context
    )

    try:
        logger.info("[OpenAI] Sending request to GPT-4o-mini")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful market analyst assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
            temperature=0
        )
        
        result = response.choices[0].message.content.strip()
        logger.info(f"[OpenAI] Generated response length: {len(result)}")
        return result
        
    except Exception as e:
        logger.error(f"[OpenAI] Error generating response: {e}")
        return f"Error generating response: {str(e)}"



