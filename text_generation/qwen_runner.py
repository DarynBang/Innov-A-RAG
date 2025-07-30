"""
Qwen 2.5-VL text generation runner.
"""
from utils.logging_utils import setup_logging, get_logger
setup_logging()

import torch
from config.prompts import GENERALIZE_PROMPT_TEMPLATE
from utils.model_utils import get_qwen_vl_model_and_processor

logger = get_logger(__name__)

def generate_caption_with_qwen(input_data: dict) -> str:
    """
    Generate a response using Qwen 2.5-VL.
    
    Args:
        input_data: dict with keys 'query', 'firm_summary_context', 'patent_context'
        
    Returns:
        str: Generated response
    """
    query = input_data.get("query", "")
    firm_summary_context = input_data.get("firm_summary_context", "")
    patent_context = input_data.get("patent_context", "")
    
    logger.info(f"[Qwen] Received query: {query}")
    logger.info(f"[Qwen] Firm context length: {len(firm_summary_context)}")
    logger.info(f"[Qwen] Patent context length: {len(patent_context)}")

    # Load Qwen2.5-VL model and processor
    try:
        model, processor = get_qwen_vl_model_and_processor()
        logger.info("[Qwen] Model and processor loaded successfully")
    except Exception as e:
        logger.error(f"[Qwen] Error loading model: {e}")
        return f"Error loading Qwen model: {str(e)}"

    # Format the prompt with correct parameter names
    prompt = GENERALIZE_PROMPT_TEMPLATE.format(
        query=query,
        firm_summary_context=firm_summary_context,
        patent_context=patent_context
    )

    try:
        logger.info("[Qwen] Generating response")
        
        # Prepare messages for Qwen
        messages = [
            {
                "role": "user", 
                "content": [{"type": "text", "text": prompt}]
            }
        ]

        # Apply chat template and tokenize
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], padding=True, return_tensors="pt")
        inputs = inputs.to(model.device)

        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512, temperature=0)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)

        # Clean up GPU memory
        inputs.to("cpu")
        del generated_ids, generated_ids_trimmed, inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        result = output_text[0].strip()
        logger.info(f"[Qwen] Generated response length: {len(result)}")
        return result
        
    except Exception as e:
        logger.error(f"[Qwen] Error generating response: {e}")
        return f"Error generating response: {str(e)}"

