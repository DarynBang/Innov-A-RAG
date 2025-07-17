# config/agent_config.py

agent_config = {
    # Shared QA model name for all agents - text (currently support: "qwen", "gemini", "openai")
    "qa_text": "gemini",
    "qa_generalize": "gemini",
    "qa_merge": "gemini",
    "qa_planning": "gemini",
    "qa_verifier": "gemini",

    # --- new code --- 
    "max_loop": 6,
    "max_tasks": 4,
    "threshold": 6,
    # --- end new code --- 
        
}


