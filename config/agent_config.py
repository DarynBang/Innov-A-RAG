# config/agent_config.py

# Default LLM type for all agents (options: "openai", "gemini", "qwen")
DEFAULT_LLM_TYPE = "gemini"

# Agent-specific LLM configurations
agent_config = {
    # Core workflow agents
    "planning_agent": "gemini",
    "normalize_query_agent": "gemini", 
    "generalize_agent": "gemini",

    # Market analysis agents
    "market_opportunity_agent": "gemini",
    "market_risk_agent": "gemini",
    "market_manager_agent": "gemini",
    
    # Validation agents
    "fact_checking_agent": "gemini",
}
