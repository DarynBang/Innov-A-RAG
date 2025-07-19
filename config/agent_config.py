# config/agent_config.py

# Default LLM type for all agents (options: "openai", "gemini", "qwen")
DEFAULT_LLM_TYPE = "openai"

# Agent-specific LLM configurations
agent_config = {
    # Core workflow agents
    "planning_agent": "openai",
    "normalize_query_agent": "openai", 
    "generalize_agent": "openai",

    # Market analysis agents
    "market_opportunity_agent": "openai",
    "market_risk_agent": "openai",
    "market_manager_agent": "openai",
    
    # Validation agents
    "fact_checking_agent": "openai",
}
