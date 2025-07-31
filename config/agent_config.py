# config/agent_config.py

# Default LLM type for all agents (options: "openai", "gemini", "ollama")
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

# Product suggestion specific configuration
PRODUCT_SUGGESTION_CONFIG = {
    "max_suggestions": 10,
    "citation_format": "[Context from {source_type} {source_id}]",
    "relevance_threshold": 0.7,
    "enable_context_preview": True,
    "detailed_view_default": False,
    "max_context_length": 2000,  # Max length for context display
    "context_preview_length": 300,  # Length for preview mode
    "enable_source_tracking": True,
    "require_citations": True
}
