"""
agents/registry.py

This module serves as a centralized agent registry for the multi-agent RAG system.

It defines and exposes a dictionary `AGENTS` that maps agent names (as strings)
to their corresponding agent class implementations. This allows dynamic and
configurable agent instantiation in the `MultiAgentRunner`.
"""
from utils.logging_utils import setup_logging, get_logger
setup_logging()

from agents.planning_agent import PlanningAgent
from agents.generalize_agent import GeneralizeAgent

from agents.normalize_query_agent import NormalizeQueryAgent
from agents.fact_checking_agent import FactCheckingAgent
from agents.market_analysts.market_opportunity_agent import MarketOpportunityAgent
from agents.market_analysts.market_risk_agent import MarketRiskAgent
from agents.market_analysts.market_manager_agent import MarketManagerAgent

logger = get_logger(__name__)

# Build agents once and reuse
# This later being called and initialized in the MultiAgentRunner
AGENTS = {
    "PlanningAgent": PlanningAgent,
    "NormalizeQueryAgent": NormalizeQueryAgent,

    "GeneralizeAgent": GeneralizeAgent,
    "FactCheckingAgent": FactCheckingAgent,
    "MarketOpportunityAgent": MarketOpportunityAgent,
    "MarketRiskAgent": MarketRiskAgent,
    "MarketManagerAgent": MarketManagerAgent,
}

logger.info(f"Agent registry initialized with {len(AGENTS)} agents: {list(AGENTS.keys())}")



