"""
agents/registry.py

This module serves as a centralized agent registry for the multi-agent RAG system.

It defines and exposes a dictionary `AGENTS` that maps agent names (as strings)
to their corresponding agent class implementations. This allows dynamic and
configurable agent instantiation in the `MultiAgentRunner`.

"""

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from agents.generalize_agent import GeneralizeAgent
from agents.planning_agent import PlanningAgent
from agents.market_analysts.market_opportunity_agent import MarketOpportunityAgent
from agents.market_analysts.market_risk_agent import MarketRiskAgent
from agents.market_analysts.market_manager_agent import MarketManagerAgent


# Build agents once and reuse
# This later being called and initialized in the MultiAgentRunner
AGENTS = {
    "GeneralizeAgent": GeneralizeAgent,
    "PlanningAgent": PlanningAgent,
    "MarketOpportunityAgent": MarketOpportunityAgent,
    "MarketRiskAgent": MarketRiskAgent,
    "MarketManagerAgent": MarketManagerAgent,
}



