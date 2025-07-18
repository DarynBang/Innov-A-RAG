
# agents/multi_agent_runner.py

from typing import List, Dict, Optional
from agents.base import BaseAgent
from agents.registry import AGENTS

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiAgentRunner:
    def __init__(self):
        self.agents: List[BaseAgent] = []
        self.shared_memory: Dict[str, str] = {}

    def register_agent(self, agent: str, qa_model: str = "qwen"):
        cls = AGENTS[agent]
        agent = cls(name=agent, qa_model=qa_model)
        self.agents.append(agent)

    def run(self,
            initial_input: Dict[str, str],
            patent_contexts: Optional[List[dict]] = None,
            firm_summary_contexts: Optional[List[dict]] = None) -> str:
        self.shared_memory.update(initial_input)

        for agent in self.agents:
            print(f"🤖 Running {agent.name}...")

            if patent_contexts is not None and firm_summary_contexts is not None and agent.name == "GeneralizeAgent":
                print(f"Patent contexts: {patent_contexts}")
                print(f"Firm summary contexts:{firm_summary_contexts}")
                output = agent.run(input_data=self.shared_memory, patent_contexts=patent_contexts, firm_summary_contexts=firm_summary_contexts)

            elif agent.name in ("MarketOpportunityAgent", "MarketRiskAgent", "MarketManagerAgent"):
                output = agent.run(self.shared_memory)

            print(f"🧠 Output from {agent.name}:\n{output}\n")

            self.shared_memory[agent.name] = output


        # Return output from final agent
        return output
