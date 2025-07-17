
# agents/multi_agent_runner.py

from typing import List, Dict, Optional
from M3ARAG.agents.base import BaseAgent
from M3ARAG.agents.registry import AGENTS

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

    def run(self, initial_input: Dict[str, str],
            textual_contexts: Optional[List[dict]] = None) -> str:
        self.shared_memory.update(initial_input)

        for agent in self.agents:
            print(f"🤖 Running {agent.name}...")

            if agent.name == "TextAgent" and textual_contexts is not None:
                print(f"Textual contexts: {textual_contexts}")
                output = agent.run(self.shared_memory, textual_contexts)
            else:
                output = agent.run(self.shared_memory)

            print(f"🧠 Output from {agent.name}:\n{output}\n")

            self.shared_memory[agent.name] = output

            if agent.name == "TextAgent":
                self.text_answer = output.strip()

        # Return output from final agent
        return output
