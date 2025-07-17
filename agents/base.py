# agents/base.py
from abc import ABC, abstractmethod
from typing import Optional, List

class BaseAgent(ABC):
    def __init__(self, name: str, qa_model: str = "qwen"):
        self.qa_model = qa_model
        self.name = name

    @abstractmethod
    def run(self, input_data: dict, contexts: Optional[List[dict]] = None) -> str:
        """Run the agent with input data and return output string."""
        pass
