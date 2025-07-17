
# agents/registry.py
from M3ARAG.agents.text_agent import TextAgent
from M3ARAG.agents.image_agent import ImageAgent
from M3ARAG.agents.generalize_agent import GeneralizeAgent
from M3ARAG.agents.finalize_agent import FinalizeAgent

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Build agents once and reuse
AGENTS = {
    "TextAgent": TextAgent,
    "GeneralizeAgent": GeneralizeAgent,
    "FinalizeAgent": FinalizeAgent,
}

