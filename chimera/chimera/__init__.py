"""
Chimera: Neuroplastic Autonomous Red-Team Organism

A sophisticated, self-learning red-team platform that combines:
- Neuroplastic reasoning and adaptation
- Character-driven decision making  
- Stealthy web reconnaissance
- LLM collaboration for tactical advice
- Autonomous tool execution with OPSEC
- Continuous learning and knowledge graph evolution

Version: 1.0.0
Author: Chimera Development Team
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Chimera Development Team"
__license__ = "MIT"

from .core.engine import ChimeraEngine
from .reasoner.reasoner import NeuroplasticReasoner
from .memory.knowledge_graph import KnowledgeGraph
from .web.searcher import StealthWebSearcher
from .llm.collaborator import LLMCollaborator
from .planner.planner import TacticalPlanner
from .executor.executor import TaskExecutor
from .validator.validator import ModuleValidator
from .reporter.reporter import ReportGenerator

__all__ = [
    "ChimeraEngine",
    "NeuroplasticReasoner", 
    "KnowledgeGraph",
    "StealthWebSearcher",
    "LLMCollaborator",
    "TacticalPlanner",
    "TaskExecutor",
    "ModuleValidator",
    "ReportGenerator"
]