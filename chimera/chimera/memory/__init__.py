"""
Neuroplastic memory and knowledge management
Uses lightweight implementations for better compatibility
"""

# Import lightweight version by default
from .knowledge_graph_lite import LiteKnowledgeGraph as KnowledgeGraph
from .learner import NeuroplasticLearner
from .persistence import DataPersistence

__all__ = ["KnowledgeGraph", "NeuroplasticLearner", "DataPersistence"]