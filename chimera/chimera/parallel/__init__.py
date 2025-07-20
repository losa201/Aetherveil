"""
Parallel specialization module for Chimera
Continuous learning and refinement of spoofing and LLM interaction techniques
"""

from .parallel_worker import ParallelSpecializationWorker, SpoofingConfiguration, LLMInteractionPattern, ExperimentResult

__all__ = [
    "ParallelSpecializationWorker",
    "SpoofingConfiguration",
    "LLMInteractionPattern", 
    "ExperimentResult"
]