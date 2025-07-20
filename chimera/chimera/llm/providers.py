"""
LLM provider implementations
"""

class LLMProvider:
    """Base LLM provider"""
    
    def __init__(self, config):
        self.config = config