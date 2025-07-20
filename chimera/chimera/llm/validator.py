"""
LLM response validator
"""

class ResponseValidator:
    """Validates LLM responses for safety"""
    
    def __init__(self, config):
        self.config = config