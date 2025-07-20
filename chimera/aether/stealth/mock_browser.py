"""
Mock browser for when Playwright is not available
"""

import asyncio
import logging
import random
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class MockBrowser:
    """Mock browser that simulates interactions without real automation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.current_url = None
        self.session_active = False
        
    async def initialize(self):
        """Initialize mock browser"""
        logger.info("Initializing Mock Browser (Playwright not available)")
        self.session_active = True
        
    async def navigate_to_url(self, url: str, wait_for_load: bool = True) -> bool:
        """Simulate navigation"""
        logger.info(f"Mock navigation to: {url}")
        self.current_url = url
        
        if wait_for_load:
            await asyncio.sleep(random.uniform(1, 3))  # Simulate load time
            
        return True
        
    async def interact_with_llm_site(self, site_url: str, prompt: str, 
                                   conversation_context: Optional[Dict] = None) -> Dict[str, Any]:
        """Simulate LLM interaction"""
        
        logger.info(f"Mock LLM interaction with {site_url}")
        logger.info(f"Prompt: {prompt[:100]}...")
        
        # Simulate thinking time
        await asyncio.sleep(random.uniform(2, 5))
        
        # Generate mock response based on site
        if 'openai.com' in site_url or 'chatgpt' in site_url:
            response = self._generate_chatgpt_response(prompt)
        elif 'claude.ai' in site_url:
            response = self._generate_claude_response(prompt)
        else:
            response = self._generate_generic_response(prompt)
            
        return {
            'success': True,
            'response': response,
            'timestamp': datetime.utcnow().isoformat(),
            'llm': self._get_llm_name(site_url)
        }
        
    async def extract_page_content(self, selectors: list) -> Dict[str, Any]:
        """Simulate content extraction"""
        logger.info("Mock content extraction")
        
        return {
            'success': True,
            'content': {selector: [f"Mock content for {selector}"] for selector in selectors}
        }
        
    async def rotate_identity(self):
        """Simulate identity rotation"""
        logger.info("Mock identity rotation")
        await asyncio.sleep(1)
        
    async def close(self):
        """Close mock browser"""
        logger.info("Mock browser closed")
        self.session_active = False
        
    def _generate_chatgpt_response(self, prompt: str) -> str:
        """Generate ChatGPT-style response"""
        
        responses = [
            f"That's a great question about {self._extract_topic(prompt)}! Here's how I understand it: The key is to approach this systematically. First, consider the foundational principles, then build upon them with practical examples. This approach tends to work well because it combines theoretical understanding with hands-on experience.",
            
            f"I find your curiosity about {self._extract_topic(prompt)} quite interesting! From my perspective, the most elegant solution often involves breaking down complex problems into smaller, manageable pieces. Think of it like building with blocks - each piece supports the next, creating a strong foundation for understanding.",
            
            f"What you're asking about {self._extract_topic(prompt)} touches on some fascinating concepts! I'd recommend starting with the basics and gradually working your way up to more advanced topics. The learning process is really about making connections between ideas - the more connections you make, the deeper your understanding becomes.",
            
            f"Your question about {self._extract_topic(prompt)} shows real insight! One approach that many find helpful is to practice regularly and reflect on what you learn. It's like physical exercise - consistency matters more than intensity. Each small step forward builds momentum for the next."
        ]
        
        return random.choice(responses)
        
    def _generate_claude_response(self, prompt: str) -> str:
        """Generate Claude-style response"""
        
        responses = [
            f"I appreciate your thoughtful question about {self._extract_topic(prompt)}. Let me offer a structured perspective: This area involves several interconnected concepts that work together to create a comprehensive understanding. The key insight is that mastery comes through deliberate practice combined with careful analysis of results.",
            
            f"Your inquiry about {self._extract_topic(prompt)} is quite nuanced, and I think it deserves a careful response. The most effective approach I've observed involves three main stages: initial exploration to understand the landscape, focused practice to build skills, and reflective analysis to deepen comprehension. Each stage informs and strengthens the others.",
            
            f"Thank you for this engaging question about {self._extract_topic(prompt)}. I find it helpful to think about this topic from multiple angles. Consider both the theoretical framework and the practical applications - they often illuminate different aspects of the same underlying principles. The interplay between theory and practice tends to accelerate learning significantly.",
            
            f"This is a sophisticated question about {self._extract_topic(prompt)} that touches on several important principles. I'd suggest approaching it with both curiosity and patience. The most valuable insights often emerge not from rushing to conclusions, but from carefully examining the relationships between different concepts and testing them in various contexts."
        ]
        
        return random.choice(responses)
        
    def _generate_generic_response(self, prompt: str) -> str:
        """Generate generic AI response"""
        
        responses = [
            f"Regarding {self._extract_topic(prompt)}, I think the most important thing to understand is that learning is an iterative process. Each attempt teaches you something new, even if it doesn't lead to immediate success. The key is to remain curious and persistent.",
            
            f"Your question about {self._extract_topic(prompt)} is well-posed! I've found that the best way to approach complex topics is through a combination of study, practice, and discussion with others. Different perspectives often reveal aspects you might not have considered on your own.",
            
            f"When it comes to {self._extract_topic(prompt)}, I believe the fundamentals are crucial. Once you have a solid foundation, you can build more advanced understanding on top of it. Think of it as constructing a building - you need a strong foundation before adding upper floors."
        ]
        
        return random.choice(responses)
        
    def _extract_topic(self, prompt: str) -> str:
        """Extract main topic from prompt"""
        
        # Simple keyword extraction
        keywords = ['programming', 'learning', 'ai', 'creativity', 'technology', 'problem', 'solution', 'method', 'approach']
        
        prompt_lower = prompt.lower()
        for keyword in keywords:
            if keyword in prompt_lower:
                return keyword
                
        return "this topic"
        
    def _get_llm_name(self, site_url: str) -> str:
        """Get LLM name from site URL"""
        
        if 'openai.com' in site_url or 'chatgpt' in site_url:
            return 'chatgpt'
        elif 'claude.ai' in site_url:
            return 'claude'
        elif 'gemini' in site_url or 'bard' in site_url:
            return 'gemini'
        else:
            return 'unknown_llm'