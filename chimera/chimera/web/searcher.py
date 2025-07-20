"""
Stealth web searcher implementation
"""

import asyncio
import logging
from typing import Dict, Any, List
from ..core.events import EventSystem, EventType, EventEmitter
from .stealth import StealthBrowser

logger = logging.getLogger(__name__)

class StealthWebSearcher(EventEmitter):
    """Stealth web searcher with anti-detection capabilities"""
    
    def __init__(self, config, event_system: EventSystem):
        super().__init__(event_system, "StealthWebSearcher")
        self.config = config
        self.browser = StealthBrowser(config.get_section("web"))
        
    async def initialize(self):
        """Initialize the web searcher"""
        await self.browser.initialize()
        logger.info("Stealth web searcher initialized")
        
    async def search(self, query: str, engines: List[str] = None) -> List[Dict[str, Any]]:
        """Perform stealth web search"""
        if engines is None:
            engines = self.config.get("web.search_engines", ["google"])
            
        all_results = []
        
        for engine in engines:
            try:
                results = await self.browser.search(engine, query)
                all_results.extend(results.get("results", []))
            except Exception as e:
                logger.error(f"Search failed on {engine}: {e}")
                
        return all_results
        
    async def shutdown(self):
        """Shutdown the searcher"""
        await self.browser.close()