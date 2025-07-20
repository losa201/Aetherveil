"""
Network utilities for Chimera operations
"""

import socket
import asyncio
import aiohttp
import random
from typing import Optional, Dict, Any

class NetworkUtils:
    """Network utilities for stealth operations"""
    
    @staticmethod
    def get_local_ip() -> str:
        """Get local IP address"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"
            
    @staticmethod
    def is_port_open(host: str, port: int, timeout: float = 3.0) -> bool:
        """Check if port is open"""
        try:
            with socket.create_connection((host, port), timeout):
                return True
        except (socket.timeout, socket.error):
            return False
            
    @staticmethod
    async def http_request(url: str, method: str = "GET", 
                          headers: Optional[Dict[str, str]] = None,
                          proxy: Optional[str] = None) -> Dict[str, Any]:
        """Make HTTP request with optional proxy"""
        
        connector = None
        if proxy:
            connector = aiohttp.ProxyConnector.from_url(proxy)
            
        async with aiohttp.ClientSession(connector=connector) as session:
            try:
                async with session.request(method, url, headers=headers) as response:
                    return {
                        "status": response.status,
                        "headers": dict(response.headers),
                        "content": await response.text()
                    }
            except Exception as e:
                return {"error": str(e)}
                
    @staticmethod
    def get_random_user_agent() -> str:
        """Get random user agent string"""
        agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
        ]
        return random.choice(agents)