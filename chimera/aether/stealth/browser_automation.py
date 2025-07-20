"""
Aether Stealth Browser Automation: Advanced human-like browser control
"""

import asyncio
import logging
import random
import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import re
import base64

try:
    from playwright.async_api import async_playwright, Browser, BrowserContext, Page
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    Browser = BrowserContext = Page = None

try:
    from fake_useragent import UserAgent
    FAKE_USERAGENT_AVAILABLE = True
except ImportError:
    FAKE_USERAGENT_AVAILABLE = False
    
try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False

logger = logging.getLogger(__name__)

class BehaviorProfile(Enum):
    """Human behavior simulation profiles"""
    CAUTIOUS = "cautious"          # Slow, careful browsing
    NORMAL = "normal"              # Average human behavior
    CURIOUS = "curious"            # Active exploration
    FOCUSED = "focused"            # Task-oriented browsing

class InteractionType(Enum):
    """Types of browser interactions"""
    SEARCH = "search"
    NAVIGATION = "navigation"
    FORM_FILLING = "form_filling"
    CONTENT_EXTRACTION = "content_extraction"
    LLM_INTERACTION = "llm_interaction"

@dataclass
class BrowserFingerprint:
    """Browser fingerprint configuration"""
    user_agent: str
    viewport_width: int
    viewport_height: int
    screen_width: int
    screen_height: int
    timezone: str
    language: str
    platform: str
    hardware_concurrency: int
    device_memory: int
    color_depth: int
    pixel_ratio: float
    
@dataclass
class HumanBehaviorParams:
    """Parameters for human-like behavior simulation"""
    typing_delay_min: float
    typing_delay_max: float
    click_delay_min: float
    click_delay_max: float
    scroll_delay_min: float
    scroll_delay_max: float
    page_load_wait_min: float
    page_load_wait_max: float
    mouse_movement_steps: int
    reading_time_per_word: float

@dataclass
class BrowsingSession:
    """Represents a browsing session"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    pages_visited: List[str]
    interactions: List[Dict[str, Any]]
    behavior_profile: BehaviorProfile
    fingerprint: BrowserFingerprint
    success: bool = True
    error_message: Optional[str] = None

class AdvancedStealthBrowser:
    """
    Advanced stealth browser with human-like behavior simulation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        
        # Stealth configuration
        self.stealth_level = config.get('stealth_level', 0.8)
        self.behavior_profile = BehaviorProfile(config.get('behavior_profile', 'normal'))
        self.proxy_config = config.get('proxy', None)
        
        # State tracking
        self.current_session: Optional[BrowsingSession] = None
        self.session_history: List[BrowsingSession] = []
        self.current_fingerprint: Optional[BrowserFingerprint] = None
        
        # Human behavior simulation
        self.behavior_params = self._get_behavior_params()
        if FAKE_USERAGENT_AVAILABLE:
            self.user_agent_generator = UserAgent()
        else:
            self.user_agent_generator = None
        
        # Performance tracking
        self.interaction_stats = {
            'total_interactions': 0,
            'successful_interactions': 0,
            'average_response_time': 0.0,
            'stealth_score': 0.0
        }
        
    async def initialize(self):
        """Initialize the stealth browser"""
        
        logger.info("Initializing Advanced Stealth Browser...")
        
        if not PLAYWRIGHT_AVAILABLE:
            logger.warning("Playwright not available - using mock browser")
            from .mock_browser import MockBrowser
            self.mock_browser = MockBrowser(self.config)
            await self.mock_browser.initialize()
            return
        
        try:
            # Launch Playwright
            self.playwright = await async_playwright().start()
            
            # Generate initial fingerprint
            self.current_fingerprint = await self._generate_fingerprint()
            
            # Launch browser with stealth configuration
            await self._launch_browser()
            
            # Start new browsing session
            await self._start_new_session()
            
            logger.info(f"Stealth browser initialized - Profile: {self.behavior_profile.value}")
            
        except Exception as e:
            logger.error(f"Failed to initialize stealth browser: {e}")
            raise
            
    async def navigate_to_url(self, url: str, wait_for_load: bool = True) -> bool:
        """Navigate to URL with human-like behavior"""
        
        # Use mock browser if Playwright not available
        if not PLAYWRIGHT_AVAILABLE and hasattr(self, 'mock_browser'):
            return await self.mock_browser.navigate_to_url(url, wait_for_load)
        
        try:
            if not self.page:
                await self._launch_browser()
                
            logger.info(f"Navigating to: {url}")
            
            # Pre-navigation delay
            await self._human_delay('page_load_wait')
            
            # Navigate
            response = await self.page.goto(url, wait_until='networkidle')
            
            if wait_for_load:
                # Simulate reading time
                await self._simulate_page_reading()
                
                # Random scroll behavior
                if random.random() < 0.7:
                    await self._human_scroll()
                    
            # Track navigation
            if self.current_session:
                self.current_session.pages_visited.append(url)
                self.current_session.interactions.append({
                    'type': InteractionType.NAVIGATION.value,
                    'url': url,
                    'timestamp': datetime.utcnow().isoformat(),
                    'success': response.ok if response else True
                })
                
            self.interaction_stats['total_interactions'] += 1
            if response and response.ok:
                self.interaction_stats['successful_interactions'] += 1
                
            return response.ok if response else True
            
        except Exception as e:
            logger.error(f"Error navigating to {url}: {e}")
            if self.current_session:
                self.current_session.interactions.append({
                    'type': InteractionType.NAVIGATION.value,
                    'url': url,
                    'timestamp': datetime.utcnow().isoformat(),
                    'success': False,
                    'error': str(e)
                })
            return False
            
    async def search_google(self, query: str) -> Dict[str, Any]:
        """Perform Google search with advanced stealth"""
        
        try:
            # Navigate to Google
            await self.navigate_to_url('https://www.google.com')
            
            # Wait for search box
            search_box = await self.page.wait_for_selector('input[name="q"]', timeout=10000)
            
            if not search_box:
                return {'success': False, 'error': 'Search box not found'}
            
            # Human-like typing
            await self._human_type(search_box, query)
            
            # Submit search
            await search_box.press('Enter')
            
            # Wait for results
            await self.page.wait_for_selector('div[data-ved]', timeout=15000)
            
            # Simulate reading search results
            await self._simulate_search_result_reading()
            
            # Extract results
            results = await self._extract_search_results()
            
            # Track interaction
            if self.current_session:
                self.current_session.interactions.append({
                    'type': InteractionType.SEARCH.value,
                    'query': query,
                    'timestamp': datetime.utcnow().isoformat(),
                    'results_count': len(results),
                    'success': True
                })
                
            logger.info(f"Search completed - {len(results)} results found")
            
            return {
                'success': True,
                'query': query,
                'results': results,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error performing Google search: {e}")
            return {'success': False, 'error': str(e)}
            
    async def interact_with_llm_site(self, site_url: str, prompt: str, 
                                   conversation_context: Optional[Dict] = None) -> Dict[str, Any]:
        """Interact with browser-based LLM sites (ChatGPT, Claude, etc.)"""
        
        # Use mock browser if Playwright not available
        if not PLAYWRIGHT_AVAILABLE and hasattr(self, 'mock_browser'):
            return await self.mock_browser.interact_with_llm_site(site_url, prompt, conversation_context)
        
        try:
            # Navigate to LLM site
            await self.navigate_to_url(site_url)
            
            # Site-specific interaction logic
            if 'chat.openai.com' in site_url or 'chatgpt.com' in site_url:
                return await self._interact_with_chatgpt(prompt, conversation_context)
            elif 'claude.ai' in site_url:
                return await self._interact_with_claude(prompt, conversation_context)
            elif 'gemini.google.com' in site_url or 'bard.google.com' in site_url:
                return await self._interact_with_gemini(prompt, conversation_context)
            else:
                return await self._generic_llm_interaction(prompt, conversation_context)
                
        except Exception as e:
            logger.error(f"Error interacting with LLM site {site_url}: {e}")
            return {'success': False, 'error': str(e)}
            
    async def extract_page_content(self, selectors: List[str]) -> Dict[str, Any]:
        """Extract content from current page using CSS selectors"""
        
        if not self.page:
            return {'error': 'No active page'}
            
        extracted = {}
        
        try:
            for selector in selectors:
                try:
                    elements = await self.page.query_selector_all(selector)
                    content = []
                    
                    for element in elements:
                        text = await element.text_content()
                        if text:
                            content.append(text.strip())
                            
                    extracted[selector] = content
                    
                except Exception as e:
                    logger.warning(f"Could not extract content with selector {selector}: {e}")
                    extracted[selector] = []
                    
            # Track interaction
            if self.current_session:
                self.current_session.interactions.append({
                    'type': InteractionType.CONTENT_EXTRACTION.value,
                    'selectors': selectors,
                    'timestamp': datetime.utcnow().isoformat(),
                    'success': True
                })
                
            return {'success': True, 'content': extracted}
            
        except Exception as e:
            logger.error(f"Error extracting page content: {e}")
            return {'success': False, 'error': str(e)}
            
    async def rotate_identity(self):
        """Rotate browser identity for enhanced stealth"""
        
        try:
            logger.info("Rotating browser identity...")
            
            # End current session
            if self.current_session:
                await self._end_current_session()
                
            # Close current browser context
            if self.context:
                await self.context.close()
                
            # Generate new fingerprint
            self.current_fingerprint = await self._generate_fingerprint()
            
            # Create new browser context with new fingerprint
            await self._create_browser_context()
            
            # Start new session
            await self._start_new_session()
            
            logger.info("Identity rotation complete")
            
        except Exception as e:
            logger.error(f"Error rotating identity: {e}")
            raise
            
    # Private helper methods
    
    async def _launch_browser(self):
        """Launch browser with stealth configuration"""
        
        browser_args = [
            '--no-sandbox',
            '--disable-dev-shm-usage',
            '--disable-blink-features=AutomationControlled',
            '--disable-web-security',
            '--disable-features=VizDisplayCompositor',
            '--disable-ipc-flooding-protection',
            '--disable-renderer-backgrounding',
            '--disable-backgrounding-occluded-windows',
            '--disable-client-side-phishing-detection',
            '--disable-sync',
            '--disable-default-apps',
            '--disable-extensions',
            '--hide-scrollbars',
            '--mute-audio'
        ]
        
        # Add proxy if configured
        if self.proxy_config:
            browser_args.append(f'--proxy-server={self.proxy_config["server"]}')
            
        # Launch browser
        self.browser = await self.playwright.chromium.launch(
            headless=self.config.get('headless', True),
            args=browser_args
        )
        
        # Create browser context
        await self._create_browser_context()
        
    async def _create_browser_context(self):
        """Create browser context with fingerprint"""
        
        context_options = {
            'viewport': {
                'width': self.current_fingerprint.viewport_width,
                'height': self.current_fingerprint.viewport_height
            },
            'user_agent': self.current_fingerprint.user_agent,
            'locale': self.current_fingerprint.language,
            'timezone_id': self.current_fingerprint.timezone,
            'permissions': [],
            'extra_http_headers': {
                'Accept-Language': f'{self.current_fingerprint.language},en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Upgrade-Insecure-Requests': '1'
            }
        }
        
        # Add proxy authentication if needed
        if self.proxy_config and 'username' in self.proxy_config:
            context_options['http_credentials'] = {
                'username': self.proxy_config['username'],
                'password': self.proxy_config['password']
            }
            
        self.context = await self.browser.new_context(**context_options)
        
        # Add stealth scripts
        await self.context.add_init_script(self._get_stealth_script())
        
        # Create page
        self.page = await self.context.new_page()
        
        # Additional stealth measures
        await self._apply_advanced_stealth()
        
    async def _generate_fingerprint(self) -> BrowserFingerprint:
        """Generate realistic browser fingerprint"""
        
        # Common viewport sizes
        viewports = [
            (1920, 1080), (1366, 768), (1440, 900),
            (1536, 864), (1280, 720), (1600, 900),
            (1024, 768), (1680, 1050)
        ]
        
        viewport = random.choice(viewports)
        
        # Generate screen size (larger than viewport)
        screen_width = viewport[0] + random.randint(0, 100)
        screen_height = viewport[1] + random.randint(50, 150)
        
        # Timezone selection
        timezones = [
            'America/New_York', 'America/Los_Angeles', 'America/Chicago',
            'Europe/London', 'Europe/Paris', 'Europe/Berlin',
            'Asia/Tokyo', 'Asia/Shanghai', 'Australia/Sydney'
        ]
        
        # Language selection
        languages = ['en-US', 'en-GB', 'fr-FR', 'de-DE', 'es-ES', 'it-IT']
        
        # Platform selection
        platforms = ['Win32', 'MacIntel', 'Linux x86_64']
        
        return BrowserFingerprint(
            user_agent=self.user_agent_generator.random if self.user_agent_generator else "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            viewport_width=viewport[0],
            viewport_height=viewport[1],
            screen_width=screen_width,
            screen_height=screen_height,
            timezone=random.choice(timezones),
            language=random.choice(languages),
            platform=random.choice(platforms),
            hardware_concurrency=random.choice([2, 4, 8, 16]),
            device_memory=random.choice([2, 4, 8, 16]),
            color_depth=random.choice([24, 32]),
            pixel_ratio=random.choice([1.0, 1.25, 1.5, 2.0])
        )
        
    def _get_stealth_script(self) -> str:
        """Get JavaScript stealth script"""
        
        return f"""
        // Remove webdriver property
        Object.defineProperty(navigator, 'webdriver', {{
            get: () => undefined,
        }});
        
        // Mock navigator properties
        Object.defineProperty(navigator, 'languages', {{
            get: () => ['{self.current_fingerprint.language}', 'en'],
        }});
        
        Object.defineProperty(navigator, 'platform', {{
            get: () => '{self.current_fingerprint.platform}',
        }});
        
        Object.defineProperty(navigator, 'hardwareConcurrency', {{
            get: () => {self.current_fingerprint.hardware_concurrency},
        }});
        
        Object.defineProperty(navigator, 'deviceMemory', {{
            get: () => {self.current_fingerprint.device_memory},
        }});
        
        // Mock screen properties
        Object.defineProperty(screen, 'width', {{
            get: () => {self.current_fingerprint.screen_width},
        }});
        
        Object.defineProperty(screen, 'height', {{
            get: () => {self.current_fingerprint.screen_height},
        }});
        
        Object.defineProperty(screen, 'colorDepth', {{
            get: () => {self.current_fingerprint.color_depth},
        }});
        
        // Mock device pixel ratio
        Object.defineProperty(window, 'devicePixelRatio', {{
            get: () => {self.current_fingerprint.pixel_ratio},
        }});
        
        // Mock plugins
        Object.defineProperty(navigator, 'plugins', {{
            get: () => [
                {{name: 'Chrome PDF Plugin', description: 'Portable Document Format'}},
                {{name: 'Chrome PDF Viewer', description: 'PDF Viewer'}},
                {{name: 'Native Client', description: 'Native Client'}}
            ]
        }});
        
        // Mock WebGL
        const getParameter = WebGLRenderingContext.prototype.getParameter;
        WebGLRenderingContext.prototype.getParameter = function(parameter) {{
            if (parameter === 37445) {{
                return 'Intel Inc.';
            }}
            if (parameter === 37446) {{
                return 'Intel Iris OpenGL Engine';
            }}
            return getParameter.call(this, parameter);
        }};
        
        // Mock permissions
        const originalQuery = window.navigator.permissions.query;
        window.navigator.permissions.query = (parameters) => (
            parameters.name === 'notifications' ?
                Promise.resolve({{ state: Notification.permission }}) :
                originalQuery(parameters)
        );
        """
        
    async def _apply_advanced_stealth(self):
        """Apply advanced stealth measures"""
        
        # Override navigator.webdriver
        await self.page.evaluate('() => { delete Object.getPrototypeOf(navigator).webdriver }')
        
        # Randomize canvas fingerprint
        await self.page.evaluate("""
        () => {
            const toBlob = HTMLCanvasElement.prototype.toBlob;
            HTMLCanvasElement.prototype.toBlob = function(callback, type, quality) {
                const canvas = this;
                const ctx = canvas.getContext('2d');
                
                // Add slight noise to canvas
                const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                for (let i = 0; i < imageData.data.length; i += 4) {
                    imageData.data[i] += Math.random() * 0.1;
                    imageData.data[i+1] += Math.random() * 0.1;
                    imageData.data[i+2] += Math.random() * 0.1;
                }
                ctx.putImageData(imageData, 0, 0);
                
                return toBlob.call(this, callback, type, quality);
            };
        }
        """)
        
    def _get_behavior_params(self) -> HumanBehaviorParams:
        """Get behavior parameters based on profile"""
        
        if self.behavior_profile == BehaviorProfile.CAUTIOUS:
            return HumanBehaviorParams(
                typing_delay_min=0.08,
                typing_delay_max=0.25,
                click_delay_min=0.3,
                click_delay_max=1.0,
                scroll_delay_min=0.5,
                scroll_delay_max=1.5,
                page_load_wait_min=2.0,
                page_load_wait_max=5.0,
                mouse_movement_steps=10,
                reading_time_per_word=0.3
            )
        elif self.behavior_profile == BehaviorProfile.FOCUSED:
            return HumanBehaviorParams(
                typing_delay_min=0.04,
                typing_delay_max=0.12,
                click_delay_min=0.1,
                click_delay_max=0.4,
                scroll_delay_min=0.2,
                scroll_delay_max=0.6,
                page_load_wait_min=0.5,
                page_load_wait_max=2.0,
                mouse_movement_steps=5,
                reading_time_per_word=0.15
            )
        elif self.behavior_profile == BehaviorProfile.CURIOUS:
            return HumanBehaviorParams(
                typing_delay_min=0.05,
                typing_delay_max=0.15,
                click_delay_min=0.2,
                click_delay_max=0.8,
                scroll_delay_min=0.3,
                scroll_delay_max=1.0,
                page_load_wait_min=1.0,
                page_load_wait_max=3.0,
                mouse_movement_steps=8,
                reading_time_per_word=0.25
            )
        else:  # NORMAL
            return HumanBehaviorParams(
                typing_delay_min=0.06,
                typing_delay_max=0.18,
                click_delay_min=0.2,
                click_delay_max=0.6,
                scroll_delay_min=0.3,
                scroll_delay_max=0.8,
                page_load_wait_min=1.0,
                page_load_wait_max=3.0,
                mouse_movement_steps=7,
                reading_time_per_word=0.2
            )
            
    async def _human_delay(self, delay_type: str):
        """Add human-like delays"""
        
        params = self.behavior_params
        
        if delay_type == 'typing':
            delay = random.uniform(params.typing_delay_min, params.typing_delay_max)
        elif delay_type == 'click':
            delay = random.uniform(params.click_delay_min, params.click_delay_max)
        elif delay_type == 'scroll':
            delay = random.uniform(params.scroll_delay_min, params.scroll_delay_max)
        elif delay_type == 'page_load_wait':
            delay = random.uniform(params.page_load_wait_min, params.page_load_wait_max)
        else:
            delay = random.uniform(0.1, 0.5)
            
        await asyncio.sleep(delay)
        
    async def _human_type(self, element, text: str):
        """Type text with human-like behavior"""
        
        # Clear existing text
        await element.click()
        await element.press('Control+a')
        await self._human_delay('typing')
        
        # Type character by character
        for char in text:
            await element.type(char)
            await self._human_delay('typing')
            
        # Random pause after typing
        if random.random() < 0.3:
            await asyncio.sleep(random.uniform(0.5, 1.5))
            
    async def _human_scroll(self):
        """Simulate human-like scrolling"""
        
        try:
            # Get page dimensions
            page_height = await self.page.evaluate('document.body.scrollHeight')
            viewport_height = await self.page.evaluate('window.innerHeight')
            
            if page_height <= viewport_height:
                return  # No need to scroll
                
            # Random scroll behavior
            scroll_count = random.randint(1, 5)
            
            for _ in range(scroll_count):
                # Random scroll distance
                scroll_distance = random.randint(100, 500)
                
                # Scroll down
                await self.page.mouse.wheel(0, scroll_distance)
                await self._human_delay('scroll')
                
            # Sometimes scroll back up
            if random.random() < 0.3:
                await asyncio.sleep(random.uniform(1, 2))
                scroll_up = random.randint(200, 800)
                await self.page.mouse.wheel(0, -scroll_up)
                
        except Exception as e:
            logger.warning(f"Error during human scroll: {e}")
            
    async def _simulate_page_reading(self):
        """Simulate time spent reading page content"""
        
        try:
            # Get text content length
            text_content = await self.page.evaluate('document.body.innerText')
            word_count = len(text_content.split()) if text_content else 100
            
            # Calculate reading time
            reading_time = word_count * self.behavior_params.reading_time_per_word
            reading_time = min(reading_time, 30.0)  # Cap at 30 seconds
            
            # Add randomness
            actual_reading_time = reading_time * random.uniform(0.5, 1.5)
            
            await asyncio.sleep(actual_reading_time)
            
        except Exception as e:
            logger.warning(f"Error simulating page reading: {e}")
            await asyncio.sleep(random.uniform(1, 3))
            
    async def _simulate_search_result_reading(self):
        """Simulate reading search results"""
        
        # Scroll through results
        for _ in range(random.randint(2, 5)):
            await self._human_scroll()
            
        # Pause as if reading
        await asyncio.sleep(random.uniform(2, 6))
        
    async def _extract_search_results(self) -> List[Dict[str, str]]:
        """Extract Google search results"""
        
        results = []
        
        try:
            # Wait for results to load
            await self.page.wait_for_selector('div[data-ved]', timeout=10000)
            
            # Extract result elements
            result_elements = await self.page.query_selector_all('div.g')
            
            for element in result_elements[:10]:  # Top 10 results
                try:
                    title_elem = await element.query_selector('h3')
                    url_elem = await element.query_selector('a')
                    snippet_elem = await element.query_selector('span[data-ved]')
                    
                    if title_elem and url_elem:
                        title = await title_elem.text_content()
                        url = await url_elem.get_attribute('href')
                        snippet = await snippet_elem.text_content() if snippet_elem else ''
                        
                        results.append({
                            'title': title.strip() if title else '',
                            'url': url,
                            'snippet': snippet.strip() if snippet else ''
                        })
                        
                except Exception as e:
                    logger.warning(f"Error extracting search result: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error extracting search results: {e}")
            
        return results
        
    async def _interact_with_chatgpt(self, prompt: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Interact with ChatGPT interface"""
        
        try:
            # Wait for the textarea
            textarea = await self.page.wait_for_selector('textarea[placeholder*="Message"]', timeout=15000)
            
            if not textarea:
                return {'success': False, 'error': 'ChatGPT textarea not found'}
            
            # Type the prompt
            await self._human_type(textarea, prompt)
            
            # Find and click send button
            send_button = await self.page.query_selector('button[data-testid="send-button"]')
            if not send_button:
                # Try alternative selector
                send_button = await self.page.query_selector('button[aria-label="Send message"]')
                
            if send_button:
                await self._human_delay('click')
                await send_button.click()
            else:
                # Use Enter key as fallback
                await textarea.press('Enter')
            
            # Wait for response
            await asyncio.sleep(2)  # Initial wait
            
            # Wait for response to complete (look for stop generating button disappearing)
            try:
                await self.page.wait_for_selector('button[aria-label="Stop generating"]', 
                                                timeout=5000, state='attached')
                await self.page.wait_for_selector('button[aria-label="Stop generating"]', 
                                                timeout=60000, state='detached')
            except:
                # Response might have completed quickly
                pass
            
            # Extract response
            response_elements = await self.page.query_selector_all('[data-message-author-role="assistant"]')
            
            if response_elements:
                latest_response = response_elements[-1]
                response_text = await latest_response.text_content()
                
                return {
                    'success': True,
                    'response': response_text.strip() if response_text else '',
                    'timestamp': datetime.utcnow().isoformat(),
                    'llm': 'chatgpt'
                }
            else:
                return {'success': False, 'error': 'No response found'}
                
        except Exception as e:
            logger.error(f"Error interacting with ChatGPT: {e}")
            return {'success': False, 'error': str(e)}
            
    async def _interact_with_claude(self, prompt: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Interact with Claude interface"""
        
        try:
            # Wait for the input area
            input_area = await self.page.wait_for_selector('div[contenteditable="true"]', timeout=15000)
            
            if not input_area:
                return {'success': False, 'error': 'Claude input area not found'}
            
            # Type the prompt
            await input_area.click()
            await self._human_type(input_area, prompt)
            
            # Send the message (Enter key)
            await input_area.press('Enter')
            
            # Wait for response
            await asyncio.sleep(3)
            
            # Wait for response completion
            try:
                # Look for streaming indicator
                await self.page.wait_for_selector('[data-testid="streaming"]', 
                                                timeout=5000, state='attached')
                await self.page.wait_for_selector('[data-testid="streaming"]', 
                                                timeout=60000, state='detached')
            except:
                pass
            
            # Extract response
            response_elements = await self.page.query_selector_all('[data-is-streaming="false"]')
            
            if response_elements:
                latest_response = response_elements[-1]
                response_text = await latest_response.text_content()
                
                return {
                    'success': True,
                    'response': response_text.strip() if response_text else '',
                    'timestamp': datetime.utcnow().isoformat(),
                    'llm': 'claude'
                }
            else:
                return {'success': False, 'error': 'No response found'}
                
        except Exception as e:
            logger.error(f"Error interacting with Claude: {e}")
            return {'success': False, 'error': str(e)}
            
    async def _interact_with_gemini(self, prompt: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Interact with Gemini interface"""
        
        # Placeholder for Gemini interaction
        return {'success': False, 'error': 'Gemini interaction not implemented yet'}
        
    async def _generic_llm_interaction(self, prompt: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Generic LLM interaction for unknown sites"""
        
        # Placeholder for generic interaction
        return {'success': False, 'error': 'Generic LLM interaction not implemented yet'}
        
    async def _start_new_session(self):
        """Start a new browsing session"""
        
        session_id = hashlib.md5(f"{datetime.utcnow().isoformat()}{random.random()}".encode()).hexdigest()[:12]
        
        self.current_session = BrowsingSession(
            session_id=session_id,
            start_time=datetime.utcnow(),
            end_time=None,
            pages_visited=[],
            interactions=[],
            behavior_profile=self.behavior_profile,
            fingerprint=self.current_fingerprint
        )
        
        logger.info(f"Started new browsing session: {session_id}")
        
    async def _end_current_session(self):
        """End the current browsing session"""
        
        if self.current_session:
            self.current_session.end_time = datetime.utcnow()
            self.session_history.append(self.current_session)
            
            logger.info(f"Ended browsing session: {self.current_session.session_id}")
            self.current_session = None
            
    async def get_stealth_statistics(self) -> Dict[str, Any]:
        """Get comprehensive stealth and performance statistics"""
        
        return {
            'interaction_stats': self.interaction_stats,
            'current_session': asdict(self.current_session) if self.current_session else None,
            'session_history_count': len(self.session_history),
            'current_fingerprint': asdict(self.current_fingerprint) if self.current_fingerprint else None,
            'behavior_profile': self.behavior_profile.value,
            'stealth_level': self.stealth_level
        }
        
    async def close(self):
        """Close browser and cleanup"""
        
        try:
            # End current session
            if self.current_session:
                await self._end_current_session()
                
            # Close browser
            if self.browser:
                await self.browser.close()
                
            # Stop Playwright
            if self.playwright:
                await self.playwright.stop()
                
            logger.info("Stealth browser closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing stealth browser: {e}")