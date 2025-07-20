"""
Advanced stealth browser capabilities for web operations
"""

import asyncio
import random
import time
import logging
from typing import Dict, Any, List, Optional
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium_stealth import stealth
from fake_useragent import UserAgent

logger = logging.getLogger(__name__)

class StealthBrowser:
    """
    Advanced stealth browser with human-like behavior simulation
    Evades common detection mechanisms used by anti-bot systems
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.driver: Optional[webdriver.Chrome] = None
        self.user_agent_generator = UserAgent()
        self.current_profile = None
        
        # Stealth configuration
        self.stealth_level = config.get("stealth_level", 0.7)
        self.human_timing = config.get("human_timing", True)
        self.fingerprint_randomization = config.get("fingerprint_randomization", True)
        
        # Timing parameters for human-like behavior
        self.min_action_delay = 0.5
        self.max_action_delay = 3.0
        self.typing_delay_range = (0.05, 0.15)
        self.scroll_delay_range = (0.2, 0.8)
        
    async def initialize(self):
        """Initialize the stealth browser"""
        try:
            chrome_options = await self._configure_chrome_options()
            
            # Create driver with stealth configuration
            self.driver = webdriver.Chrome(options=chrome_options)
            
            # Apply stealth patches
            stealth(
                self.driver,
                languages=["en-US", "en"],
                vendor="Google Inc.",
                platform="Win32",
                webgl_vendor="Intel Inc.",
                renderer="Intel Iris OpenGL Engine",
                fix_hairline=True,
            )
            
            # Set initial browser properties
            await self._randomize_viewport()
            await self._inject_stealth_scripts()
            
            logger.info("Stealth browser initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize stealth browser: {e}")
            raise
            
    async def navigate_to(self, url: str) -> bool:
        """Navigate to a URL with stealth behavior"""
        try:
            if not self.driver:
                await self.initialize()
                
            # Add random delay before navigation
            if self.human_timing:
                await self._human_delay()
                
            # Navigate to URL
            self.driver.get(url)
            
            # Simulate human-like post-navigation behavior
            await self._simulate_human_browsing()
            
            return True
            
        except Exception as e:
            logger.error(f"Error navigating to {url}: {e}")
            return False
            
    async def search(self, search_engine: str, query: str) -> Dict[str, Any]:
        """Perform a search with stealth behavior"""
        
        search_urls = {
            "google": "https://www.google.com",
            "bing": "https://www.bing.com", 
            "duckduckgo": "https://duckduckgo.com"
        }
        
        if search_engine not in search_urls:
            raise ValueError(f"Unsupported search engine: {search_engine}")
            
        try:
            # Navigate to search engine
            await self.navigate_to(search_urls[search_engine])
            
            # Wait for page load
            await asyncio.sleep(random.uniform(1, 3))
            
            # Find search box and perform search
            if search_engine == "google":
                return await self._google_search(query)
            elif search_engine == "bing":
                return await self._bing_search(query)
            elif search_engine == "duckduckgo":
                return await self._duckduckgo_search(query)
                
        except Exception as e:
            logger.error(f"Error performing search on {search_engine}: {e}")
            return {"results": [], "error": str(e)}
            
    async def extract_content(self, selectors: List[str]) -> Dict[str, Any]:
        """Extract content from current page using CSS selectors"""
        
        if not self.driver:
            return {"error": "Browser not initialized"}
            
        extracted = {}
        
        for selector in selectors:
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                extracted[selector] = [elem.text for elem in elements]
            except Exception as e:
                logger.warning(f"Could not extract content with selector {selector}: {e}")
                extracted[selector] = []
                
        return extracted
        
    async def close(self):
        """Close the browser and clean up"""
        if self.driver:
            try:
                self.driver.quit()
                self.driver = None
                logger.info("Stealth browser closed")
            except Exception as e:
                logger.error(f"Error closing browser: {e}")
                
    # Private methods for stealth implementation
    
    async def _configure_chrome_options(self) -> Options:
        """Configure Chrome options for maximum stealth"""
        
        options = Options()
        
        # Basic stealth options
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        # Randomize user agent
        if self.fingerprint_randomization:
            user_agent = self.user_agent_generator.random
            options.add_argument(f"--user-agent={user_agent}")
            
        # Additional stealth options based on stealth level
        if self.stealth_level > 0.5:
            options.add_argument("--disable-web-security")
            options.add_argument("--allow-running-insecure-content")
            options.add_argument("--disable-features=TranslateUI")
            
        if self.stealth_level > 0.7:
            options.add_argument("--disable-ipc-flooding-protection")
            options.add_argument("--disable-renderer-backgrounding")
            options.add_argument("--disable-backgrounding-occluded-windows")
            options.add_argument("--disable-client-side-phishing-detection")
            
        # Headless mode configuration
        if self.config.get("headless", True):
            options.add_argument("--headless=new")
            
        return options
        
    async def _randomize_viewport(self):
        """Randomize viewport size to avoid fingerprinting"""
        
        if not self.fingerprint_randomization:
            return
            
        # Common viewport sizes
        viewports = [
            (1920, 1080), (1366, 768), (1440, 900),
            (1536, 864), (1280, 720), (1600, 900)
        ]
        
        width, height = random.choice(viewports)
        self.driver.set_window_size(width, height)
        
    async def _inject_stealth_scripts(self):
        """Inject JavaScript to further enhance stealth"""
        
        stealth_scripts = [
            # Remove webdriver property
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})",
            
            # Randomize screen properties
            f"Object.defineProperty(screen, 'availHeight', {{get: () => {random.randint(700, 1080)}}});",
            f"Object.defineProperty(screen, 'availWidth', {{get: () => {random.randint(1200, 1920)}}});",
            
            # Mock plugins
            """
            Object.defineProperty(navigator, 'plugins', {
                get: () => [
                    {name: 'Chrome PDF Plugin', description: 'Portable Document Format'},
                    {name: 'Chrome PDF Viewer', description: 'PDF Viewer'},
                    {name: 'Native Client', description: 'Native Client'}
                ]
            });
            """,
            
            # Mock languages
            "Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});",
        ]
        
        for script in stealth_scripts:
            try:
                self.driver.execute_script(script)
            except Exception as e:
                logger.warning(f"Could not inject stealth script: {e}")
                
    async def _human_delay(self):
        """Add human-like delays between actions"""
        
        if not self.human_timing:
            return
            
        delay = random.uniform(self.min_action_delay, self.max_action_delay)
        await asyncio.sleep(delay)
        
    async def _simulate_human_browsing(self):
        """Simulate human-like browsing behavior"""
        
        if not self.human_timing:
            return
            
        # Random scroll behavior
        if random.random() < 0.7:  # 70% chance to scroll
            await self._human_scroll()
            
        # Random mouse movements (simulated with small page interactions)
        if random.random() < 0.3:  # 30% chance for mouse movement simulation
            await self._simulate_mouse_movement()
            
    async def _human_scroll(self):
        """Simulate human-like scrolling behavior"""
        
        try:
            # Get page height
            page_height = self.driver.execute_script("return document.body.scrollHeight")
            viewport_height = self.driver.execute_script("return window.innerHeight")
            
            if page_height <= viewport_height:
                return  # No need to scroll
                
            # Simulate gradual scrolling
            scroll_positions = []
            current_pos = 0
            
            while current_pos < page_height - viewport_height:
                scroll_distance = random.randint(100, 300)
                current_pos = min(current_pos + scroll_distance, page_height - viewport_height)
                scroll_positions.append(current_pos)
                
            # Perform scrolling with delays
            for pos in scroll_positions:
                self.driver.execute_script(f"window.scrollTo(0, {pos});")
                delay = random.uniform(*self.scroll_delay_range)
                await asyncio.sleep(delay)
                
            # Scroll back to top sometimes
            if random.random() < 0.3:
                await asyncio.sleep(random.uniform(1, 2))
                self.driver.execute_script("window.scrollTo(0, 0);")
                
        except Exception as e:
            logger.warning(f"Error during human scroll simulation: {e}")
            
    async def _simulate_mouse_movement(self):
        """Simulate mouse movement by focusing on random elements"""
        
        try:
            # Find clickable elements
            clickable_elements = self.driver.find_elements(
                By.CSS_SELECTOR, 
                "a, button, input, [onclick], [role='button']"
            )
            
            if clickable_elements:
                # Focus on a random element without clicking
                element = random.choice(clickable_elements)
                self.driver.execute_script("arguments[0].focus();", element)
                await asyncio.sleep(random.uniform(0.1, 0.5))
                
        except Exception as e:
            logger.warning(f"Error during mouse movement simulation: {e}")
            
    async def _google_search(self, query: str) -> Dict[str, Any]:
        """Perform Google search with stealth"""
        
        try:
            # Find search box
            search_box = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.NAME, "q"))
            )
            
            # Type query with human-like timing
            await self._human_type(search_box, query)
            
            # Submit search
            search_box.submit()
            
            # Wait for results
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.g"))
            )
            
            # Extract results
            results = []
            result_elements = self.driver.find_elements(By.CSS_SELECTOR, "div.g")
            
            for element in result_elements[:10]:  # Top 10 results
                try:
                    title_elem = element.find_element(By.CSS_SELECTOR, "h3")
                    url_elem = element.find_element(By.CSS_SELECTOR, "a")
                    
                    results.append({
                        "title": title_elem.text,
                        "url": url_elem.get_attribute("href"),
                        "snippet": element.text
                    })
                except:
                    continue
                    
            return {"results": results, "source": "google"}
            
        except Exception as e:
            logger.error(f"Error in Google search: {e}")
            return {"results": [], "error": str(e)}
            
    async def _bing_search(self, query: str) -> Dict[str, Any]:
        """Perform Bing search with stealth"""
        
        try:
            # Find search box
            search_box = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.NAME, "q"))
            )
            
            # Type query
            await self._human_type(search_box, query)
            search_box.submit()
            
            # Wait for results
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".b_algo"))
            )
            
            # Extract results
            results = []
            result_elements = self.driver.find_elements(By.CSS_SELECTOR, ".b_algo")
            
            for element in result_elements[:10]:
                try:
                    title_elem = element.find_element(By.CSS_SELECTOR, "h2 a")
                    results.append({
                        "title": title_elem.text,
                        "url": title_elem.get_attribute("href"),
                        "snippet": element.text
                    })
                except:
                    continue
                    
            return {"results": results, "source": "bing"}
            
        except Exception as e:
            logger.error(f"Error in Bing search: {e}")
            return {"results": [], "error": str(e)}
            
    async def _duckduckgo_search(self, query: str) -> Dict[str, Any]:
        """Perform DuckDuckGo search with stealth"""
        
        try:
            # Find search box
            search_box = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.NAME, "q"))
            )
            
            # Type query
            await self._human_type(search_box, query)
            search_box.submit()
            
            # Wait for results
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".result"))
            )
            
            # Extract results
            results = []
            result_elements = self.driver.find_elements(By.CSS_SELECTOR, ".result")
            
            for element in result_elements[:10]:
                try:
                    title_elem = element.find_element(By.CSS_SELECTOR, ".result__title a")
                    results.append({
                        "title": title_elem.text,
                        "url": title_elem.get_attribute("href"),
                        "snippet": element.text
                    })
                except:
                    continue
                    
            return {"results": results, "source": "duckduckgo"}
            
        except Exception as e:
            logger.error(f"Error in DuckDuckGo search: {e}")
            return {"results": [], "error": str(e)}
            
    async def _human_type(self, element, text: str):
        """Type text with human-like timing"""
        
        element.clear()
        
        if not self.human_timing:
            element.send_keys(text)
            return
            
        # Type character by character with random delays
        for char in text:
            element.send_keys(char)
            delay = random.uniform(*self.typing_delay_range)
            await asyncio.sleep(delay)