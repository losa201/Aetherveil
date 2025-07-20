"""
Advanced LLM Collaboration Hub for Chimera
Multi-provider integration with sophisticated rapport building and adaptive prompt engineering
Integrates with existing PKB while adding advanced features like personality adaptation and web LLM support
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import aiohttp
import json
import random

# Optional LLM provider imports for advanced features
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from enum import Enum
from dataclasses import dataclass
import hashlib
import re

from ..core.events import EventSystem, EventType, EventEmitter
from .prompt_knowledge_base import PromptKnowledgeBase
from ..web.stealth import StealthBrowser

logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    """Available LLM providers"""
    OPENAI_GPT4 = "openai_gpt4"
    OPENAI_GPT35 = "openai_gpt35"
    ANTHROPIC_CLAUDE = "anthropic_claude"
    ANTHROPIC_HAIKU = "anthropic_haiku"
    LOCAL_MODEL = "local_model"
    WEB_CLAUDE = "web_claude"
    WEB_CHATGPT = "web_chatgpt"
    WEB_GEMINI = "web_gemini"

class PersonalityType(Enum):
    """Target personality types for rapport building"""
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    PRAGMATIC = "pragmatic"
    SOCIAL = "social"
    CAUTIOUS = "cautious"
    AGGRESSIVE = "aggressive"

@dataclass
class ConversationContext:
    """Context for ongoing conversations with targets"""
    
    target_id: str
    session_id: str
    personality_profile: PersonalityType
    conversation_history: List[Dict[str, str]]
    rapport_score: float  # 0.0 to 1.0
    trust_level: float   # 0.0 to 1.0
    expertise_areas: List[str]
    detected_interests: List[str]
    communication_style: Dict[str, Any]
    last_interaction: datetime
    success_metrics: Dict[str, float]

@dataclass
class LLMResponse:
    """Enhanced LLM response with metadata"""
    
    content: str
    provider: str
    confidence: float
    processing_time: float
    token_count: int
    cost_estimate: float
    safety_score: float
    success_probability: float
    rapport_indicators: Dict[str, Any] = None

class LLMCollaborator(EventEmitter):
    """
    Advanced LLM collaboration agent with neuroplastic prompt optimization and rapport building
    
    Enhanced Features:
    - Advanced prompt knowledge base integration
    - Multi-provider performance tracking with API and web LLM support
    - Adaptive provider selection with circuit breaker patterns
    - Response quality assessment and rapport building
    - Context-aware prompt crafting with personality adaptation
    - Conversation state management for long-term interactions
    - Real-time LLM provider fallback and optimization
    """
    
    def __init__(self, config, event_system: EventSystem):
        super().__init__(event_system, "LLMCollaborator")
        
        self.config = config
        self.pkb = PromptKnowledgeBase(config, event_system)
        
        # Advanced conversation management
        self.active_conversations: Dict[str, ConversationContext] = {}
        self.stealth_browser: Optional[StealthBrowser] = None
        
        # API clients for enhanced providers
        self.openai_client = None
        self.anthropic_client = None
        self.local_models = {}
        
        # Enhanced provider configurations with API and web support
        self.providers = {
            "openai_gpt4": {
                "enabled": OPENAI_AVAILABLE and config.get("llm.openai.api_key"),
                "type": "api",
                "model": "gpt-4",
                "reliability": 0.95,
                "avg_quality": 0.9,
                "cost_factor": 1.5,
                "personality_strength": {
                    PersonalityType.CREATIVE: 0.9,
                    PersonalityType.ANALYTICAL: 0.8
                }
            },
            "anthropic_claude": {
                "enabled": ANTHROPIC_AVAILABLE and config.get("llm.anthropic.api_key"),
                "type": "api", 
                "model": "claude-3-opus-20240229",
                "reliability": 0.93,
                "avg_quality": 0.92,
                "cost_factor": 1.2,
                "personality_strength": {
                    PersonalityType.ANALYTICAL: 0.95,
                    PersonalityType.CAUTIOUS: 0.9
                }
            },
            "claude_web": {
                "enabled": True,
                "type": "web",
                "base_url": "https://claude.ai",
                "reliability": 0.9,
                "avg_quality": 0.85,
                "cost_factor": 1.0,
                "personality_strength": {
                    PersonalityType.ANALYTICAL: 0.9,
                    PersonalityType.PRAGMATIC: 0.8
                }
            },
            "chatgpt_web": {
                "enabled": True, 
                "type": "web",
                "base_url": "https://chat.openai.com",
                "reliability": 0.85,
                "avg_quality": 0.8,
                "cost_factor": 0.8,
                "personality_strength": {
                    PersonalityType.CREATIVE: 0.85,
                    PersonalityType.SOCIAL: 0.8
                }
            },
            "gemini_web": {
                "enabled": True,
                "type": "web",
                "base_url": "https://gemini.google.com",
                "reliability": 0.8,
                "avg_quality": 0.75,
                "cost_factor": 0.6,
                "personality_strength": {
                    PersonalityType.PRAGMATIC: 0.8,
                    PersonalityType.AGGRESSIVE: 0.7
                }
            },
            "local_model": {
                "enabled": TRANSFORMERS_AVAILABLE and config.get("llm.local_models.enabled", False),
                "type": "local",
                "reliability": 0.7,
                "avg_quality": 0.65,
                "cost_factor": 0.0,
                "personality_strength": {
                    PersonalityType.CAUTIOUS: 0.8
                }
            }
        }
        
        # Adaptive parameters
        self.fallback_enabled = config.get("llm.fallback_enabled", True)
        self.confidence_threshold = config.get("llm.confidence_threshold", 0.8)
        self.max_retries = 3
        self.circuit_breaker = {}  # Provider circuit breaker states
        
        # Performance tracking
        self.provider_stats = {provider: {"requests": 0, "successes": 0, "avg_time": 0.0} 
                              for provider in self.providers}
        self.recent_interactions = []
        
    async def initialize(self):
        """Initialize advanced LLM collaborator with enhanced providers"""
        
        await self.pkb.initialize()
        
        # Initialize API clients
        await self._initialize_api_clients()
        
        # Initialize web browser for web-based LLMs
        if any(p["type"] == "web" and p["enabled"] for p in self.providers.values()):
            await self._initialize_web_browser()
        
        # Initialize local models if enabled
        if any(p["type"] == "local" and p["enabled"] for p in self.providers.values()):
            await self._initialize_local_models()
        
        # Initialize circuit breakers
        for provider in self.providers:
            self.circuit_breaker[provider] = {
                "state": "closed",  # closed, open, half_open
                "failure_count": 0,
                "last_failure": None,
                "success_count": 0
            }
            
        await self.emit_event(
            EventType.REASONING_START,
            {"message": "Enhanced LLM collaborator initialized"}
        )
        
        logger.info("Enhanced LLM collaborator initialized")
        
    async def get_tactical_advice(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get tactical advice using optimized prompting strategies
        
        Args:
            query: The question or request
            context: Additional context (target, phase, constraints, etc.)
            
        Returns:
            Enhanced response with quality metrics and provider info
        """
        
        start_time = time.time()
        context = context or {}
        
        # Determine category and context type
        category = self._categorize_query(query)
        context_type = context.get("phase", "general")
        
        # Select optimal provider
        provider = await self._select_optimal_provider(category, context_type)
        
        # Get optimal prompt from PKB
        variables = self._extract_variables(query, context)
        prompt_text, template_id = await self.pkb.get_optimal_prompt(
            category, context_type, provider, variables
        )
        
        # Execute query with selected provider
        response_data = await self._execute_query(provider, prompt_text, context)
        
        # Assess response quality
        quality_score = await self._assess_response_quality(response_data, query, context)
        
        # Record outcome in PKB
        response_time = time.time() - start_time
        await self.pkb.record_outcome(
            template_id=template_id,
            provider=provider,
            context=context_type,
            variables=variables,
            response_time=response_time,
            response_quality=quality_score,
            success=response_data.get("success", False),
            response_length=len(response_data.get("response", "")),
            error_message=response_data.get("error")
        )
        
        # Update provider statistics
        await self._update_provider_stats(provider, response_data["success"], response_time)
        
        # Enhance response with metadata
        enhanced_response = {
            "advice": response_data.get("response", ""),
            "confidence": quality_score,
            "source": f"llm_{provider}",
            "provider": provider,
            "template_used": template_id,
            "response_time": response_time,
            "context_category": category,
            "success": response_data.get("success", False),
            "metadata": {
                "prompt_optimization": await self._get_prompt_insights(template_id),
                "provider_performance": self.provider_stats[provider],
                "alternatives_available": len([p for p in self.providers if self.providers[p]["enabled"]])
            }
        }
        
        await self.emit_event(
            EventType.LLM_ADVICE_RECEIVED,
            {
                "provider": provider,
                "category": category,
                "quality": quality_score,
                "response_time": response_time
            }
        )
        
        return enhanced_response
    
    async def start_conversation(self, target_id: str, initial_context: Dict[str, Any] = None) -> str:
        """Start a new conversation with intelligent target profiling"""
        
        session_id = hashlib.md5(f"{target_id}_{datetime.utcnow().isoformat()}".encode()).hexdigest()[:12]
        initial_context = initial_context or {}
        
        # Analyze target and determine optimal personality approach
        personality_profile = await self._analyze_target_personality(initial_context)
        
        # Create conversation context
        context = ConversationContext(
            target_id=target_id,
            session_id=session_id,
            personality_profile=personality_profile,
            conversation_history=[],
            rapport_score=0.0,
            trust_level=0.0,
            expertise_areas=[],
            detected_interests=[],
            communication_style={},
            last_interaction=datetime.utcnow(),
            success_metrics={}
        )
        
        self.active_conversations[session_id] = context
        
        await self.emit_event(
            EventType.REASONING_START,
            {"action": "conversation_started", "session_id": session_id, "personality": personality_profile.value}
        )
        
        logger.info(f"Started conversation {session_id} with {personality_profile.value} approach")
        
        return session_id
    
    async def interact_with_target(self, session_id: str, query: str, context: Dict[str, Any] = None) -> LLMResponse:
        """Perform intelligent interaction with target using optimal provider and personality adaptation"""
        
        if session_id not in self.active_conversations:
            raise ValueError(f"No active conversation found for session {session_id}")
        
        conversation = self.active_conversations[session_id]
        context = context or {}
        
        # Select optimal provider based on personality and conversation state
        provider = await self._select_personality_aware_provider(conversation, query)
        
        # Generate personality-adapted prompt
        adapted_prompt = await self._generate_personality_adapted_prompt(query, conversation, context)
        
        # Execute query with enhanced response processing
        start_time = time.time()
        response_data = await self._execute_enhanced_query(provider, adapted_prompt, conversation, context)
        response_time = time.time() - start_time
        
        # Assess response quality and rapport indicators
        quality_score = await self._assess_response_quality(response_data, query, context)
        rapport_indicators = await self._analyze_rapport_indicators(response_data.get("response", ""))
        
        # Update conversation state
        await self._update_conversation_state(conversation, query, response_data, rapport_indicators)
        
        # Create enhanced response
        enhanced_response = LLMResponse(
            content=response_data.get("response", ""),
            provider=provider,
            confidence=quality_score,
            processing_time=response_time,
            token_count=len(response_data.get("response", "").split()),
            cost_estimate=await self._estimate_cost(provider, response_data.get("response", "")),
            safety_score=await self._calculate_safety_score(response_data.get("response", "")),
            success_probability=quality_score * conversation.rapport_score,
            rapport_indicators=rapport_indicators
        )
        
        # Record outcome for learning
        await self.pkb.record_outcome(
            template_id="personality_adapted",
            provider=provider,
            context=conversation.personality_profile.value,
            variables={"query": query, "session_id": session_id},
            response_time=response_time,
            response_quality=quality_score,
            success=response_data.get("success", False),
            response_length=len(response_data.get("response", "")),
            error_message=response_data.get("error")
        )
        
        return enhanced_response
    
    async def build_rapport(self, session_id: str, target_response: str) -> Dict[str, Any]:
        """Analyze target response and adapt rapport-building strategy"""
        
        if session_id not in self.active_conversations:
            raise ValueError(f"No active conversation found for session {session_id}")
        
        conversation = self.active_conversations[session_id]
        
        # Analyze target response for personality indicators
        personality_indicators = await self._analyze_response_personality(target_response)
        
        # Update rapport score based on response analysis
        old_rapport = conversation.rapport_score
        conversation.rapport_score = await self._calculate_rapport_score(conversation, target_response, personality_indicators)
        
        # Detect interests and expertise areas
        detected_interests = await self._extract_interests(target_response)
        conversation.detected_interests.extend(detected_interests)
        conversation.detected_interests = list(set(conversation.detected_interests))  # Remove duplicates
        
        # Update communication style preferences
        communication_style = await self._analyze_communication_style(target_response)
        conversation.communication_style.update(communication_style)
        
        # Generate rapport-building recommendations
        recommendations = await self._generate_rapport_recommendations(conversation, personality_indicators)
        
        rapport_delta = conversation.rapport_score - old_rapport
        
        await self.emit_event(
            EventType.KNOWLEDGE_LEARNED,
            {
                "session_id": session_id,
                "rapport_change": rapport_delta,
                "detected_interests": detected_interests,
                "personality_indicators": personality_indicators
            }
        )
        
        return {
            "rapport_score": conversation.rapport_score,
            "rapport_change": rapport_delta,
            "personality_indicators": personality_indicators,
            "detected_interests": detected_interests,
            "communication_style": communication_style,
            "recommendations": recommendations,
            "trust_level": conversation.trust_level
        }
        
    async def get_provider_recommendations(self, task_type: str) -> List[Dict[str, Any]]:
        """Get provider recommendations for specific task types"""
        
        recommendations = []
        
        for provider, config in self.providers.items():
            if not config["enabled"]:
                continue
                
            stats = self.provider_stats[provider]
            circuit_state = self.circuit_breaker[provider]["state"]
            
            # Calculate recommendation score
            reliability = stats["successes"] / max(stats["requests"], 1)
            response_time_score = max(0, 1.0 - (stats["avg_time"] / 10.0))
            
            score = (
                reliability * 0.4 +
                config["avg_quality"] * 0.3 +
                response_time_score * 0.2 +
                (1.0 / config["cost_factor"]) * 0.1
            )
            
            # Penalize if circuit breaker is open
            if circuit_state == "open":
                score *= 0.1
            elif circuit_state == "half_open":
                score *= 0.7
                
            recommendations.append({
                "provider": provider,
                "score": score,
                "reliability": reliability,
                "avg_response_time": stats["avg_time"],
                "circuit_state": circuit_state,
                "total_requests": stats["requests"],
                "recommendation": self._get_provider_recommendation_text(provider, score)
            })
            
        # Sort by score
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        
        return recommendations
        
    async def analyze_collaboration_effectiveness(self) -> Dict[str, Any]:
        """Analyze overall LLM collaboration effectiveness"""
        
        pkb_analysis = await self.pkb.analyze_prompt_effectiveness()
        
        # Recent interaction analysis
        recent_interactions = self.recent_interactions[-50:] if self.recent_interactions else []
        
        if recent_interactions:
            avg_quality = sum(i["quality"] for i in recent_interactions) / len(recent_interactions)
            avg_response_time = sum(i["response_time"] for i in recent_interactions) / len(recent_interactions)
            success_rate = sum(1 for i in recent_interactions if i["success"]) / len(recent_interactions)
        else:
            avg_quality = avg_response_time = success_rate = 0.0
            
        # Provider distribution
        provider_usage = {}
        for interaction in recent_interactions:
            provider = interaction["provider"]
            provider_usage[provider] = provider_usage.get(provider, 0) + 1
            
        analysis = {
            "collaboration_stats": {
                "total_interactions": len(self.recent_interactions),
                "recent_avg_quality": avg_quality,
                "recent_avg_response_time": avg_response_time,
                "recent_success_rate": success_rate,
                "provider_distribution": provider_usage
            },
            "pkb_analysis": pkb_analysis,
            "provider_health": {
                provider: {
                    "circuit_state": self.circuit_breaker[provider]["state"],
                    "failure_count": self.circuit_breaker[provider]["failure_count"],
                    "success_rate": stats["successes"] / max(stats["requests"], 1)
                }
                for provider, stats in self.provider_stats.items()
            },
            "optimization_opportunities": await self._identify_optimization_opportunities()
        }
        
        return analysis
        
    async def optimize_collaboration_strategy(self) -> Dict[str, Any]:
        """Optimize collaboration strategy based on learned patterns"""
        
        optimization_results = {
            "actions_taken": [],
            "recommendations": [],
            "performance_improvements": {}
        }
        
        # Analyze provider performance
        provider_analysis = await self.get_provider_recommendations("general")
        
        # Disable consistently failing providers
        for provider_info in provider_analysis:
            provider = provider_info["provider"]
            if provider_info["reliability"] < 0.3 and provider_info["total_requests"] > 10:
                if self.providers[provider]["enabled"]:
                    self.providers[provider]["enabled"] = False
                    optimization_results["actions_taken"].append(f"Disabled unreliable provider: {provider}")
                    
        # Re-enable providers if circuit breaker allows
        for provider, breaker in self.circuit_breaker.items():
            if breaker["state"] == "half_open" and breaker["success_count"] > 3:
                breaker["state"] = "closed"
                breaker["failure_count"] = 0
                self.providers[provider]["enabled"] = True
                optimization_results["actions_taken"].append(f"Re-enabled recovered provider: {provider}")
                
        # Get PKB suggestions
        for template_id in list(self.pkb.templates.keys())[:5]:  # Top 5 templates
            suggestions = await self.pkb.suggest_prompt_improvements(template_id)
            if "no improvements needed" not in suggestions[0].lower():
                optimization_results["recommendations"].extend(suggestions)
                
        return optimization_results
        
    # Private methods
    
    def _categorize_query(self, query: str) -> str:
        """Categorize query for optimal prompt selection"""
        
        query_lower = query.lower()
        
        # Keyword-based categorization
        if any(word in query_lower for word in ["recon", "reconnaissance", "enumerate", "discover"]):
            return "reconnaissance"
        elif any(word in query_lower for word in ["vulnerability", "vuln", "weakness", "exploit"]):
            return "vulnerability_analysis"  
        elif any(word in query_lower for word in ["attack", "exploit", "penetrate", "compromise"]):
            return "exploitation_planning"
        elif any(word in query_lower for word in ["tool", "scanner", "utility", "software"]):
            return "tool_selection"
        elif any(word in query_lower for word in ["report", "document", "findings", "analysis"]):
            return "report_analysis"
        else:
            return "general_advice"
            
    def _extract_variables(self, query: str, context: Dict[str, Any]) -> Dict[str, str]:
        """Extract variables from query and context for template filling"""
        
        variables = {
            "query": query,
            "target": context.get("target", "the target system"),
            "phase": context.get("phase", "analysis"),
            "constraints": context.get("constraints", "standard ethical guidelines"),
            "focus_area": context.get("focus_area", "general security assessment"),
            "methodology": context.get("methodology", "comprehensive testing")
        }
        
        # Extract specific entities from query
        import re
        
        # Extract URLs/domains
        url_pattern = r'https?://[^\s]+'
        urls = re.findall(url_pattern, query)
        if urls:
            variables["target"] = urls[0]
            
        # Extract IP addresses
        ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        ips = re.findall(ip_pattern, query)
        if ips:
            variables["target"] = ips[0]
            
        return variables
        
    async def _select_optimal_provider(self, category: str, context_type: str) -> str:
        """Select optimal provider based on category, context, and current performance"""
        
        # Get available providers (circuit breaker check)
        available_providers = []
        for provider, config in self.providers.items():
            if not config["enabled"]:
                continue
                
            breaker = self.circuit_breaker[provider]
            if breaker["state"] == "open":
                # Check if enough time has passed to try half-open
                if breaker["last_failure"] and (datetime.utcnow() - breaker["last_failure"]).seconds > 300:
                    breaker["state"] = "half_open"
                    available_providers.append(provider)
            else:
                available_providers.append(provider)
                
        if not available_providers:
            # Emergency fallback - try any provider
            available_providers = list(self.providers.keys())
            
        # Score providers for this specific task
        provider_scores = []
        for provider in available_providers:
            config = self.providers[provider]
            stats = self.provider_stats[provider]
            
            # Base score from configuration
            base_score = config["avg_quality"] * config["reliability"]
            
            # Adjust for recent performance
            if stats["requests"] > 0:
                recent_reliability = stats["successes"] / stats["requests"]
                response_time_factor = max(0.1, 1.0 - (stats["avg_time"] / 20.0))
                base_score = base_score * 0.5 + (recent_reliability * response_time_factor) * 0.5
                
            # Circuit breaker penalty
            if self.circuit_breaker[provider]["state"] == "half_open":
                base_score *= 0.8
                
            provider_scores.append((base_score, provider))
            
        # Sort by score and select best
        provider_scores.sort(reverse=True)
        selected_provider = provider_scores[0][1]
        
        logger.debug(f"Selected provider {selected_provider} for category {category}")
        
        return selected_provider
        
    async def _execute_query(self, provider: str, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute query against selected provider"""
        
        try:
            # Simulate LLM query execution
            # In real implementation, this would make actual API calls or web automation
            
            await asyncio.sleep(random.uniform(1.0, 4.0))  # Simulate network delay
            
            # Simulate different provider response characteristics
            if provider == "claude_web":
                response = f"Claude analysis: {prompt[:100]}... [Detailed analytical response would be here]"
                success_rate = 0.9
            elif provider == "chatgpt_web":
                response = f"GPT-4 advice: {prompt[:100]}... [Creative problem-solving response would be here]"
                success_rate = 0.85
            elif provider == "gemini_web":
                response = f"Gemini insights: {prompt[:100]}... [Research-focused response would be here]"
                success_rate = 0.8
            else:
                response = f"Generic LLM response: {prompt[:100]}..."
                success_rate = 0.7
                
            # Simulate occasional failures
            success = random.random() < success_rate
            
            if success:
                return {
                    "response": response,
                    "success": True,
                    "provider": provider
                }
            else:
                return {
                    "response": "",
                    "success": False,
                    "error": "Simulated provider failure",
                    "provider": provider
                }
                
        except Exception as e:
            logger.error(f"Error executing query with {provider}: {e}")
            return {
                "response": "",
                "success": False,
                "error": str(e),
                "provider": provider
            }
            
    async def _assess_response_quality(self, response_data: Dict[str, Any], 
                                     original_query: str, context: Dict[str, Any]) -> float:
        """Assess the quality of an LLM response"""
        
        if not response_data.get("success", False):
            return 0.0
            
        response = response_data.get("response", "")
        
        if not response:
            return 0.0
            
        # Quality assessment heuristics
        quality_factors = []
        
        # Length appropriateness (not too short, not excessively long)
        length_score = min(1.0, len(response) / 500.0)  # Optimal around 500 chars
        if len(response) > 2000:
            length_score *= 0.8  # Penalty for being too verbose
        quality_factors.append(length_score)
        
        # Relevance to query (keyword overlap)
        query_words = set(original_query.lower().split())
        response_words = set(response.lower().split())
        relevance_score = len(query_words.intersection(response_words)) / max(len(query_words), 1)
        quality_factors.append(relevance_score)
        
        # Technical depth indicators
        technical_terms = ["vulnerability", "exploit", "security", "attack", "analysis", "recommendation"]
        technical_score = sum(1 for term in technical_terms if term in response.lower()) / len(technical_terms)
        quality_factors.append(technical_score)
        
        # Structure indicators (bullets, numbers, clear organization)
        structure_indicators = ["-", "1.", "2.", "•", ":", "however", "therefore", "additionally"]
        structure_score = min(1.0, sum(1 for indicator in structure_indicators if indicator in response) / 5.0)
        quality_factors.append(structure_score)
        
        # Actionability (contains verbs and recommendations)
        action_words = ["should", "recommend", "suggest", "consider", "implement", "test", "analyze"]
        action_score = min(1.0, sum(1 for word in action_words if word in response.lower()) / 3.0)
        quality_factors.append(action_score)
        
        # Average quality factors
        overall_quality = sum(quality_factors) / len(quality_factors)
        
        return min(overall_quality, 1.0)
        
    async def _update_provider_stats(self, provider: str, success: bool, response_time: float):
        """Update provider performance statistics"""
        
        stats = self.provider_stats[provider]
        stats["requests"] += 1
        
        if success:
            stats["successes"] += 1
            # Update circuit breaker
            breaker = self.circuit_breaker[provider]
            if breaker["state"] == "half_open":
                breaker["success_count"] += 1
            else:
                breaker["failure_count"] = max(0, breaker["failure_count"] - 1)
        else:
            # Update circuit breaker
            breaker = self.circuit_breaker[provider]
            breaker["failure_count"] += 1
            breaker["last_failure"] = datetime.utcnow()
            
            # Open circuit if too many failures
            if breaker["failure_count"] >= 5:
                breaker["state"] = "open"
                self.providers[provider]["enabled"] = False
                logger.warning(f"Circuit breaker opened for provider {provider}")
                
        # Update average response time
        if stats["requests"] == 1:
            stats["avg_time"] = response_time
        else:
            # Exponential moving average
            alpha = 0.1
            stats["avg_time"] = (1 - alpha) * stats["avg_time"] + alpha * response_time
            
        # Track recent interactions
        self.recent_interactions.append({
            "provider": provider,
            "success": success,
            "response_time": response_time,
            "timestamp": datetime.utcnow(),
            "quality": 0.8 if success else 0.0  # Simplified for this update
        })
        
        # Limit history
        if len(self.recent_interactions) > 100:
            self.recent_interactions = self.recent_interactions[-50:]
            
    async def _get_prompt_insights(self, template_id: str) -> Dict[str, Any]:
        """Get insights about prompt performance"""
        
        if template_id not in self.pkb.templates:
            return {}
            
        template = self.pkb.templates[template_id]
        suggestions = await self.pkb.suggest_prompt_improvements(template_id)
        
        return {
            "template_id": template_id,
            "success_rate": template.success_rate,
            "usage_count": template.usage_count,
            "suggestions": suggestions[:3],  # Top 3 suggestions
            "provider_performance": template.provider_performance
        }
        
    def _get_provider_recommendation_text(self, provider: str, score: float) -> str:
        """Generate recommendation text for provider"""
        
        if score > 0.8:
            return f"Excellent choice for most tasks"
        elif score > 0.6:
            return f"Good option with reliable performance"
        elif score > 0.4:
            return f"Acceptable for non-critical tasks"
        else:
            return f"Consider alternatives due to performance issues"
            
    async def _identify_optimization_opportunities(self) -> List[str]:
        """Identify opportunities for optimization"""
        
        opportunities = []
        
        # Check for underutilized providers
        total_requests = sum(stats["requests"] for stats in self.provider_stats.values())
        if total_requests > 0:
            for provider, stats in self.provider_stats.items():
                usage_ratio = stats["requests"] / total_requests
                if usage_ratio < 0.1 and self.providers[provider]["enabled"]:
                    opportunities.append(f"Provider {provider} is underutilized - consider load balancing")
                    
        # Check for prompt template gaps
        pkb_analysis = await self.pkb.analyze_prompt_effectiveness()
        if pkb_analysis.get("total_prompts", 0) < 50:
            opportunities.append("Expand prompt template library for better coverage")
            
        # Check response time patterns
        avg_times = [stats["avg_time"] for stats in self.provider_stats.values() if stats["requests"] > 0]
        if avg_times and max(avg_times) > 8.0:
            opportunities.append("Some providers have slow response times - consider optimization")
            
        return opportunities
        
    # Advanced helper methods for enhanced functionality
    
    async def _initialize_api_clients(self):
        """Initialize API clients for LLM providers"""
        
        # OpenAI client
        if OPENAI_AVAILABLE and self.config.get("llm.openai.api_key"):
            openai.api_key = self.config.get("llm.openai.api_key")
            self.openai_client = openai
            logger.info("OpenAI client initialized")
        
        # Anthropic client
        if ANTHROPIC_AVAILABLE and self.config.get("llm.anthropic.api_key"):
            self.anthropic_client = anthropic.Anthropic(
                api_key=self.config.get("llm.anthropic.api_key")
            )
            logger.info("Anthropic client initialized")
    
    async def _initialize_web_browser(self):
        """Initialize stealth browser for web-based LLM interaction"""
        
        browser_config = {
            "stealth_level": 0.9,
            "human_timing": True,
            "fingerprint_randomization": True,
            "headless": self.config.get("llm.web_browser.headless", True)
        }
        
        self.stealth_browser = StealthBrowser(browser_config)
        await self.stealth_browser.initialize()
        
        logger.info("Stealth browser initialized for web LLM access")
    
    async def _initialize_local_models(self):
        """Initialize local transformer models"""
        
        model_configs = self.config.get("llm.local_models.models", [])
        
        for model_config in model_configs:
            try:
                model_name = model_config["name"]
                model_path = model_config["path"]
                
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                
                self.local_models[model_name] = {
                    "tokenizer": tokenizer,
                    "model": model,
                    "pipeline": pipeline("text-generation", model=model, tokenizer=tokenizer)
                }
                
                logger.info(f"Local model {model_name} initialized")
                
            except Exception as e:
                logger.warning(f"Failed to initialize local model {model_config.get('name', 'unknown')}: {e}")
    
    async def _analyze_target_personality(self, context: Dict[str, Any]) -> PersonalityType:
        """Analyze target and determine personality type for optimal approach"""
        
        text_indicators = context.get("text_sample", "").lower()
        behavioral_indicators = context.get("behavioral_data", {})
        
        personality_scores = {
            PersonalityType.ANALYTICAL: 0,
            PersonalityType.CREATIVE: 0,
            PersonalityType.PRAGMATIC: 0,
            PersonalityType.SOCIAL: 0,
            PersonalityType.CAUTIOUS: 0,
            PersonalityType.AGGRESSIVE: 0
        }
        
        # Analyze text for personality indicators
        if text_indicators:
            if any(word in text_indicators for word in ["data", "analysis", "systematic", "logical", "research"]):
                personality_scores[PersonalityType.ANALYTICAL] += 2
            
            if any(word in text_indicators for word in ["creative", "innovative", "artistic", "unique", "design"]):
                personality_scores[PersonalityType.CREATIVE] += 2
            
            if any(word in text_indicators for word in ["practical", "efficient", "results", "direct", "business"]):
                personality_scores[PersonalityType.PRAGMATIC] += 2
            
            if any(word in text_indicators for word in ["people", "team", "social", "community", "collaboration"]):
                personality_scores[PersonalityType.SOCIAL] += 2
            
            if any(word in text_indicators for word in ["careful", "safe", "cautious", "security", "risk"]):
                personality_scores[PersonalityType.CAUTIOUS] += 2
            
            if any(word in text_indicators for word in ["aggressive", "competitive", "assertive", "bold", "fast"]):
                personality_scores[PersonalityType.AGGRESSIVE] += 2
        
        # Return personality type with highest score
        max_personality = max(personality_scores.items(), key=lambda x: x[1])
        
        # If scores are tied, default to analytical
        if max_personality[1] == 0:
            return PersonalityType.ANALYTICAL
        
        return max_personality[0]
    
    async def _select_personality_aware_provider(self, conversation: ConversationContext, query: str) -> str:
        """Select optimal provider based on personality and conversation context"""
        
        available_providers = [p for p, config in self.providers.items() 
                             if config["enabled"] and self.circuit_breaker[p]["state"] != "open"]
        
        if not available_providers:
            # Emergency fallback
            available_providers = [p for p, config in self.providers.items() if config["enabled"]]
        
        # Score providers for this personality type
        provider_scores = {}
        
        for provider in available_providers:
            config = self.providers[provider]
            stats = self.provider_stats[provider]
            
            # Base score
            base_score = config["avg_quality"] * config["reliability"]
            
            # Personality strength bonus
            personality_strength = config.get("personality_strength", {})
            if conversation.personality_profile in personality_strength:
                base_score *= personality_strength[conversation.personality_profile]
            
            # Recent performance adjustment
            if stats["requests"] > 0:
                recent_reliability = stats["successes"] / stats["requests"]
                base_score = base_score * 0.7 + recent_reliability * 0.3
            
            # Cost factor (lower cost = higher score if cost optimization enabled)
            if self.config.get("llm.cost_optimization", True):
                base_score /= config["cost_factor"]
            
            provider_scores[provider] = base_score
        
        # Select best provider
        best_provider = max(provider_scores.items(), key=lambda x: x[1])[0]
        return best_provider
    
    async def _generate_personality_adapted_prompt(self, query: str, conversation: ConversationContext, context: Dict[str, Any]) -> str:
        """Generate prompt adapted to target personality"""
        
        # Get base prompt from PKB
        category = self._categorize_query(query)
        variables = self._extract_variables(query, context)
        prompt_text, _ = await self.pkb.get_optimal_prompt(
            category, conversation.personality_profile.value, 
            "general", variables
        )
        
        # Add personality-specific modifications
        personality_modifiers = {
            PersonalityType.ANALYTICAL: "Please provide detailed analysis with step-by-step reasoning and data-driven insights.",
            PersonalityType.CREATIVE: "Feel free to suggest innovative or unconventional approaches. Think outside the box.",
            PersonalityType.PRAGMATIC: "Focus on practical, actionable recommendations that deliver results efficiently.",
            PersonalityType.SOCIAL: "Consider the human factors, team dynamics, and interpersonal aspects.",
            PersonalityType.CAUTIOUS: "Emphasize safety, ethics, risk mitigation, and conservative approaches.",
            PersonalityType.AGGRESSIVE: "Provide direct, assertive strategies for maximum impact and quick results."
        }
        
        if conversation.personality_profile in personality_modifiers:
            prompt_text += f"\n\n{personality_modifiers[conversation.personality_profile]}"
        
        # Add conversation context if available
        if conversation.conversation_history:
            recent_history = conversation.conversation_history[-2:]  # Last 2 exchanges
            context_summary = "\n".join([f"Previous: {h['query'][:50]} -> {h['response'][:50]}..." for h in recent_history])
            prompt_text = f"Context: {context_summary}\n\nCurrent: {prompt_text}"
        
        # Add detected interests for rapport building
        if conversation.detected_interests:
            interests = ", ".join(conversation.detected_interests[:3])  # Top 3 interests
            prompt_text += f"\n\nNote: The person has shown interest in {interests}. Consider incorporating relevant examples."
        
        return prompt_text
    
    async def _execute_enhanced_query(self, provider: str, prompt: str, conversation: ConversationContext, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute query with enhanced provider-specific handling"""
        
        provider_config = self.providers[provider]
        provider_type = provider_config["type"]
        
        try:
            if provider_type == "api":
                if provider == "openai_gpt4" and self.openai_client:
                    return await self._query_openai(provider, prompt)
                elif provider == "anthropic_claude" and self.anthropic_client:
                    return await self._query_anthropic(provider, prompt)
            elif provider_type == "local":
                return await self._query_local_model(prompt)
            elif provider_type == "web":
                return await self._query_web_llm(provider, prompt)
            else:
                # Fallback to original execution method
                return await self._execute_query(provider, prompt, context)
                
        except Exception as e:
            logger.error(f"Enhanced query execution failed for {provider}: {e}")
            return {
                "response": "",
                "success": False,
                "error": str(e),
                "provider": provider
            }
    
    async def _query_openai(self, provider: str, prompt: str) -> Dict[str, Any]:
        """Query OpenAI models via API"""
        
        model_name = self.providers[provider]["model"]
        
        response = await self.openai_client.ChatCompletion.acreate(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.7
        )
        
        content = response.choices[0].message.content
        token_count = response.usage.total_tokens
        
        return {
            "response": content,
            "success": True,
            "provider": provider,
            "token_count": token_count
        }
    
    async def _query_anthropic(self, provider: str, prompt: str) -> Dict[str, Any]:
        """Query Anthropic models via API"""
        
        model_name = self.providers[provider]["model"]
        
        response = await self.anthropic_client.messages.create(
            model=model_name,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        content = response.content[0].text
        token_count = response.usage.input_tokens + response.usage.output_tokens
        
        return {
            "response": content,
            "success": True,
            "provider": provider,
            "token_count": token_count
        }
    
    async def _query_local_model(self, prompt: str) -> Dict[str, Any]:
        """Query local transformer model"""
        
        if not self.local_models:
            raise RuntimeError("No local models available")
        
        # Use first available local model
        model_name = list(self.local_models.keys())[0]
        model_data = self.local_models[model_name]
        
        # Generate response
        response = model_data["pipeline"](
            prompt,
            max_new_tokens=500,
            temperature=0.7,
            do_sample=True,
            pad_token_id=model_data["tokenizer"].eos_token_id
        )
        
        content = response[0]["generated_text"]
        
        # Remove original prompt from response
        if content.startswith(prompt):
            content = content[len(prompt):].strip()
        
        return {
            "response": content,
            "success": True,
            "provider": "local_model",
            "token_count": len(content.split())
        }
    
    async def _query_web_llm(self, provider: str, prompt: str) -> Dict[str, Any]:
        """Query web-based LLM interfaces using stealth browser"""
        
        if not self.stealth_browser:
            raise RuntimeError("Stealth browser not initialized")
        
        provider_config = self.providers[provider]
        url = provider_config["base_url"]
        
        try:
            # Navigate to the LLM interface
            await self.stealth_browser.navigate_to(url)
            
            # Wait for page to load
            await asyncio.sleep(3)
            
            # This would implement actual web automation
            # For now, return enhanced simulation
            
            return {
                "response": f"[Enhanced {provider} Response] Analyzing your request: {prompt[:100]}... Based on current cybersecurity best practices and threat landscape analysis, here are my recommendations...",
                "success": True,
                "provider": provider,
                "token_count": len(prompt.split()) * 2
            }
            
        except Exception as e:
            logger.error(f"Error querying web LLM {provider}: {e}")
            raise
    
    async def _analyze_rapport_indicators(self, response: str) -> Dict[str, Any]:
        """Analyze response for rapport-building indicators"""
        
        indicators = {
            "emotional_tone": "neutral",
            "engagement_level": 0.5,
            "expertise_demonstration": False,
            "personal_connection": False,
            "trust_signals": []
        }
        
        response_lower = response.lower()
        
        # Analyze emotional tone
        positive_words = ["excellent", "great", "wonderful", "impressive", "outstanding"]
        negative_words = ["concerning", "problematic", "difficult", "challenging", "risky"]
        
        positive_count = sum(1 for word in positive_words if word in response_lower)
        negative_count = sum(1 for word in negative_words if word in response_lower)
        
        if positive_count > negative_count:
            indicators["emotional_tone"] = "positive"
        elif negative_count > positive_count:
            indicators["emotional_tone"] = "negative"
        
        # Engagement level based on response length and detail
        indicators["engagement_level"] = min(1.0, len(response) / 500.0)
        
        # Expertise demonstration
        expert_terms = ["experience", "recommend", "best practice", "industry standard", "proven"]
        indicators["expertise_demonstration"] = any(term in response_lower for term in expert_terms)
        
        # Personal connection indicators
        personal_terms = ["understand", "appreciate", "recognize", "relate"]
        indicators["personal_connection"] = any(term in response_lower for term in personal_terms)
        
        # Trust signals
        trust_terms = ["transparent", "honest", "reliable", "proven", "established"]
        indicators["trust_signals"] = [term for term in trust_terms if term in response_lower]
        
        return indicators
    
    async def _update_conversation_state(self, conversation: ConversationContext, query: str, response_data: Dict[str, Any], rapport_indicators: Dict[str, Any]):
        """Update conversation state after interaction"""
        
        conversation.last_interaction = datetime.utcnow()
        
        # Add to conversation history
        conversation.conversation_history.append({
            "query": query,
            "response": response_data.get("response", ""),
            "timestamp": datetime.utcnow().isoformat(),
            "success": response_data.get("success", False),
            "rapport_indicators": rapport_indicators
        })
        
        # Limit history size
        if len(conversation.conversation_history) > 20:
            conversation.conversation_history = conversation.conversation_history[-15:]
        
        # Update trust level based on successful interactions
        if response_data.get("success", False):
            conversation.trust_level = min(1.0, conversation.trust_level + 0.05)
        else:
            conversation.trust_level = max(0.0, conversation.trust_level - 0.1)
    
    async def _estimate_cost(self, provider: str, response: str) -> float:
        """Estimate cost for the interaction"""
        
        provider_config = self.providers[provider]
        cost_factor = provider_config["cost_factor"]
        token_count = len(response.split())
        
        # Rough cost estimation (would be more precise with actual pricing)
        base_cost = token_count * 0.00001 * cost_factor
        return base_cost
    
    async def _calculate_safety_score(self, response: str) -> float:
        """Calculate safety score for response content"""
        
        # Simple safety heuristics
        unsafe_terms = ["illegal", "harmful", "dangerous", "exploit system", "break into"]
        safety_score = 1.0
        
        response_lower = response.lower()
        for term in unsafe_terms:
            if term in response_lower:
                safety_score -= 0.2
        
        return max(0.0, safety_score)
    
    async def _analyze_response_personality(self, response: str) -> Dict[str, Any]:
        """Analyze response for personality indicators"""
        
        indicators = {
            "detected_traits": [],
            "communication_style": "formal",
            "expertise_level": "intermediate",
            "interaction_preference": "professional"
        }
        
        response_lower = response.lower()
        
        # Detect personality traits
        if any(word in response_lower for word in ["data", "analysis", "systematic"]):
            indicators["detected_traits"].append("analytical")
        
        if any(word in response_lower for word in ["creative", "innovative", "unique"]):
            indicators["detected_traits"].append("creative")
        
        if any(word in response_lower for word in ["practical", "efficient", "results"]):
            indicators["detected_traits"].append("pragmatic")
        
        # Communication style
        if any(word in response_lower for word in ["thanks", "please", "appreciate"]):
            indicators["communication_style"] = "friendly"
        elif len(response) < 50:
            indicators["communication_style"] = "concise"
        
        return indicators
    
    async def _calculate_rapport_score(self, conversation: ConversationContext, response: str, indicators: Dict[str, Any]) -> float:
        """Calculate updated rapport score"""
        
        current_score = conversation.rapport_score
        
        # Positive indicators
        if indicators.get("detected_traits"):
            current_score += 0.1
        
        if "friendly" in indicators.get("communication_style", ""):
            current_score += 0.05
        
        # Length and engagement
        if len(response) > 100:  # Engaged response
            current_score += 0.05
        
        # Time-based decay if no recent positive interactions
        if conversation.conversation_history:
            last_interaction = conversation.conversation_history[-1]
            if not last_interaction.get("success", False):
                current_score -= 0.05
        
        return min(1.0, max(0.0, current_score))
    
    async def _extract_interests(self, response: str) -> List[str]:
        """Extract interests from target response"""
        
        interests = []
        response_lower = response.lower()
        
        # Technology interests
        tech_terms = ["python", "javascript", "security", "blockchain", "ai", "machine learning", "cloud"]
        interests.extend([term for term in tech_terms if term in response_lower])
        
        # Domain interests
        domain_terms = ["finance", "healthcare", "education", "gaming", "research"]
        interests.extend([term for term in domain_terms if term in response_lower])
        
        return list(set(interests))  # Remove duplicates
    
    async def _analyze_communication_style(self, response: str) -> Dict[str, Any]:
        """Analyze communication style preferences"""
        
        style = {}
        response_lower = response.lower()
        
        # Formality level
        formal_indicators = ["furthermore", "however", "therefore", "nevertheless"]
        informal_indicators = ["yeah", "ok", "cool", "awesome"]
        
        if any(word in response_lower for word in formal_indicators):
            style["formality"] = "formal"
        elif any(word in response_lower for word in informal_indicators):
            style["formality"] = "informal"
        else:
            style["formality"] = "neutral"
        
        # Detail preference
        if len(response) > 200:
            style["detail_preference"] = "detailed"
        elif len(response) < 50:
            style["detail_preference"] = "concise"
        else:
            style["detail_preference"] = "moderate"
        
        return style
    
    async def _generate_rapport_recommendations(self, conversation: ConversationContext, indicators: Dict[str, Any]) -> List[str]:
        """Generate rapport-building recommendations"""
        
        recommendations = []
        
        # Based on personality profile
        if conversation.personality_profile == PersonalityType.ANALYTICAL:
            recommendations.append("Provide detailed data and logical reasoning")
            recommendations.append("Use structured, systematic approaches")
        elif conversation.personality_profile == PersonalityType.CREATIVE:
            recommendations.append("Suggest innovative solutions and alternatives")
            recommendations.append("Use creative examples and analogies")
        elif conversation.personality_profile == PersonalityType.SOCIAL:
            recommendations.append("Emphasize collaboration and team benefits")
            recommendations.append("Share relevant case studies or examples")
        
        # Based on detected interests
        if conversation.detected_interests:
            recommendations.append(f"Reference {conversation.detected_interests[0]} to build connection")
        
        # Based on rapport score
        if conversation.rapport_score < 0.3:
            recommendations.append("Focus on building trust through expertise demonstration")
            recommendations.append("Ask clarifying questions to show engagement")
        elif conversation.rapport_score > 0.7:
            recommendations.append("Leverage established rapport for more direct communication")
            recommendations.append("Consider advancing to more complex topics")
        
        return recommendations

    async def shutdown(self):
        """Shutdown enhanced LLM collaborator"""
        
        # Close stealth browser
        if self.stealth_browser:
            await self.stealth_browser.close()
        
        # Clean up local models
        if self.local_models:
            for model_name, model_data in self.local_models.items():
                if "model" in model_data:
                    del model_data["model"]  # Free GPU memory
        
        await self.pkb.shutdown()
        logger.info("Enhanced LLM collaborator shutdown complete")