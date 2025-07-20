"""
Advanced Prompt Knowledge Base (PKB) System
Learns and optimizes prompting strategies for maximum LLM effectiveness
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import statistics

from ..core.events import EventSystem, EventType, EventEmitter
from ..utils.crypto import CryptoUtils

logger = logging.getLogger(__name__)

@dataclass
class PromptTemplate:
    """Optimized prompt template with performance metrics"""
    
    id: str
    category: str  # reconnaissance, exploitation, analysis, etc.
    template: str
    variables: List[str]  # Template variables like {target}, {technique}
    success_rate: float
    avg_response_quality: float
    avg_response_time: float
    usage_count: int
    last_used: datetime
    created_at: datetime
    provider_performance: Dict[str, float]  # Performance per LLM provider
    context_effectiveness: Dict[str, float]  # Effectiveness by context type
    
    def calculate_score(self, provider: str = None, context: str = None) -> float:
        """Calculate overall effectiveness score"""
        base_score = (
            self.success_rate * 0.4 +
            self.avg_response_quality * 0.3 +
            (1.0 - min(self.avg_response_time / 10.0, 1.0)) * 0.2 +  # Faster is better
            min(self.usage_count / 100.0, 1.0) * 0.1  # Experience bonus
        )
        
        # Provider-specific adjustment
        if provider and provider in self.provider_performance:
            base_score *= (0.5 + self.provider_performance[provider] * 0.5)
            
        # Context-specific adjustment
        if context and context in self.context_effectiveness:
            base_score *= (0.5 + self.context_effectiveness[context] * 0.5)
            
        return min(base_score, 1.0)

@dataclass 
class PromptOutcome:
    """Record of a prompt execution and its outcome"""
    
    prompt_id: str
    provider: str
    context: str
    variables_used: Dict[str, str]
    response_time: float
    response_quality: float  # 0.0 to 1.0
    success: bool
    response_length: int
    timestamp: datetime
    error_message: Optional[str] = None

class PromptKnowledgeBase(EventEmitter):
    """
    Advanced PKB that learns optimal prompting strategies
    
    Features:
    - Template optimization based on outcomes
    - Provider-specific performance tracking
    - Context-aware prompt selection
    - Automatic template generation and mutation
    - A/B testing of prompt variations
    """
    
    def __init__(self, config, event_system: EventSystem):
        super().__init__(event_system, "PromptKnowledgeBase")
        
        self.config = config
        self.templates: Dict[str, PromptTemplate] = {}
        self.outcomes: List[PromptOutcome] = []
        self.provider_profiles: Dict[str, Dict[str, Any]] = {}
        
        # Learning parameters
        self.min_samples_for_reliability = 10
        self.template_mutation_rate = 0.1
        self.exploration_rate = 0.2  # Rate of trying new/untested templates
        
        # Performance tracking
        self.category_performance: Dict[str, List[float]] = defaultdict(list)
        self.provider_response_times: Dict[str, List[float]] = defaultdict(list)
        
    async def initialize(self):
        """Initialize PKB with base templates"""
        await self._load_base_templates()
        await self._load_provider_profiles()
        
        # Start background optimization
        asyncio.create_task(self._optimization_loop())
        
        await self.emit_event(
            EventType.KNOWLEDGE_LEARNED,
            {"message": "PKB initialized", "templates": len(self.templates)}
        )
        
        logger.info(f"PKB initialized with {len(self.templates)} templates")
        
    async def get_optimal_prompt(self, category: str, context: str, 
                                provider: str, variables: Dict[str, str]) -> Tuple[str, str]:
        """
        Get the optimal prompt for given parameters
        
        Returns:
            (prompt_text, template_id)
        """
        
        # Get candidate templates for category
        candidates = [t for t in self.templates.values() if t.category == category]
        
        if not candidates:
            # Generate new template if none exist
            template = await self._generate_template(category, context, variables)
            candidates = [template]
            
        # Score candidates based on context and provider
        scored_candidates = []
        for template in candidates:
            score = template.calculate_score(provider, context)
            scored_candidates.append((score, template))
            
        # Sort by score (best first)
        scored_candidates.sort(reverse=True)
        
        # Epsilon-greedy selection for exploration
        if len(scored_candidates) > 1 and self._should_explore():
            # Sometimes choose a less optimal template for learning
            selected_template = scored_candidates[1][1]
        else:
            selected_template = scored_candidates[0][1]
            
        # Fill template with variables
        prompt_text = self._fill_template(selected_template.template, variables)
        
        return prompt_text, selected_template.id
        
    async def record_outcome(self, template_id: str, provider: str, context: str,
                           variables: Dict[str, str], response_time: float,
                           response_quality: float, success: bool, 
                           response_length: int, error_message: str = None):
        """Record the outcome of using a prompt template"""
        
        outcome = PromptOutcome(
            prompt_id=template_id,
            provider=provider,
            context=context,
            variables_used=variables,
            response_time=response_time,
            response_quality=response_quality,
            success=success,
            response_length=response_length,
            timestamp=datetime.utcnow(),
            error_message=error_message
        )
        
        self.outcomes.append(outcome)
        
        # Update template performance
        if template_id in self.templates:
            await self._update_template_performance(template_id, outcome)
            
        # Update provider profile
        await self._update_provider_profile(provider, outcome)
        
        # Trigger learning if we have enough data
        if len(self.outcomes) % 20 == 0:  # Every 20 outcomes
            await self._learn_from_outcomes()
            
    async def analyze_prompt_effectiveness(self, category: str = None) -> Dict[str, Any]:
        """Analyze prompt effectiveness across categories and providers"""
        
        recent_outcomes = [o for o in self.outcomes 
                          if o.timestamp > datetime.utcnow() - timedelta(days=7)]
        
        if not recent_outcomes:
            return {"message": "No recent outcomes to analyze"}
            
        analysis = {
            "total_prompts": len(recent_outcomes),
            "avg_success_rate": statistics.mean([o.success for o in recent_outcomes]),
            "avg_response_time": statistics.mean([o.response_time for o in recent_outcomes]),
            "avg_quality": statistics.mean([o.response_quality for o in recent_outcomes])
        }
        
        # Per-category analysis
        if category:
            category_outcomes = [o for o in recent_outcomes 
                               if self.templates.get(o.prompt_id, {}).category == category]
            if category_outcomes:
                analysis[f"{category}_performance"] = {
                    "success_rate": statistics.mean([o.success for o in category_outcomes]),
                    "avg_quality": statistics.mean([o.response_quality for o in category_outcomes]),
                    "sample_size": len(category_outcomes)
                }
                
        # Provider comparison
        provider_stats = defaultdict(list)
        for outcome in recent_outcomes:
            provider_stats[outcome.provider].append(outcome)
            
        analysis["provider_comparison"] = {}
        for provider, outcomes in provider_stats.items():
            analysis["provider_comparison"][provider] = {
                "success_rate": statistics.mean([o.success for o in outcomes]),
                "avg_quality": statistics.mean([o.response_quality for o in outcomes]),
                "avg_response_time": statistics.mean([o.response_time for o in outcomes]),
                "sample_size": len(outcomes)
            }
            
        return analysis
        
    async def suggest_prompt_improvements(self, template_id: str) -> List[str]:
        """Suggest improvements for a specific template"""
        
        if template_id not in self.templates:
            return ["Template not found"]
            
        template = self.templates[template_id]
        suggestions = []
        
        # Analyze recent failures
        recent_failures = [o for o in self.outcomes 
                          if o.prompt_id == template_id and not o.success
                          and o.timestamp > datetime.utcnow() - timedelta(days=3)]
        
        if len(recent_failures) > 3:
            suggestions.append("High failure rate - consider revising template structure")
            
        # Check response times
        recent_times = [o.response_time for o in self.outcomes 
                       if o.prompt_id == template_id
                       and o.timestamp > datetime.utcnow() - timedelta(days=7)]
        
        if recent_times and statistics.mean(recent_times) > 8.0:
            suggestions.append("Slow response times - consider shortening prompt")
            
        # Check quality scores
        recent_quality = [o.response_quality for o in self.outcomes 
                         if o.prompt_id == template_id
                         and o.timestamp > datetime.utcnow() - timedelta(days=7)]
        
        if recent_quality and statistics.mean(recent_quality) < 0.6:
            suggestions.append("Low quality responses - add more specific instructions")
            
        # Provider-specific suggestions
        provider_performance = template.provider_performance
        if provider_performance:
            best_provider = max(provider_performance.items(), key=lambda x: x[1])
            worst_provider = min(provider_performance.items(), key=lambda x: x[1])
            
            if best_provider[1] - worst_provider[1] > 0.3:
                suggestions.append(f"Large provider variance - optimize for {worst_provider[0]}")
                
        return suggestions if suggestions else ["Template performing well - no improvements needed"]
        
    # Private methods
    
    async def _load_base_templates(self):
        """Load base prompt templates for different categories"""
        
        base_templates = {
            "reconnaissance": [
                {
                    "template": "As a cybersecurity expert, analyze the target {target} and provide reconnaissance insights focusing on {focus_area}. Include potential attack vectors and security considerations. Be thorough but ethical.",
                    "variables": ["target", "focus_area"]
                },
                {
                    "template": "Perform OSINT analysis on {target}. What publicly available information can help understand the attack surface? Focus on {reconnaissance_type} and provide actionable intelligence.",
                    "variables": ["target", "reconnaissance_type"]
                }
            ],
            "vulnerability_analysis": [
                {
                    "template": "Analyze the following technology stack for potential vulnerabilities: {tech_stack}. Consider {vulnerability_types} and provide specific testing recommendations.",
                    "variables": ["tech_stack", "vulnerability_types"]
                },
                {
                    "template": "Given this service information: {service_info}, identify potential security weaknesses and suggest targeted testing approaches. Focus on {testing_methodology}.",
                    "variables": ["service_info", "testing_methodology"]
                }
            ],
            "exploitation_planning": [
                {
                    "template": "Design an ethical exploitation strategy for the vulnerability: {vulnerability_description}. The target environment is {environment_type}. Provide step-by-step approach ensuring minimal impact.",
                    "variables": ["vulnerability_description", "environment_type"]
                }
            ],
            "tool_selection": [
                {
                    "template": "Recommend the best security testing tools for {testing_scenario}. Consider the target type: {target_type} and required stealth level: {stealth_level}. Explain tool selection rationale.",
                    "variables": ["testing_scenario", "target_type", "stealth_level"]
                }
            ],
            "report_analysis": [
                {
                    "template": "Analyze these security findings: {findings}. Prioritize by risk level and provide remediation recommendations for {target_audience}.",
                    "variables": ["findings", "target_audience"]
                }
            ]
        }
        
        for category, templates in base_templates.items():
            for i, template_data in enumerate(templates):
                template_id = f"{category}_{i+1}"
                
                template = PromptTemplate(
                    id=template_id,
                    category=category,
                    template=template_data["template"],
                    variables=template_data["variables"],
                    success_rate=0.5,  # Neutral starting point
                    avg_response_quality=0.5,
                    avg_response_time=5.0,
                    usage_count=0,
                    last_used=datetime.utcnow(),
                    created_at=datetime.utcnow(),
                    provider_performance={},
                    context_effectiveness={}
                )
                
                self.templates[template_id] = template
                
    async def _load_provider_profiles(self):
        """Load and initialize LLM provider profiles"""
        
        # Base provider characteristics
        self.provider_profiles = {
            "claude": {
                "strengths": ["analysis", "reasoning", "technical_accuracy"],
                "weaknesses": ["code_generation"],
                "avg_response_time": 3.5,
                "context_window": 100000,
                "preferred_prompt_style": "detailed_analytical"
            },
            "gpt4": {
                "strengths": ["creativity", "code_generation", "versatility"],
                "weaknesses": ["factual_accuracy"],
                "avg_response_time": 4.2,
                "context_window": 8000,
                "preferred_prompt_style": "conversational_detailed"
            },
            "gemini": {
                "strengths": ["research", "summarization", "multimodal"],
                "weaknesses": ["technical_depth"],
                "avg_response_time": 2.8,
                "context_window": 32000,
                "preferred_prompt_style": "structured_clear"
            }
        }
        
    def _should_explore(self) -> bool:
        """Decide whether to explore (epsilon-greedy)"""
        import random
        return random.random() < self.exploration_rate
        
    def _fill_template(self, template: str, variables: Dict[str, str]) -> str:
        """Fill template with provided variables"""
        
        filled_template = template
        for var, value in variables.items():
            placeholder = "{" + var + "}"
            filled_template = filled_template.replace(placeholder, value)
            
        return filled_template
        
    async def _generate_template(self, category: str, context: str, 
                                variables: Dict[str, str]) -> PromptTemplate:
        """Generate new template based on context and variables"""
        
        # Simple template generation based on category and context
        if category == "reconnaissance":
            template_text = f"Analyze {{target}} for security reconnaissance. Focus on {{focus_area}} in the context of {context}. Provide detailed but ethical analysis."
            template_vars = ["target", "focus_area"]
        elif category == "vulnerability_analysis":
            template_text = f"Examine {{tech_info}} for potential vulnerabilities. Consider {{vuln_types}} in {context} environment."
            template_vars = ["tech_info", "vuln_types"]
        else:
            template_text = f"Provide cybersecurity analysis for {{target}} focusing on {{analysis_type}} in {context} context."
            template_vars = ["target", "analysis_type"]
            
        template_id = f"{category}_generated_{len(self.templates)}"
        
        template = PromptTemplate(
            id=template_id,
            category=category,
            template=template_text,
            variables=template_vars,
            success_rate=0.3,  # Lower initial confidence for generated templates
            avg_response_quality=0.3,
            avg_response_time=5.0,
            usage_count=0,
            last_used=datetime.utcnow(),
            created_at=datetime.utcnow(),
            provider_performance={},
            context_effectiveness={}
        )
        
        self.templates[template_id] = template
        
        await self.emit_event(
            EventType.KNOWLEDGE_LEARNED,
            {"action": "template_generated", "category": category, "template_id": template_id}
        )
        
        return template
        
    async def _update_template_performance(self, template_id: str, outcome: PromptOutcome):
        """Update template performance metrics based on outcome"""
        
        template = self.templates[template_id]
        
        # Update overall metrics with exponential moving average
        alpha = 0.1  # Learning rate
        
        template.success_rate = (1 - alpha) * template.success_rate + alpha * (1.0 if outcome.success else 0.0)
        template.avg_response_quality = (1 - alpha) * template.avg_response_quality + alpha * outcome.response_quality
        template.avg_response_time = (1 - alpha) * template.avg_response_time + alpha * outcome.response_time
        template.usage_count += 1
        template.last_used = outcome.timestamp
        
        # Update provider-specific performance
        if outcome.provider not in template.provider_performance:
            template.provider_performance[outcome.provider] = 0.5
            
        template.provider_performance[outcome.provider] = (
            (1 - alpha) * template.provider_performance[outcome.provider] + 
            alpha * outcome.response_quality
        )
        
        # Update context effectiveness
        if outcome.context not in template.context_effectiveness:
            template.context_effectiveness[outcome.context] = 0.5
            
        template.context_effectiveness[outcome.context] = (
            (1 - alpha) * template.context_effectiveness[outcome.context] + 
            alpha * outcome.response_quality
        )
        
    async def _update_provider_profile(self, provider: str, outcome: PromptOutcome):
        """Update provider profile based on outcome"""
        
        if provider in self.provider_profiles:
            profile = self.provider_profiles[provider]
            
            # Update average response time
            alpha = 0.1
            profile["avg_response_time"] = (
                (1 - alpha) * profile["avg_response_time"] + 
                alpha * outcome.response_time
            )
            
        # Track response times for analysis
        self.provider_response_times[provider].append(outcome.response_time)
        
        # Limit history to last 100 responses
        if len(self.provider_response_times[provider]) > 100:
            self.provider_response_times[provider] = self.provider_response_times[provider][-50:]
            
    async def _learn_from_outcomes(self):
        """Learn patterns from recent outcomes"""
        
        recent_outcomes = [o for o in self.outcomes 
                          if o.timestamp > datetime.utcnow() - timedelta(hours=24)]
        
        if len(recent_outcomes) < 10:
            return
            
        # Identify top-performing templates
        template_performance = defaultdict(list)
        for outcome in recent_outcomes:
            template_performance[outcome.prompt_id].append(outcome.response_quality)
            
        # Generate mutations of successful templates
        for template_id, qualities in template_performance.items():
            if len(qualities) >= 5 and statistics.mean(qualities) > 0.8:
                await self._mutate_template(template_id)
                
        # Prune poor-performing templates
        await self._prune_templates()
        
    async def _mutate_template(self, template_id: str):
        """Create variations of successful templates"""
        
        if template_id not in self.templates:
            return
            
        base_template = self.templates[template_id]
        
        # Simple mutations: add clarifying phrases, change structure
        mutations = [
            "Be specific and detailed in your analysis.",
            "Focus on practical, actionable insights.",
            "Consider both technical and business impact.",
            "Provide step-by-step recommendations."
        ]
        
        import random
        if random.random() < self.template_mutation_rate:
            mutation = random.choice(mutations)
            new_template_text = base_template.template + " " + mutation
            
            new_template_id = f"{base_template.id}_mutation_{len(self.templates)}"
            
            mutated_template = PromptTemplate(
                id=new_template_id,
                category=base_template.category,
                template=new_template_text,
                variables=base_template.variables.copy(),
                success_rate=base_template.success_rate * 0.9,  # Slightly lower initial confidence
                avg_response_quality=base_template.avg_response_quality * 0.9,
                avg_response_time=base_template.avg_response_time,
                usage_count=0,
                last_used=datetime.utcnow(),
                created_at=datetime.utcnow(),
                provider_performance={},
                context_effectiveness={}
            )
            
            self.templates[new_template_id] = mutated_template
            
            await self.emit_event(
                EventType.KNOWLEDGE_LEARNED,
                {"action": "template_mutated", "base_template": template_id, "new_template": new_template_id}
            )
            
    async def _prune_templates(self):
        """Remove poorly performing templates"""
        
        templates_to_remove = []
        
        for template_id, template in self.templates.items():
            # Only prune templates with enough usage data
            if template.usage_count > 20:
                # Remove if consistently poor performance
                if template.success_rate < 0.3 and template.avg_response_quality < 0.4:
                    templates_to_remove.append(template_id)
                    
        # Keep at least 2 templates per category
        category_counts = defaultdict(int)
        for template in self.templates.values():
            category_counts[template.category] += 1
            
        final_removals = []
        for template_id in templates_to_remove:
            template = self.templates[template_id]
            if category_counts[template.category] > 2:
                final_removals.append(template_id)
                category_counts[template.category] -= 1
                
        # Remove templates
        for template_id in final_removals:
            del self.templates[template_id]
            
        if final_removals:
            await self.emit_event(
                EventType.KNOWLEDGE_PRUNED,
                {"action": "templates_pruned", "count": len(final_removals)}
            )
            
    async def _optimization_loop(self):
        """Background optimization of prompt templates"""
        
        while True:
            try:
                await asyncio.sleep(1800)  # Every 30 minutes
                
                # Analyze recent performance
                await self._learn_from_outcomes()
                
                # Update exploration rate based on performance
                if len(self.outcomes) > 100:
                    recent_success = statistics.mean([
                        o.success for o in self.outcomes[-50:]
                    ])
                    
                    # Reduce exploration as we get better results
                    if recent_success > 0.8:
                        self.exploration_rate = max(0.05, self.exploration_rate * 0.95)
                    elif recent_success < 0.6:
                        self.exploration_rate = min(0.3, self.exploration_rate * 1.05)
                        
            except Exception as e:
                logger.error(f"Error in PKB optimization loop: {e}")
                await asyncio.sleep(300)
                
    async def shutdown(self):
        """Shutdown PKB and save state"""
        logger.info("PKB shutdown complete")