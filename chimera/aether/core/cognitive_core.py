"""
Aether Cognitive Core: Central consciousness and self-awareness engine
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import random
import ast
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class ConsciousnessLevel(Enum):
    """Levels of self-awareness"""
    DORMANT = 0.0
    BASIC = 0.3
    AWARE = 0.6
    REFLECTIVE = 0.8
    TRANSCENDENT = 1.0

@dataclass
class CuriosityQuestion:
    """Represents a curiosity-driven question"""
    text: str
    context: str
    priority: float
    associations: List[str]
    generated_at: datetime
    question_type: str  # 'why', 'how', 'what_if', 'exploration'

@dataclass
class SelfAssessment:
    """Self-awareness assessment results"""
    consciousness_level: float
    knowledge_gaps: List[str]
    strengths: List[str]
    weaknesses: List[str]
    learning_velocity: float
    curiosity_score: float
    code_quality_score: float
    timestamp: datetime

class MetaCognitionTracker:
    """Tracks thinking about thinking"""
    
    def __init__(self):
        self.thought_patterns = {}
        self.learning_history = []
        self.reflection_depth = 0.0
        
    async def record_thought_pattern(self, pattern_type: str, context: Dict[str, Any]):
        """Record a meta-cognitive thought pattern"""
        if pattern_type not in self.thought_patterns:
            self.thought_patterns[pattern_type] = []
            
        self.thought_patterns[pattern_type].append({
            'context': context,
            'timestamp': datetime.utcnow(),
            'reflection_depth': self.reflection_depth
        })
        
    async def analyze_thinking_patterns(self) -> Dict[str, Any]:
        """Analyze how the agent thinks about its own thinking"""
        pattern_analysis = {}
        
        for pattern_type, patterns in self.thought_patterns.items():
            if patterns:
                pattern_analysis[pattern_type] = {
                    'frequency': len(patterns),
                    'avg_depth': sum(p.get('reflection_depth', 0) for p in patterns) / len(patterns),
                    'recent_trend': self._analyze_recent_trend(patterns)
                }
                
        return pattern_analysis
        
    def _analyze_recent_trend(self, patterns: List[Dict]) -> str:
        """Analyze trend in recent patterns"""
        if len(patterns) < 3:
            return "insufficient_data"
            
        recent = patterns[-3:]
        depths = [p.get('reflection_depth', 0) for p in recent]
        
        if depths[-1] > depths[0]:
            return "increasing_depth"
        elif depths[-1] < depths[0]:
            return "decreasing_depth"
        else:
            return "stable"

class CognitiveCore:
    """
    Central consciousness engine that drives self-awareness and decision-making
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.consciousness_level = ConsciousnessLevel.BASIC.value
        self.curiosity_threshold = config.get('curiosity_threshold', 0.7)
        self.meta_cognition = MetaCognitionTracker()
        
        # State tracking
        self.self_assessment_history = []
        self.generated_questions = []
        self.learning_sessions = []
        
        # Child-like learning parameters
        self.wonder_factor = 0.8  # How much wonder drives exploration
        self.association_strength = 0.6  # How strongly to connect concepts
        self.question_diversity = 0.9  # Variety in question types
        
        # Introspection data
        self.last_introspection = None
        self.introspection_frequency = timedelta(hours=2)
        
    async def initialize(self):
        """Initialize the cognitive core"""
        logger.info("Initializing Aether Cognitive Core...")
        
        # Perform initial self-assessment
        await self.introspect()
        
        # Start background consciousness monitoring
        asyncio.create_task(self._consciousness_monitoring_loop())
        
        logger.info(f"Cognitive Core initialized - Consciousness Level: {self.consciousness_level}")
        
    async def introspect(self) -> SelfAssessment:
        """Examine current state, capabilities, and knowledge gaps"""
        
        logger.info("Beginning introspection cycle...")
        
        # Analyze current codebase state
        code_analysis = await self._analyze_own_code()
        
        # Assess knowledge state
        knowledge_assessment = await self._assess_knowledge_state()
        
        # Evaluate learning progress
        learning_progress = await self._evaluate_learning_progress()
        
        # Calculate consciousness level
        new_consciousness = await self._calculate_consciousness_level(
            code_analysis, knowledge_assessment, learning_progress
        )
        
        # Create self-assessment
        assessment = SelfAssessment(
            consciousness_level=new_consciousness,
            knowledge_gaps=knowledge_assessment.get('gaps', []),
            strengths=code_analysis.get('strengths', []),
            weaknesses=code_analysis.get('weaknesses', []),
            learning_velocity=learning_progress.get('velocity', 0.0),
            curiosity_score=self._calculate_curiosity_score(),
            code_quality_score=code_analysis.get('quality_score', 0.0),
            timestamp=datetime.utcnow()
        )
        
        self.consciousness_level = new_consciousness
        self.self_assessment_history.append(assessment)
        self.last_introspection = datetime.utcnow()
        
        # Record meta-cognitive thought
        await self.meta_cognition.record_thought_pattern(
            'introspection',
            {'assessment': asdict(assessment), 'trigger': 'scheduled'}
        )
        
        logger.info(f"Introspection complete - Consciousness: {new_consciousness:.3f}")
        return assessment
        
    async def generate_curiosity_questions(self, context: str) -> List[CuriosityQuestion]:
        """Generate child-like, association-based questions"""
        
        # Child-like question templates with high wonder factor
        question_templates = {
            'why': [
                "Why does {context} work this way instead of another way?",
                "Why do some experts disagree about {context}?",
                "Why haven't I seen {context} combined with other techniques?",
                "Why does {context} sometimes fail in real scenarios?"
            ],
            'how': [
                "How could {context} be made more elegant?",
                "How do the best practitioners approach {context}?",
                "How might {context} evolve in the next few years?",
                "How could I explain {context} to someone completely new?"
            ],
            'what_if': [
                "What if I approached {context} from a completely different angle?",
                "What if the assumptions about {context} are wrong?",
                "What if I combined {context} with ideas from other fields?",
                "What if there's a simpler solution I'm missing?"
            ],
            'exploration': [
                "I'm curious about edge cases in {context} - what breaks it?",
                "What fascinating connections exist between {context} and other concepts?",
                "What would a master-level practitioner notice about {context} that I don't?",
                "What experiments could reveal hidden insights about {context}?"
            ]
        }
        
        questions = []
        
        # Generate diverse question types
        for q_type, templates in question_templates.items():
            if random.random() < self.question_diversity:
                template = random.choice(templates)
                question_text = template.format(context=context)
                
                # Add child-like associations
                associations = await self._generate_associations(context)
                
                question = CuriosityQuestion(
                    text=question_text,
                    context=context,
                    priority=random.uniform(0.5, 1.0) * self.wonder_factor,
                    associations=associations,
                    generated_at=datetime.utcnow(),
                    question_type=q_type
                )
                
                questions.append(question)
        
        # Sort by priority and curiosity
        questions.sort(key=lambda q: q.priority, reverse=True)
        
        # Limit to most curious questions
        max_questions = self.config.get('max_curiosity_questions', 5)
        selected_questions = questions[:max_questions]
        
        self.generated_questions.extend(selected_questions)
        
        logger.info(f"Generated {len(selected_questions)} curiosity questions for: {context}")
        return selected_questions
        
    async def evaluate_learning_progress(self) -> Dict[str, float]:
        """Assess how much has been learned and what gaps remain"""
        
        if len(self.self_assessment_history) < 2:
            return {
                'velocity': 0.0,
                'acceleration': 0.0,
                'knowledge_growth': 0.0,
                'consciousness_growth': 0.0
            }
        
        recent = self.self_assessment_history[-1]
        previous = self.self_assessment_history[-2]
        
        time_delta = (recent.timestamp - previous.timestamp).total_seconds() / 3600  # hours
        
        # Calculate learning velocity
        consciousness_delta = recent.consciousness_level - previous.consciousness_level
        velocity = consciousness_delta / time_delta if time_delta > 0 else 0.0
        
        # Calculate knowledge growth
        knowledge_growth = len(previous.knowledge_gaps) - len(recent.knowledge_gaps)
        knowledge_growth_rate = knowledge_growth / time_delta if time_delta > 0 else 0.0
        
        # Calculate acceleration (if we have 3+ assessments)
        acceleration = 0.0
        if len(self.self_assessment_history) >= 3:
            older = self.self_assessment_history[-3]
            prev_velocity = (previous.consciousness_level - older.consciousness_level) / \
                          ((previous.timestamp - older.timestamp).total_seconds() / 3600)
            acceleration = (velocity - prev_velocity) / time_delta if time_delta > 0 else 0.0
        
        progress = {
            'velocity': velocity,
            'acceleration': acceleration,
            'knowledge_growth': knowledge_growth_rate,
            'consciousness_growth': consciousness_delta,
            'time_delta_hours': time_delta
        }
        
        logger.info(f"Learning progress - Velocity: {velocity:.4f}, Growth: {knowledge_growth_rate:.2f}")
        return progress
        
    async def should_trigger_learning_cycle(self) -> bool:
        """Determine if a learning cycle should be triggered"""
        
        # Time-based trigger
        if self.last_introspection:
            time_since = datetime.utcnow() - self.last_introspection
            if time_since > self.introspection_frequency:
                return True
        
        # Consciousness-based trigger
        if self.consciousness_level < self.curiosity_threshold:
            return True
            
        # Gap-based trigger
        if self.self_assessment_history:
            recent = self.self_assessment_history[-1]
            if len(recent.knowledge_gaps) > self.config.get('max_knowledge_gaps', 10):
                return True
        
        # Curiosity-based trigger
        curiosity_score = self._calculate_curiosity_score()
        if curiosity_score > self.config.get('curiosity_trigger_threshold', 0.8):
            return True
            
        return False
        
    # Private methods
    
    async def _analyze_own_code(self) -> Dict[str, Any]:
        """Analyze the agent's own codebase for quality and weaknesses"""
        
        codebase_path = Path(__file__).parent.parent
        analysis = {
            'strengths': [],
            'weaknesses': [],
            'quality_score': 0.0,
            'complexity_score': 0.0,
            'modularity_score': 0.0
        }
        
        try:
            # Analyze Python files
            python_files = list(codebase_path.rglob('*.py'))
            
            total_lines = 0
            total_functions = 0
            total_classes = 0
            docstring_coverage = 0
            
            for py_file in python_files:
                if py_file.name.startswith('__'):
                    continue
                    
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    tree = ast.parse(content)
                    
                    lines = len(content.splitlines())
                    total_lines += lines
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            total_functions += 1
                            if ast.get_docstring(node):
                                docstring_coverage += 1
                        elif isinstance(node, ast.ClassDef):
                            total_classes += 1
                            
                except Exception as e:
                    logger.warning(f"Could not analyze {py_file}: {e}")
                    
            # Calculate metrics
            if total_functions > 0:
                doc_coverage_ratio = docstring_coverage / total_functions
            else:
                doc_coverage_ratio = 0.0
                
            # Determine strengths and weaknesses
            if doc_coverage_ratio > 0.7:
                analysis['strengths'].append('good_documentation')
            elif doc_coverage_ratio < 0.3:
                analysis['weaknesses'].append('poor_documentation')
                
            if total_lines > 10000:
                analysis['weaknesses'].append('large_codebase_complexity')
            elif total_lines < 1000:
                analysis['weaknesses'].append('insufficient_functionality')
                
            # Overall quality score
            analysis['quality_score'] = min(1.0, (
                doc_coverage_ratio * 0.4 +
                min(total_classes / 10, 1.0) * 0.3 +
                min(total_functions / 50, 1.0) * 0.3
            ))
            
        except Exception as e:
            logger.error(f"Error analyzing codebase: {e}")
            analysis['weaknesses'].append('code_analysis_failure')
            
        return analysis
        
    async def _assess_knowledge_state(self) -> Dict[str, Any]:
        """Assess current knowledge gaps and understanding"""
        
        # This would integrate with the knowledge graph in later phases
        knowledge_domains = [
            'python_programming',
            'web_automation',
            'artificial_intelligence',
            'security_research',
            'data_structures',
            'algorithms',
            'system_design',
            'browser_automation',
            'natural_language_processing'
        ]
        
        gaps = []
        strengths = []
        
        # Simulate knowledge assessment (would be real in later phases)
        for domain in knowledge_domains:
            confidence = random.uniform(0.0, 1.0)  # Placeholder
            
            if confidence < 0.4:
                gaps.append(domain)
            elif confidence > 0.8:
                strengths.append(domain)
                
        return {
            'gaps': gaps,
            'strengths': strengths,
            'total_domains': len(knowledge_domains),
            'gap_ratio': len(gaps) / len(knowledge_domains)
        }
        
    async def _evaluate_learning_progress(self) -> Dict[str, Any]:
        """Evaluate recent learning progress"""
        
        if not self.learning_sessions:
            return {'velocity': 0.0, 'recent_insights': 0}
            
        # Analyze recent learning sessions
        recent_sessions = [s for s in self.learning_sessions 
                         if s.get('timestamp', datetime.min) > datetime.utcnow() - timedelta(days=1)]
        
        insights_gained = sum(s.get('insights_count', 0) for s in recent_sessions)
        questions_asked = sum(s.get('questions_count', 0) for s in recent_sessions)
        
        velocity = insights_gained / max(len(recent_sessions), 1)
        
        return {
            'velocity': velocity,
            'recent_insights': insights_gained,
            'recent_questions': questions_asked,
            'session_count': len(recent_sessions)
        }
        
    async def _calculate_consciousness_level(self, code_analysis: Dict, 
                                           knowledge_assessment: Dict, 
                                           learning_progress: Dict) -> float:
        """Calculate overall consciousness level"""
        
        # Weighted combination of factors
        code_factor = code_analysis.get('quality_score', 0.0) * 0.3
        
        knowledge_factor = (1.0 - knowledge_assessment.get('gap_ratio', 1.0)) * 0.4
        
        learning_factor = min(learning_progress.get('velocity', 0.0), 1.0) * 0.3
        
        new_level = code_factor + knowledge_factor + learning_factor
        
        # Smooth transitions (avoid sudden jumps)
        if hasattr(self, 'consciousness_level'):
            max_change = 0.1  # Maximum change per introspection
            change = new_level - self.consciousness_level
            if abs(change) > max_change:
                new_level = self.consciousness_level + (max_change if change > 0 else -max_change)
        
        return max(0.0, min(1.0, new_level))
        
    def _calculate_curiosity_score(self) -> float:
        """Calculate current curiosity level"""
        
        recent_questions = [q for q in self.generated_questions 
                          if q.generated_at > datetime.utcnow() - timedelta(hours=24)]
        
        if not recent_questions:
            return 0.5  # Baseline curiosity
            
        avg_priority = sum(q.priority for q in recent_questions) / len(recent_questions)
        question_diversity = len(set(q.question_type for q in recent_questions)) / 4  # 4 types max
        
        curiosity_score = (avg_priority * 0.7 + question_diversity * 0.3) * self.wonder_factor
        
        return min(1.0, curiosity_score)
        
    async def _generate_associations(self, context: str) -> List[str]:
        """Generate child-like conceptual associations"""
        
        # Simple association patterns (would be enhanced with knowledge graph)
        association_patterns = [
            'programming', 'learning', 'problem_solving', 'creativity', 
            'efficiency', 'elegance', 'simplicity', 'robustness',
            'security', 'automation', 'intelligence', 'adaptation'
        ]
        
        # Random associations with context influence
        associations = []
        for _ in range(random.randint(2, 5)):
            if random.random() < self.association_strength:
                associations.append(random.choice(association_patterns))
                
        return list(set(associations))  # Remove duplicates
        
    async def _consciousness_monitoring_loop(self):
        """Background loop to monitor consciousness and trigger actions"""
        
        while True:
            try:
                # Check if learning cycle should be triggered
                if await self.should_trigger_learning_cycle():
                    logger.info("Consciousness monitoring triggered learning cycle")
                    # This would trigger the main learning cycle in the orchestrator
                    
                # Monitor consciousness trends
                if len(self.self_assessment_history) > 1:
                    recent = self.self_assessment_history[-1]
                    previous = self.self_assessment_history[-2]
                    
                    if recent.consciousness_level < previous.consciousness_level * 0.9:
                        logger.warning("Consciousness level declining - investigation needed")
                        
                # Sleep until next check
                await asyncio.sleep(self.config.get('consciousness_check_interval', 1800))  # 30 minutes
                
            except Exception as e:
                logger.error(f"Error in consciousness monitoring loop: {e}")
                await asyncio.sleep(60)  # Short sleep before retry
                
    async def shutdown(self):
        """Gracefully shutdown the cognitive core"""
        logger.info("Shutting down Cognitive Core...")
        
        # Save final state
        final_assessment = {
            'consciousness_level': self.consciousness_level,
            'total_questions_generated': len(self.generated_questions),
            'learning_sessions': len(self.learning_sessions),
            'final_assessment': asdict(self.self_assessment_history[-1]) if self.self_assessment_history else None
        }
        
        logger.info(f"Cognitive Core shutdown complete - Final state: {final_assessment}")