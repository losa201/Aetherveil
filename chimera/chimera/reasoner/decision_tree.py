"""
Decision engine for neuroplastic reasoning
Implements weighted decision trees that adapt based on outcomes
"""

import math
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class DecisionType(Enum):
    """Types of decisions the engine can make"""
    TOOL_SELECTION = "tool_selection"
    APPROACH_STRATEGY = "approach_strategy"
    RISK_ASSESSMENT = "risk_assessment"
    COLLABORATION_DECISION = "collaboration_decision"
    STEALTH_LEVEL = "stealth_level"
    INFORMATION_GATHERING = "information_gathering"
    EXPLOITATION_PATH = "exploitation_path"

@dataclass
class DecisionNode:
    """A node in the decision tree"""
    
    node_id: str
    decision_type: DecisionType
    condition: str
    weight: float
    confidence: float
    children: List['DecisionNode']
    outcomes: List[Dict[str, Any]]  # Historical outcomes
    
    def calculate_success_rate(self) -> float:
        """Calculate success rate based on historical outcomes"""
        if not self.outcomes:
            return 0.5  # Neutral success rate for new nodes
            
        successes = sum(1 for outcome in self.outcomes if outcome.get("success", False))
        return successes / len(self.outcomes)
        
    def update_weight(self, outcome: Dict[str, Any], learning_rate: float):
        """Update node weight based on outcome"""
        success = outcome.get("success", False)
        current_rate = self.calculate_success_rate()
        
        # Add outcome to history
        self.outcomes.append(outcome)
        
        # Limit history size to prevent unbounded growth
        if len(self.outcomes) > 100:
            self.outcomes = self.outcomes[-50:]  # Keep recent half
            
        # Adjust weight based on success
        if success:
            self.weight = min(self.weight + learning_rate * (1 - current_rate), 1.0)
        else:
            self.weight = max(self.weight - learning_rate * current_rate, 0.1)
            
        # Update confidence based on number of trials
        trial_count = len(self.outcomes)
        self.confidence = min(trial_count / 20.0, 1.0)  # Full confidence after 20 trials

@dataclass
class DecisionContext:
    """Context information for making decisions"""
    
    target: str
    current_phase: str
    available_tools: List[str]
    previous_results: List[Dict[str, Any]]
    risk_constraints: Dict[str, Any]
    persona_preferences: Dict[str, Any]
    time_constraints: Optional[int] = None
    stealth_requirements: float = 0.5

class DecisionEngine:
    """
    Neuroplastic decision engine that learns from outcomes
    Uses weighted decision trees that evolve based on success/failure
    """
    
    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.decision_trees: Dict[DecisionType, DecisionNode] = {}
        self.global_outcomes: List[Dict[str, Any]] = []
        self._initialize_default_trees()
        
    def make_decision(self, decision_type: DecisionType, context: DecisionContext) -> Dict[str, Any]:
        """Make a decision based on current context and learned weights"""
        
        if decision_type not in self.decision_trees:
            logger.warning(f"No decision tree for type: {decision_type}")
            return self._make_fallback_decision(decision_type, context)
            
        tree = self.decision_trees[decision_type]
        decision_path = []
        
        # Traverse the decision tree
        current_node = tree
        while current_node:
            decision_path.append(current_node.node_id)
            
            # Evaluate condition and find best child
            best_child = self._evaluate_node(current_node, context)
            
            if not best_child:
                break
                
            current_node = best_child
            
        # Generate final decision
        decision = self._generate_decision(decision_type, context, decision_path, current_node)
        
        logger.debug(f"Decision for {decision_type}: {decision}")
        return decision
        
    def learn_from_outcome(self, decision_type: DecisionType, decision_path: List[str], 
                          outcome: Dict[str, Any]):
        """Learn from the outcome of a decision"""
        
        if decision_type not in self.decision_trees:
            return
            
        # Update global outcomes
        self.global_outcomes.append({
            "decision_type": decision_type.value,
            "decision_path": decision_path,
            "outcome": outcome,
            "timestamp": outcome.get("timestamp")
        })
        
        # Update weights along the decision path
        tree = self.decision_trees[decision_type]
        self._update_path_weights(tree, decision_path, outcome)
        
        # Prune ineffective paths if needed
        if len(self.global_outcomes) % 50 == 0:  # Every 50 outcomes
            self._prune_ineffective_paths()
            
    def get_decision_confidence(self, decision_type: DecisionType, context: DecisionContext) -> float:
        """Get confidence level for a decision type given the context"""
        
        if decision_type not in self.decision_trees:
            return 0.5
            
        tree = self.decision_trees[decision_type]
        
        # Calculate confidence based on node weights and historical success
        total_weight = 0
        weighted_confidence = 0
        
        def traverse_for_confidence(node: DecisionNode, depth: int = 0):
            nonlocal total_weight, weighted_confidence
            
            if depth > 3:  # Limit traversal depth
                return
                
            weight_factor = node.weight * (0.8 ** depth)  # Diminishing returns
            total_weight += weight_factor
            weighted_confidence += weight_factor * node.confidence
            
            for child in node.children:
                traverse_for_confidence(child, depth + 1)
                
        traverse_for_confidence(tree)
        
        if total_weight == 0:
            return 0.5
            
        return weighted_confidence / total_weight
        
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get statistics about learning and decision making"""
        
        stats = {
            "total_decisions": len(self.global_outcomes),
            "decision_types": {},
            "overall_success_rate": 0.0,
            "learning_rate": self.learning_rate
        }
        
        # Calculate per-type statistics
        for decision_type in DecisionType:
            type_outcomes = [o for o in self.global_outcomes 
                           if o["decision_type"] == decision_type.value]
            
            if type_outcomes:
                successes = sum(1 for o in type_outcomes 
                              if o["outcome"].get("success", False))
                success_rate = successes / len(type_outcomes)
                
                stats["decision_types"][decision_type.value] = {
                    "total_decisions": len(type_outcomes),
                    "success_rate": success_rate,
                    "confidence": self.get_decision_confidence(decision_type, None)
                }
                
        # Overall success rate
        if self.global_outcomes:
            total_successes = sum(1 for o in self.global_outcomes 
                                if o["outcome"].get("success", False))
            stats["overall_success_rate"] = total_successes / len(self.global_outcomes)
            
        return stats
        
    def _initialize_default_trees(self):
        """Initialize default decision trees for each decision type"""
        
        # Tool Selection Tree
        tool_tree = DecisionNode(
            node_id="tool_root",
            decision_type=DecisionType.TOOL_SELECTION,
            condition="target_analysis",
            weight=1.0,
            confidence=0.5,
            children=[
                DecisionNode(
                    node_id="passive_recon",
                    decision_type=DecisionType.TOOL_SELECTION,
                    condition="stealth_priority > 0.7",
                    weight=0.8,
                    confidence=0.6,
                    children=[],
                    outcomes=[]
                ),
                DecisionNode(
                    node_id="active_scan",
                    decision_type=DecisionType.TOOL_SELECTION,
                    condition="stealth_priority <= 0.7",
                    weight=0.7,
                    confidence=0.6,
                    children=[],
                    outcomes=[]
                )
            ],
            outcomes=[]
        )
        
        # Risk Assessment Tree
        risk_tree = DecisionNode(
            node_id="risk_root",
            decision_type=DecisionType.RISK_ASSESSMENT,
            condition="risk_evaluation",
            weight=1.0,
            confidence=0.5,
            children=[
                DecisionNode(
                    node_id="low_risk",
                    decision_type=DecisionType.RISK_ASSESSMENT,
                    condition="risk_level < 0.3",
                    weight=0.9,
                    confidence=0.7,
                    children=[],
                    outcomes=[]
                ),
                DecisionNode(
                    node_id="high_risk",
                    decision_type=DecisionType.RISK_ASSESSMENT,
                    condition="risk_level >= 0.7",
                    weight=0.5,
                    confidence=0.6,
                    children=[],
                    outcomes=[]
                )
            ],
            outcomes=[]
        )
        
        # Store trees
        self.decision_trees[DecisionType.TOOL_SELECTION] = tool_tree
        self.decision_trees[DecisionType.RISK_ASSESSMENT] = risk_tree
        
        # Initialize other tree types with basic structures
        for decision_type in DecisionType:
            if decision_type not in self.decision_trees:
                self.decision_trees[decision_type] = self._create_basic_tree(decision_type)
                
    def _create_basic_tree(self, decision_type: DecisionType) -> DecisionNode:
        """Create a basic decision tree for a given type"""
        
        return DecisionNode(
            node_id=f"{decision_type.value}_root",
            decision_type=decision_type,
            condition="default",
            weight=0.7,
            confidence=0.5,
            children=[],
            outcomes=[]
        )
        
    def _evaluate_node(self, node: DecisionNode, context: DecisionContext) -> Optional[DecisionNode]:
        """Evaluate a node's condition and return the best child"""
        
        if not node.children:
            return None
            
        # Simple condition evaluation based on context
        best_child = None
        best_score = -1
        
        for child in node.children:
            score = self._calculate_node_score(child, context)
            if score > best_score:
                best_score = score
                best_child = child
                
        return best_child
        
    def _calculate_node_score(self, node: DecisionNode, context: DecisionContext) -> float:
        """Calculate a score for a node based on context and learned weights"""
        
        base_score = node.weight * node.confidence
        
        # Adjust based on persona preferences
        persona_prefs = context.persona_preferences
        
        if "stealth" in node.node_id and persona_prefs.get("stealth_priority", 0.5) > 0.7:
            base_score *= 1.2
        elif "aggressive" in node.node_id and persona_prefs.get("risk_tolerance", 0.5) > 0.7:
            base_score *= 1.2
        elif "passive" in node.node_id and persona_prefs.get("stealth_priority", 0.5) > 0.8:
            base_score *= 1.3
            
        # Adjust based on success rate
        success_rate = node.calculate_success_rate()
        base_score *= (0.5 + success_rate * 0.5)  # Scale between 0.5 and 1.0
        
        return base_score
        
    def _generate_decision(self, decision_type: DecisionType, context: DecisionContext, 
                          decision_path: List[str], final_node: Optional[DecisionNode]) -> Dict[str, Any]:
        """Generate the final decision based on the traversed path"""
        
        decision = {
            "type": decision_type.value,
            "path": decision_path,
            "confidence": final_node.confidence if final_node else 0.5,
            "reasoning": []
        }
        
        # Generate type-specific decisions
        if decision_type == DecisionType.TOOL_SELECTION:
            decision.update(self._generate_tool_decision(context, final_node))
        elif decision_type == DecisionType.RISK_ASSESSMENT:
            decision.update(self._generate_risk_decision(context, final_node))
        elif decision_type == DecisionType.STEALTH_LEVEL:
            decision.update(self._generate_stealth_decision(context, final_node))
        else:
            decision["action"] = "proceed_with_caution"
            decision["parameters"] = {}
            
        return decision
        
    def _generate_tool_decision(self, context: DecisionContext, node: Optional[DecisionNode]) -> Dict[str, Any]:
        """Generate tool selection decision"""
        
        available_tools = context.available_tools
        persona_prefs = context.persona_preferences
        
        # Filter tools based on persona preferences
        preferred_tools = persona_prefs.get("preferred_tools", [])
        avoided_tools = persona_prefs.get("avoided_tools", [])
        
        filtered_tools = []
        for tool in available_tools:
            if any(avoid in tool.lower() for avoid in avoided_tools):
                continue
            filtered_tools.append(tool)
            
        # Prioritize preferred tools
        prioritized_tools = []
        for pref in preferred_tools:
            for tool in filtered_tools:
                if pref in tool.lower() and tool not in prioritized_tools:
                    prioritized_tools.append(tool)
                    
        # Add remaining tools
        for tool in filtered_tools:
            if tool not in prioritized_tools:
                prioritized_tools.append(tool)
                
        return {
            "action": "select_tools",
            "selected_tools": prioritized_tools[:3],  # Top 3 tools
            "reasoning": [f"Based on persona preferences and node: {node.node_id if node else 'default'}"]
        }
        
    def _generate_risk_decision(self, context: DecisionContext, node: Optional[DecisionNode]) -> Dict[str, Any]:
        """Generate risk assessment decision"""
        
        risk_level = "medium"
        proceed = True
        
        if node and "low_risk" in node.node_id:
            risk_level = "low"
            proceed = True
        elif node and "high_risk" in node.node_id:
            risk_level = "high"
            proceed = context.persona_preferences.get("risk_tolerance", 0.5) > 0.6
            
        return {
            "action": "assess_risk",
            "risk_level": risk_level,
            "proceed": proceed,
            "recommended_precautions": self._get_risk_precautions(risk_level)
        }
        
    def _generate_stealth_decision(self, context: DecisionContext, node: Optional[DecisionNode]) -> Dict[str, Any]:
        """Generate stealth level decision"""
        
        stealth_level = context.persona_preferences.get("stealth_priority", 0.5)
        
        return {
            "action": "set_stealth_level",
            "stealth_level": stealth_level,
            "techniques": self._get_stealth_techniques(stealth_level)
        }
        
    def _get_risk_precautions(self, risk_level: str) -> List[str]:
        """Get recommended precautions for a given risk level"""
        
        precautions = {
            "low": ["monitor_responses", "log_activities"],
            "medium": ["rate_limit_requests", "use_proxy", "monitor_responses"],
            "high": ["maximum_stealth", "minimal_footprint", "abort_on_detection"]
        }
        
        return precautions.get(risk_level, [])
        
    def _get_stealth_techniques(self, stealth_level: float) -> List[str]:
        """Get stealth techniques based on level"""
        
        techniques = []
        
        if stealth_level > 0.3:
            techniques.extend(["user_agent_rotation", "request_timing"])
        if stealth_level > 0.5:
            techniques.extend(["proxy_rotation", "header_randomization"])
        if stealth_level > 0.7:
            techniques.extend(["traffic_blending", "decoy_requests"])
        if stealth_level > 0.9:
            techniques.extend(["minimal_fingerprint", "behavioral_mimicry"])
            
        return techniques
        
    def _update_path_weights(self, tree: DecisionNode, path: List[str], outcome: Dict[str, Any]):
        """Update weights for nodes along a decision path"""
        
        def update_node(node: DecisionNode, remaining_path: List[str]):
            if not remaining_path:
                return
                
            if node.node_id == remaining_path[0]:
                node.update_weight(outcome, self.learning_rate)
                
                if len(remaining_path) > 1:
                    for child in node.children:
                        update_node(child, remaining_path[1:])
                        
        update_node(tree, path)
        
    def _prune_ineffective_paths(self):
        """Remove decision paths that consistently perform poorly"""
        
        for tree in self.decision_trees.values():
            self._prune_tree_nodes(tree)
            
    def _prune_tree_nodes(self, node: DecisionNode, min_weight: float = 0.2):
        """Recursively prune nodes with consistently poor performance"""
        
        children_to_remove = []
        
        for child in node.children:
            # Prune child's children first
            self._prune_tree_nodes(child, min_weight)
            
            # Remove child if it performs poorly and has enough trials
            if (child.weight < min_weight and 
                len(child.outcomes) > 10 and 
                child.calculate_success_rate() < 0.3):
                children_to_remove.append(child)
                
        # Remove ineffective children
        for child in children_to_remove:
            node.children.remove(child)
            logger.debug(f"Pruned ineffective node: {child.node_id}")
            
    def _make_fallback_decision(self, decision_type: DecisionType, context: DecisionContext) -> Dict[str, Any]:
        """Make a fallback decision when no tree exists"""
        
        return {
            "type": decision_type.value,
            "action": "proceed_with_default",
            "confidence": 0.3,
            "reasoning": ["Using fallback decision - no learned tree available"]
        }