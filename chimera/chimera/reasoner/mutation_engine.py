"""
Neuroplastic mutation and exploration engine
Implements genetic algorithm-inspired techniques for evolving tactics and strategies
"""

import asyncio
import logging
import random
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import copy

from ..core.events import EventSystem, EventType, EventEmitter
from ..memory.knowledge_graph_lite import LiteKnowledgeGraph

logger = logging.getLogger(__name__)

@dataclass
class TacticGene:
    """
    A 'gene' representing a specific tactic or approach
    Can be mutated, combined, and selected based on fitness
    """
    
    id: str
    category: str  # reconnaissance, exploitation, analysis, etc.
    approach: str  # specific technique or method
    parameters: Dict[str, Any]  # configurable parameters
    fitness_score: float  # 0.0 to 1.0 based on success rate
    usage_count: int
    success_count: int
    last_success: Optional[datetime]
    creation_method: str  # original, mutated, crossover
    parent_genes: List[str]  # IDs of parent genes if created through combination
    age: int  # generations since creation
    
    def calculate_fitness(self) -> float:
        """Calculate fitness based on success rate and other factors"""
        if self.usage_count == 0:
            return 0.5  # Neutral fitness for untested genes
            
        success_rate = self.success_count / self.usage_count
        
        # Age penalty (older genes lose fitness if not recently successful)
        age_factor = max(0.5, 1.0 - (self.age * 0.01))
        
        # Recency bonus
        recency_factor = 1.0
        if self.last_success:
            days_since_success = (datetime.utcnow() - self.last_success).days
            recency_factor = max(0.7, 1.0 - (days_since_success * 0.02))
            
        # Usage diversity bonus (genes used frequently but not overused)
        usage_factor = min(1.0, self.usage_count / 10.0) * max(0.8, 1.0 - (self.usage_count / 100.0))
        
        fitness = success_rate * age_factor * recency_factor * usage_factor
        self.fitness_score = min(fitness, 1.0)
        
        return self.fitness_score

@dataclass
class ExplorationTask:
    """A task for exploring new approaches or validating hypotheses"""
    
    id: str
    task_type: str  # mutation_test, novel_approach, cross_domain_transfer
    description: str
    target_category: str
    hypothesis: str
    test_parameters: Dict[str, Any]
    expected_outcome: str
    priority: float
    created_at: datetime
    attempts: int = 0
    max_attempts: int = 3

class MutationEngine(EventEmitter):
    """
    Neuroplastic mutation and exploration engine
    
    Implements genetic algorithm concepts for evolving red-team tactics:
    - Mutation of successful approaches
    - Crossover between different tactics
    - Natural selection based on success rates
    - Exploration of novel approaches
    - Periodic 'speciation' events
    """
    
    def __init__(self, config, event_system: EventSystem, knowledge_graph: LiteKnowledgeGraph):
        super().__init__(event_system, "MutationEngine")
        
        self.config = config
        self.knowledge_graph = knowledge_graph
        
        # Genetic algorithm parameters
        self.population_size = config.get("mutation.population_size", 50)
        self.mutation_rate = config.get("mutation.mutation_rate", 0.15)
        self.crossover_rate = config.get("mutation.crossover_rate", 0.3)
        self.selection_pressure = config.get("mutation.selection_pressure", 0.7)
        self.exploration_budget = config.get("mutation.exploration_budget", 0.2)  # 20% of resources
        
        # Gene pool and exploration
        self.gene_pool: Dict[str, TacticGene] = {}
        self.exploration_queue: List[ExplorationTask] = []
        self.generation_count = 0
        
        # Performance tracking
        self.generation_stats: List[Dict[str, Any]] = []
        self.innovation_log: List[Dict[str, Any]] = []
        
    async def initialize(self):
        """Initialize mutation engine with seed population"""
        
        await self._create_seed_population()
        
        # Start background evolution
        asyncio.create_task(self._evolution_loop())
        asyncio.create_task(self._exploration_loop())
        
        await self.emit_event(
            EventType.KNOWLEDGE_LEARNED,
            {"message": "Mutation engine initialized", "population_size": len(self.gene_pool)}
        )
        
        logger.info(f"Mutation engine initialized with {len(self.gene_pool)} genes")
        
    async def evolve_tactic(self, category: str, current_performance: float, 
                           context: Dict[str, Any]) -> Optional[TacticGene]:
        """
        Evolve a tactic for improved performance
        
        Args:
            category: Tactic category (recon, exploitation, etc.)
            current_performance: Current performance score (0.0-1.0)
            context: Situational context for evolution
            
        Returns:
            Evolved tactic gene or None if no improvement found
        """
        
        # Get candidate genes from category
        candidates = [gene for gene in self.gene_pool.values() 
                     if gene.category == category and gene.fitness_score > current_performance]
        
        if not candidates:
            # No better existing genes, try to create new ones
            return await self._generate_novel_approach(category, context, current_performance)
            
        # Select best candidates for mutation/crossover
        selected_candidates = self._tournament_selection(candidates, 3)
        
        # Decide on evolution strategy
        if len(selected_candidates) >= 2 and random.random() < self.crossover_rate:
            # Crossover between two good genes
            return await self._crossover_genes(selected_candidates[0], selected_candidates[1], context)
        else:
            # Mutate the best gene
            return await self._mutate_gene(selected_candidates[0], context)
            
    async def record_tactic_outcome(self, gene_id: str, success: bool, 
                                   performance_score: float, context: Dict[str, Any]):
        """Record the outcome of using a specific tactic gene"""
        
        if gene_id not in self.gene_pool:
            logger.warning(f"Unknown gene ID: {gene_id}")
            return
            
        gene = self.gene_pool[gene_id]
        gene.usage_count += 1
        
        if success:
            gene.success_count += 1
            gene.last_success = datetime.utcnow()
            
        # Update fitness
        gene.calculate_fitness()
        
        # Learn from this outcome
        await self._learn_from_outcome(gene, success, performance_score, context)
        
        await self.emit_event(
            EventType.KNOWLEDGE_UPDATED,
            {"gene_id": gene_id, "success": success, "fitness": gene.fitness_score}
        )
        
    async def suggest_exploration_targets(self, current_context: Dict[str, Any]) -> List[ExplorationTask]:
        """Suggest high-value exploration targets"""
        
        suggestions = []
        
        # Identify underperforming categories
        category_performance = self._analyze_category_performance()
        
        for category, avg_performance in category_performance.items():
            if avg_performance < 0.6:  # Below satisfactory threshold
                
                # Suggest mutation experiments
                exploration_task = ExplorationTask(
                    id=f"mutation_experiment_{category}_{len(self.exploration_queue)}",
                    task_type="mutation_test",
                    description=f"Explore mutations in {category} to improve performance",
                    target_category=category,
                    hypothesis=f"Mutated approaches in {category} can exceed {avg_performance:.2f} performance",
                    test_parameters={"category": category, "performance_target": avg_performance + 0.2},
                    expected_outcome=f"20% performance improvement in {category}",
                    priority=1.0 - avg_performance,  # Lower performance = higher priority
                    created_at=datetime.utcnow()
                )
                
                suggestions.append(exploration_task)
                
        # Suggest cross-domain knowledge transfer
        successful_categories = [cat for cat, perf in category_performance.items() if perf > 0.8]
        struggling_categories = [cat for cat, perf in category_performance.items() if perf < 0.5]
        
        for source_cat in successful_categories:
            for target_cat in struggling_categories:
                transfer_task = ExplorationTask(
                    id=f"knowledge_transfer_{source_cat}_to_{target_cat}",
                    task_type="cross_domain_transfer",
                    description=f"Transfer successful patterns from {source_cat} to {target_cat}",
                    target_category=target_cat,
                    hypothesis=f"Successful {source_cat} patterns can improve {target_cat} performance",
                    test_parameters={"source_category": source_cat, "target_category": target_cat},
                    expected_outcome=f"Performance improvement in {target_cat}",
                    priority=0.8,
                    created_at=datetime.utcnow()
                )
                
                suggestions.append(transfer_task)
                
        # Sort by priority
        suggestions.sort(key=lambda x: x.priority, reverse=True)
        
        return suggestions[:5]  # Top 5 suggestions
        
    async def get_evolution_statistics(self) -> Dict[str, Any]:
        """Get statistics about evolution progress"""
        
        if not self.gene_pool:
            return {"message": "No genes in pool"}
            
        # Overall statistics
        total_genes = len(self.gene_pool)
        avg_fitness = sum(gene.fitness_score for gene in self.gene_pool.values()) / total_genes
        
        # Fitness distribution
        fitness_scores = [gene.fitness_score for gene in self.gene_pool.values()]
        fitness_distribution = {
            "excellent": len([f for f in fitness_scores if f > 0.8]),
            "good": len([f for f in fitness_scores if 0.6 < f <= 0.8]),
            "average": len([f for f in fitness_scores if 0.4 < f <= 0.6]),
            "poor": len([f for f in fitness_scores if f <= 0.4])
        }
        
        # Category analysis
        category_stats = {}
        for gene in self.gene_pool.values():
            if gene.category not in category_stats:
                category_stats[gene.category] = {"count": 0, "avg_fitness": 0.0, "total_fitness": 0.0}
            category_stats[gene.category]["count"] += 1
            category_stats[gene.category]["total_fitness"] += gene.fitness_score
            
        for category, stats in category_stats.items():
            stats["avg_fitness"] = stats["total_fitness"] / stats["count"]
            del stats["total_fitness"]
            
        # Innovation tracking
        recent_innovations = [log for log in self.innovation_log 
                            if log["timestamp"] > datetime.utcnow() - timedelta(days=7)]
        
        return {
            "generation": self.generation_count,
            "population_size": total_genes,
            "average_fitness": avg_fitness,
            "fitness_distribution": fitness_distribution,
            "category_performance": category_stats,
            "recent_innovations": len(recent_innovations),
            "exploration_queue_size": len(self.exploration_queue),
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate
        }
        
    async def force_speciation_event(self):
        """Force a speciation event to explore radically new approaches"""
        
        logger.info("Initiating forced speciation event")
        
        # Generate novel approaches for each category
        categories = set(gene.category for gene in self.gene_pool.values())
        
        for category in categories:
            # Create radical mutations
            base_genes = [gene for gene in self.gene_pool.values() if gene.category == category]
            if base_genes:
                best_gene = max(base_genes, key=lambda g: g.fitness_score)
                
                # Create 3 radical mutations
                for i in range(3):
                    radical_gene = await self._create_radical_mutation(best_gene)
                    self.gene_pool[radical_gene.id] = radical_gene
                    
        # Log speciation event
        self.innovation_log.append({
            "type": "speciation_event",
            "timestamp": datetime.utcnow(),
            "new_genes_created": len(categories) * 3,
            "trigger": "forced"
        })
        
        await self.emit_event(
            EventType.KNOWLEDGE_LEARNED,
            {"action": "speciation_event", "new_genes": len(categories) * 3}
        )
        
    # Private methods
    
    async def _create_seed_population(self):
        """Create initial population of tactic genes"""
        
        # Base reconnaissance tactics
        recon_tactics = [
            {
                "approach": "passive_osint",
                "parameters": {"sources": ["search_engines", "social_media"], "depth": "comprehensive"}
            },
            {
                "approach": "dns_enumeration", 
                "parameters": {"methods": ["zone_transfer", "brute_force"], "wordlists": ["common", "extended"]}
            },
            {
                "approach": "subdomain_discovery",
                "parameters": {"tools": ["amass", "subfinder"], "passive_only": True}
            }
        ]
        
        # Base exploitation tactics
        exploit_tactics = [
            {
                "approach": "web_application_testing",
                "parameters": {"tests": ["injection", "xss", "auth_bypass"], "automation_level": "guided"}
            },
            {
                "approach": "network_service_analysis",
                "parameters": {"scan_type": "stealth", "service_enumeration": True}
            }
        ]
        
        # Base analysis tactics
        analysis_tactics = [
            {
                "approach": "vulnerability_prioritization",
                "parameters": {"scoring_method": "cvss", "business_context": True}
            },
            {
                "approach": "attack_surface_mapping",
                "parameters": {"comprehensive": True, "include_dependencies": True}
            }
        ]
        
        all_tactics = [
            ("reconnaissance", recon_tactics),
            ("exploitation", exploit_tactics), 
            ("analysis", analysis_tactics)
        ]
        
        gene_id = 0
        for category, tactics in all_tactics:
            for tactic in tactics:
                gene = TacticGene(
                    id=f"seed_{category}_{gene_id}",
                    category=category,
                    approach=tactic["approach"],
                    parameters=tactic["parameters"],
                    fitness_score=0.5,  # Neutral starting fitness
                    usage_count=0,
                    success_count=0,
                    last_success=None,
                    creation_method="original",
                    parent_genes=[],
                    age=0
                )
                
                self.gene_pool[gene.id] = gene
                gene_id += 1
                
        logger.info(f"Created seed population with {len(self.gene_pool)} genes")
        
    def _tournament_selection(self, candidates: List[TacticGene], tournament_size: int) -> List[TacticGene]:
        """Tournament selection for choosing genes for evolution"""
        
        if len(candidates) <= tournament_size:
            return candidates
            
        selected = []
        remaining_candidates = candidates.copy()
        
        for _ in range(min(tournament_size, len(candidates))):
            # Random tournament
            tournament_participants = random.sample(remaining_candidates, 
                                                   min(3, len(remaining_candidates)))
            
            # Select best from tournament
            winner = max(tournament_participants, key=lambda g: g.fitness_score)
            selected.append(winner)
            remaining_candidates.remove(winner)
            
        return selected
        
    async def _mutate_gene(self, base_gene: TacticGene, context: Dict[str, Any]) -> TacticGene:
        """Create a mutated version of a gene"""
        
        mutated_gene = copy.deepcopy(base_gene)
        mutated_gene.id = f"mutated_{base_gene.id}_{len(self.gene_pool)}"
        mutated_gene.creation_method = "mutated"
        mutated_gene.parent_genes = [base_gene.id]
        mutated_gene.age = 0
        mutated_gene.usage_count = 0
        mutated_gene.success_count = 0
        mutated_gene.fitness_score = base_gene.fitness_score * 0.8  # Slight penalty for being untested
        
        # Mutate parameters
        for param_name, param_value in mutated_gene.parameters.items():
            if random.random() < self.mutation_rate:
                mutated_gene.parameters[param_name] = self._mutate_parameter(param_value, context)
                
        # Sometimes add new parameters
        if random.random() < 0.1:  # 10% chance
            new_param = self._generate_novel_parameter(mutated_gene.category, context)
            if new_param:
                mutated_gene.parameters.update(new_param)
                
        # Store the mutated gene
        self.gene_pool[mutated_gene.id] = mutated_gene
        
        # Log innovation
        self.innovation_log.append({
            "type": "mutation",
            "timestamp": datetime.utcnow(),
            "parent_gene": base_gene.id,
            "new_gene": mutated_gene.id,
            "changes": "parameter_mutation"
        })
        
        return mutated_gene
        
    async def _crossover_genes(self, parent1: TacticGene, parent2: TacticGene, 
                              context: Dict[str, Any]) -> TacticGene:
        """Create offspring through crossover of two parent genes"""
        
        # Create new gene combining traits from both parents
        offspring = TacticGene(
            id=f"crossover_{parent1.id}_{parent2.id}_{len(self.gene_pool)}",
            category=parent1.category,  # Assume same category
            approach=f"hybrid_{parent1.approach}_{parent2.approach}",
            parameters={},
            fitness_score=(parent1.fitness_score + parent2.fitness_score) / 2 * 0.9,  # Slight penalty
            usage_count=0,
            success_count=0,
            last_success=None,
            creation_method="crossover",
            parent_genes=[parent1.id, parent2.id],
            age=0
        )
        
        # Combine parameters from both parents
        all_params = set(parent1.parameters.keys()) | set(parent2.parameters.keys())
        
        for param in all_params:
            if param in parent1.parameters and param in parent2.parameters:
                # Both parents have this parameter - choose randomly or blend
                if random.random() < 0.5:
                    offspring.parameters[param] = parent1.parameters[param]
                else:
                    offspring.parameters[param] = parent2.parameters[param]
            elif param in parent1.parameters:
                offspring.parameters[param] = parent1.parameters[param]
            elif param in parent2.parameters:
                offspring.parameters[param] = parent2.parameters[param]
                
        # Store the offspring
        self.gene_pool[offspring.id] = offspring
        
        # Log innovation
        self.innovation_log.append({
            "type": "crossover",
            "timestamp": datetime.utcnow(),
            "parent_genes": [parent1.id, parent2.id],
            "offspring_gene": offspring.id,
            "combined_parameters": len(offspring.parameters)
        })
        
        return offspring
        
    def _mutate_parameter(self, original_value: Any, context: Dict[str, Any]) -> Any:
        """Mutate a single parameter value"""
        
        if isinstance(original_value, bool):
            return not original_value
        elif isinstance(original_value, int):
            return max(1, original_value + random.randint(-2, 2))
        elif isinstance(original_value, float):
            return max(0.1, original_value * random.uniform(0.8, 1.2))
        elif isinstance(original_value, str):
            # Simple string mutations
            alternatives = {
                "comprehensive": "focused",
                "passive": "active", 
                "stealth": "aggressive",
                "automated": "manual",
                "deep": "shallow"
            }
            return alternatives.get(original_value, original_value)
        elif isinstance(original_value, list):
            # Add, remove, or modify list elements
            mutated_list = original_value.copy()
            if mutated_list and random.random() < 0.5:
                # Remove random element
                mutated_list.remove(random.choice(mutated_list))
            else:
                # Add potential new element
                potential_additions = ["extended", "custom", "advanced", "specialized"]
                if potential_additions:
                    mutated_list.append(random.choice(potential_additions))
            return mutated_list
        else:
            return original_value
            
    def _generate_novel_parameter(self, category: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate a novel parameter for a gene"""
        
        novel_params = {
            "reconnaissance": {
                "ai_assistance": True,
                "correlation_analysis": True,
                "real_time_monitoring": False
            },
            "exploitation": {
                "payload_encoding": "advanced",
                "evasion_techniques": ["encoding", "fragmentation"],
                "timing_optimization": True
            },
            "analysis": {
                "ml_scoring": True,
                "threat_modeling": "STRIDE",
                "business_impact_analysis": True
            }
        }
        
        if category in novel_params:
            param_name = random.choice(list(novel_params[category].keys()))
            return {param_name: novel_params[category][param_name]}
            
        return None
        
    async def _generate_novel_approach(self, category: str, context: Dict[str, Any], 
                                     target_performance: float) -> Optional[TacticGene]:
        """Generate a completely novel approach for a category"""
        
        novel_approaches = {
            "reconnaissance": [
                "ai_powered_osint",
                "behavioral_pattern_analysis", 
                "supply_chain_mapping",
                "dark_web_intelligence"
            ],
            "exploitation": [
                "ml_guided_fuzzing",
                "adaptive_payload_generation",
                "context_aware_exploitation",
                "zero_day_discovery"
            ],
            "analysis": [
                "quantum_threat_modeling",
                "predictive_vulnerability_analysis",
                "ecosystem_risk_assessment",
                "ai_assisted_prioritization"
            ]
        }
        
        if category not in novel_approaches:
            return None
            
        approach = random.choice(novel_approaches[category])
        
        # Generate novel parameters based on approach
        if "ai" in approach or "ml" in approach:
            parameters = {
                "model_type": "neural_network",
                "training_data": "historical_campaigns",
                "confidence_threshold": 0.8,
                "human_validation": True
            }
        elif "predictive" in approach:
            parameters = {
                "prediction_horizon": "30_days",
                "risk_factors": ["emerging_threats", "technology_changes"],
                "update_frequency": "daily"
            }
        else:
            parameters = {
                "innovation_level": "experimental",
                "validation_required": True,
                "fallback_method": "traditional"
            }
            
        novel_gene = TacticGene(
            id=f"novel_{category}_{approach}_{len(self.gene_pool)}",
            category=category,
            approach=approach,
            parameters=parameters,
            fitness_score=0.3,  # Lower initial fitness for novel approaches
            usage_count=0,
            success_count=0,
            last_success=None,
            creation_method="novel_generation",
            parent_genes=[],
            age=0
        )
        
        self.gene_pool[novel_gene.id] = novel_gene
        
        # Log innovation
        self.innovation_log.append({
            "type": "novel_generation",
            "timestamp": datetime.utcnow(),
            "category": category,
            "approach": approach,
            "target_performance": target_performance
        })
        
        return novel_gene
        
    async def _create_radical_mutation(self, base_gene: TacticGene) -> TacticGene:
        """Create a radical mutation that significantly changes the approach"""
        
        radical_gene = copy.deepcopy(base_gene)
        radical_gene.id = f"radical_{base_gene.id}_{len(self.gene_pool)}"
        radical_gene.creation_method = "radical_mutation"
        radical_gene.parent_genes = [base_gene.id]
        radical_gene.fitness_score = 0.2  # Very low initial fitness
        radical_gene.usage_count = 0
        radical_gene.success_count = 0
        
        # Radical changes to approach
        radical_gene.approach = f"experimental_{base_gene.approach}"
        
        # Significantly mutate all parameters
        for param_name in radical_gene.parameters:
            radical_gene.parameters[param_name] = self._mutate_parameter(
                radical_gene.parameters[param_name], {}
            )
            
        # Add several novel parameters
        for _ in range(3):
            novel_param = self._generate_novel_parameter(radical_gene.category, {})
            if novel_param:
                radical_gene.parameters.update(novel_param)
                
        return radical_gene
        
    def _analyze_category_performance(self) -> Dict[str, float]:
        """Analyze average performance by category"""
        
        category_performance = {}
        
        for gene in self.gene_pool.values():
            if gene.category not in category_performance:
                category_performance[gene.category] = []
            category_performance[gene.category].append(gene.fitness_score)
            
        # Calculate averages
        for category, scores in category_performance.items():
            category_performance[category] = sum(scores) / len(scores)
            
        return category_performance
        
    async def _learn_from_outcome(self, gene: TacticGene, success: bool, 
                                 performance_score: float, context: Dict[str, Any]):
        """Learn from tactic outcome and update related knowledge"""
        
        # Update knowledge graph with performance insights
        insight_content = f"Tactic {gene.approach} in {gene.category} achieved {performance_score:.2f} performance"
        
        await self.knowledge_graph.add_knowledge(
            content=insight_content,
            node_type="tactic_performance",
            source="mutation_engine",
            metadata={
                "gene_id": gene.id,
                "performance_score": performance_score,
                "success": success,
                "parameters": gene.parameters
            }
        )
        
        # If this was a particularly good or bad outcome, create exploration tasks
        if performance_score > 0.9:
            # Excellent performance - explore mutations of this gene
            exploration_task = ExplorationTask(
                id=f"explore_excellent_{gene.id}",
                task_type="mutation_test",
                description=f"Explore mutations of high-performing gene {gene.id}",
                target_category=gene.category,
                hypothesis=f"Mutations of {gene.approach} can maintain high performance",
                test_parameters={"base_gene": gene.id, "mutation_count": 5},
                expected_outcome="Consistently high performance variants",
                priority=0.9,
                created_at=datetime.utcnow()
            )
            self.exploration_queue.append(exploration_task)
            
        elif performance_score < 0.3 and gene.usage_count > 5:
            # Poor performance - mark for potential removal
            gene.fitness_score *= 0.5  # Severe penalty
            
    async def _evolution_loop(self):
        """Background evolution process"""
        
        while True:
            try:
                await asyncio.sleep(3600)  # Every hour
                
                # Age all genes
                for gene in self.gene_pool.values():
                    gene.age += 1
                    gene.calculate_fitness()  # Recalculate with age factor
                    
                # Selection pressure - remove worst performing genes
                if len(self.gene_pool) > self.population_size:
                    sorted_genes = sorted(self.gene_pool.values(), 
                                        key=lambda g: g.fitness_score, reverse=True)
                    
                    # Keep top performers
                    survivors = sorted_genes[:self.population_size]
                    
                    # Clear gene pool and repopulate with survivors
                    self.gene_pool.clear()
                    for gene in survivors:
                        self.gene_pool[gene.id] = gene
                        
                    # Log selection event
                    self.innovation_log.append({
                        "type": "natural_selection",
                        "timestamp": datetime.utcnow(),
                        "survivors": len(survivors),
                        "removed": len(sorted_genes) - len(survivors)
                    })
                    
                self.generation_count += 1
                
                # Periodic speciation events
                if self.generation_count % 24 == 0:  # Every 24 hours
                    await self.force_speciation_event()
                    
            except Exception as e:
                logger.error(f"Error in evolution loop: {e}")
                await asyncio.sleep(300)
                
    async def _exploration_loop(self):
        """Background exploration task processing"""
        
        while True:
            try:
                await asyncio.sleep(1800)  # Every 30 minutes
                
                if not self.exploration_queue:
                    continue
                    
                # Process highest priority exploration task
                task = max(self.exploration_queue, key=lambda t: t.priority)
                
                if task.attempts >= task.max_attempts:
                    self.exploration_queue.remove(task)
                    continue
                    
                # Execute exploration task
                await self._execute_exploration_task(task)
                task.attempts += 1
                
                # Remove completed tasks
                if task.attempts >= task.max_attempts:
                    self.exploration_queue.remove(task)
                    
            except Exception as e:
                logger.error(f"Error in exploration loop: {e}")
                await asyncio.sleep(300)
                
    async def _execute_exploration_task(self, task: ExplorationTask):
        """Execute a specific exploration task"""
        
        if task.task_type == "mutation_test":
            # Create mutations and test them
            base_gene_id = task.test_parameters.get("base_gene")
            if base_gene_id and base_gene_id in self.gene_pool:
                base_gene = self.gene_pool[base_gene_id]
                
                # Create several mutations
                for i in range(3):
                    mutated_gene = await self._mutate_gene(base_gene, {})
                    # Mutations will be tested naturally through usage
                    
        elif task.task_type == "cross_domain_transfer":
            # Transfer successful patterns between categories
            source_cat = task.test_parameters.get("source_category")
            target_cat = task.test_parameters.get("target_category")
            
            if source_cat and target_cat:
                # Find best genes in source category
                source_genes = [g for g in self.gene_pool.values() 
                              if g.category == source_cat and g.fitness_score > 0.7]
                
                if source_genes:
                    best_source = max(source_genes, key=lambda g: g.fitness_score)
                    
                    # Create adapted version for target category
                    adapted_gene = copy.deepcopy(best_source)
                    adapted_gene.id = f"transfer_{source_cat}_to_{target_cat}_{len(self.gene_pool)}"
                    adapted_gene.category = target_cat
                    adapted_gene.creation_method = "cross_domain_transfer"
                    adapted_gene.fitness_score = best_source.fitness_score * 0.6  # Penalty for untested domain
                    adapted_gene.usage_count = 0
                    adapted_gene.success_count = 0
                    
                    self.gene_pool[adapted_gene.id] = adapted_gene
                    
        # Log exploration attempt
        self.innovation_log.append({
            "type": "exploration_task",
            "timestamp": datetime.utcnow(),
            "task_id": task.id,
            "task_type": task.task_type,
            "attempt": task.attempts + 1
        })
        
    async def shutdown(self):
        """Shutdown mutation engine"""
        logger.info("Mutation engine shutdown complete")