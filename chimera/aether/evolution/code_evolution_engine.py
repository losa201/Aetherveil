"""
Aether Code Evolution Engine: Self-modifying code system with sandbox testing
"""

import asyncio
import logging
import ast
import os
import json
import hashlib
import shutil
import subprocess
import tempfile
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import difflib
import re
import inspect

logger = logging.getLogger(__name__)

class MutationType(Enum):
    """Types of code mutations"""
    REFACTOR = "refactor"
    ENHANCEMENT = "enhancement"
    BUG_FIX = "bug_fix"
    OPTIMIZATION = "optimization"
    NEW_FEATURE = "new_feature"
    DOCUMENTATION = "documentation"

class MutationStatus(Enum):
    """Status of code mutations"""
    PROPOSED = "proposed"
    TESTING = "testing"
    VALIDATED = "validated"
    APPLIED = "applied"
    REJECTED = "rejected"
    FAILED = "failed"

@dataclass
class CodeMutation:
    """Represents a proposed code change"""
    mutation_id: str
    mutation_type: MutationType
    file_path: str
    original_code: str
    modified_code: str
    description: str
    confidence: float
    expected_benefits: List[str]
    risk_level: float
    test_results: Optional[Dict[str, Any]] = None
    status: MutationStatus = MutationStatus.PROPOSED
    created_at: datetime = None
    applied_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class TestResult:
    """Results from testing a code mutation"""
    success: bool
    test_output: str
    execution_time: float
    memory_usage: float
    error_message: Optional[str] = None
    performance_metrics: Optional[Dict[str, float]] = None
    security_analysis: Optional[Dict[str, Any]] = None

@dataclass
class Hypothesis:
    """Code improvement hypothesis"""
    hypothesis_id: str
    description: str
    target_area: str
    proposed_changes: List[str]
    expected_impact: str
    confidence: float
    supporting_evidence: List[str]
    related_mutations: List[str] = None

class CodeEvolutionEngine:
    """
    Continuously scans and improves the agent's own codebase
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.codebase_path = Path(config.get('codebase_path', Path(__file__).parent.parent))
        
        # Mutation tracking
        self.proposed_mutations: List[CodeMutation] = []
        self.applied_mutations: List[CodeMutation] = []
        self.hypotheses: List[Hypothesis] = []
        
        # Safety constraints
        self.max_mutations_per_cycle = config.get('max_mutations_per_cycle', 3)
        self.min_confidence_threshold = config.get('min_confidence_threshold', 0.7)
        self.backup_retention_days = config.get('backup_retention_days', 7)
        
        # Performance tracking
        self.mutation_success_rate = 0.0
        self.performance_history = []
        
        # Sandbox configuration
        self.sandbox_path = Path(tempfile.gettempdir()) / "aether_sandbox"
        self.test_timeout = config.get('test_timeout', 300)  # 5 minutes
        
    async def initialize(self):
        """Initialize the code evolution engine"""
        logger.info("Initializing Code Evolution Engine...")
        
        # Create sandbox directory
        self.sandbox_path.mkdir(exist_ok=True)
        
        # Create backup directory
        self.backup_path = self.codebase_path / ".aether_backups"
        self.backup_path.mkdir(exist_ok=True)
        
        # Initial codebase scan
        await self.scan_codebase()
        
        logger.info("Code Evolution Engine initialized")
        
    async def scan_codebase(self) -> Dict[str, Any]:
        """Deep analysis of current code quality, patterns, and weaknesses"""
        
        logger.info("Scanning codebase for improvement opportunities...")
        
        scan_results = {
            'timestamp': datetime.utcnow(),
            'files_analyzed': 0,
            'total_lines': 0,
            'functions_analyzed': 0,
            'classes_analyzed': 0,
            'weaknesses': [],
            'strengths': [],
            'improvement_opportunities': [],
            'complexity_metrics': {},
            'quality_score': 0.0
        }
        
        try:
            python_files = list(self.codebase_path.rglob('*.py'))
            python_files = [f for f in python_files if not any(skip in str(f) for skip in ['.git', '__pycache__', '.aether_backups'])]
            
            for py_file in python_files:
                file_analysis = await self._analyze_file(py_file)
                
                scan_results['files_analyzed'] += 1
                scan_results['total_lines'] += file_analysis.get('lines', 0)
                scan_results['functions_analyzed'] += file_analysis.get('functions', 0)
                scan_results['classes_analyzed'] += file_analysis.get('classes', 0)
                
                # Collect weaknesses and opportunities
                scan_results['weaknesses'].extend(file_analysis.get('weaknesses', []))
                scan_results['improvement_opportunities'].extend(file_analysis.get('opportunities', []))
                
            # Calculate overall quality metrics
            scan_results['quality_score'] = await self._calculate_quality_score(scan_results)
            scan_results['complexity_metrics'] = await self._calculate_complexity_metrics(scan_results)
            
            # Generate improvement hypotheses
            self.hypotheses = await self.generate_improvement_hypotheses(scan_results)
            
            logger.info(f"Codebase scan complete - {scan_results['files_analyzed']} files, "
                       f"Quality Score: {scan_results['quality_score']:.3f}")
            
        except Exception as e:
            logger.error(f"Error during codebase scan: {e}")
            scan_results['error'] = str(e)
            
        return scan_results
        
    async def generate_improvement_hypotheses(self, scan_results: Dict[str, Any]) -> List[Hypothesis]:
        """Create testable hypotheses about code improvements"""
        
        hypotheses = []
        
        # Analyze weaknesses and generate hypotheses
        weaknesses = scan_results.get('weaknesses', [])
        
        for weakness in weaknesses:
            if weakness['type'] == 'long_function':
                hypothesis = Hypothesis(
                    hypothesis_id=f"refactor_{weakness['file']}_{weakness['function']}",
                    description=f"Break down long function {weakness['function']} into smaller, more focused functions",
                    target_area=weakness['file'],
                    proposed_changes=[
                        "Extract logical units into separate functions",
                        "Maintain single responsibility principle",
                        "Improve readability and testability"
                    ],
                    expected_impact="Improved maintainability and readability",
                    confidence=0.8,
                    supporting_evidence=[
                        f"Function has {weakness.get('lines', 0)} lines",
                        "Long functions are harder to understand and test"
                    ]
                )
                hypotheses.append(hypothesis)
                
            elif weakness['type'] == 'missing_docstring':
                hypothesis = Hypothesis(
                    hypothesis_id=f"document_{weakness['file']}_{weakness['function']}",
                    description=f"Add comprehensive docstring to {weakness['function']}",
                    target_area=weakness['file'],
                    proposed_changes=[
                        "Add detailed function description",
                        "Document parameters and return values",
                        "Include usage examples"
                    ],
                    expected_impact="Improved code documentation and maintainability",
                    confidence=0.9,
                    supporting_evidence=[
                        "Function lacks documentation",
                        "Documentation improves code understanding"
                    ]
                )
                hypotheses.append(hypothesis)
                
            elif weakness['type'] == 'high_complexity':
                hypothesis = Hypothesis(
                    hypothesis_id=f"simplify_{weakness['file']}_{weakness['function']}",
                    description=f"Reduce complexity in {weakness['function']}",
                    target_area=weakness['file'],
                    proposed_changes=[
                        "Extract complex logic into helper functions",
                        "Reduce nesting levels",
                        "Simplify conditional logic"
                    ],
                    expected_impact="Reduced cognitive load and improved maintainability",
                    confidence=0.7,
                    supporting_evidence=[
                        f"Cyclomatic complexity: {weakness.get('complexity', 0)}",
                        "High complexity increases bug probability"
                    ]
                )
                hypotheses.append(hypothesis)
        
        # Generate enhancement hypotheses based on opportunities
        opportunities = scan_results.get('improvement_opportunities', [])
        
        for opportunity in opportunities:
            if opportunity['type'] == 'performance_optimization':
                hypothesis = Hypothesis(
                    hypothesis_id=f"optimize_{opportunity['file']}_{opportunity['location']}",
                    description=f"Optimize performance in {opportunity['location']}",
                    target_area=opportunity['file'],
                    proposed_changes=opportunity.get('suggestions', []),
                    expected_impact="Improved execution speed and resource usage",
                    confidence=opportunity.get('confidence', 0.6),
                    supporting_evidence=opportunity.get('evidence', [])
                )
                hypotheses.append(hypothesis)
        
        logger.info(f"Generated {len(hypotheses)} improvement hypotheses")
        return hypotheses
        
    async def create_mutation_from_hypothesis(self, hypothesis: Hypothesis) -> Optional[CodeMutation]:
        """Create a testable code mutation from a hypothesis"""
        
        try:
            # Read the target file
            file_path = self.codebase_path / hypothesis.target_area
            
            if not file_path.exists():
                logger.warning(f"Target file not found: {file_path}")
                return None
                
            with open(file_path, 'r', encoding='utf-8') as f:
                original_code = f.read()
            
            # Generate modified code based on hypothesis
            modified_code = await self._apply_hypothesis_changes(original_code, hypothesis)
            
            if modified_code == original_code:
                logger.info(f"No changes generated for hypothesis: {hypothesis.hypothesis_id}")
                return None
            
            # Create mutation
            mutation = CodeMutation(
                mutation_id=f"mut_{hypothesis.hypothesis_id}_{int(datetime.utcnow().timestamp())}",
                mutation_type=self._determine_mutation_type(hypothesis),
                file_path=str(file_path),
                original_code=original_code,
                modified_code=modified_code,
                description=hypothesis.description,
                confidence=hypothesis.confidence,
                expected_benefits=hypothesis.proposed_changes,
                risk_level=self._calculate_risk_level(hypothesis, original_code, modified_code)
            )
            
            self.proposed_mutations.append(mutation)
            
            logger.info(f"Created mutation: {mutation.mutation_id}")
            return mutation
            
        except Exception as e:
            logger.error(f"Error creating mutation from hypothesis {hypothesis.hypothesis_id}: {e}")
            return None
            
    async def test_in_sandbox(self, mutation: CodeMutation) -> TestResult:
        """Test all changes in isolated environment"""
        
        mutation.status = MutationStatus.TESTING
        
        logger.info(f"Testing mutation {mutation.mutation_id} in sandbox...")
        
        try:
            # Create sandbox copy of codebase
            sandbox_codebase = self.sandbox_path / f"test_{mutation.mutation_id}"
            
            if sandbox_codebase.exists():
                shutil.rmtree(sandbox_codebase)
                
            shutil.copytree(self.codebase_path, sandbox_codebase, 
                          ignore=shutil.ignore_patterns('*.git*', '__pycache__', '.aether_backups'))
            
            # Apply mutation to sandbox
            target_file = sandbox_codebase / Path(mutation.file_path).relative_to(self.codebase_path)
            
            with open(target_file, 'w', encoding='utf-8') as f:
                f.write(mutation.modified_code)
            
            # Run tests
            test_start = datetime.utcnow()
            
            # Basic syntax check
            syntax_check = await self._check_syntax(target_file)
            if not syntax_check['success']:
                return TestResult(
                    success=False,
                    test_output=syntax_check['output'],
                    execution_time=0.0,
                    memory_usage=0.0,
                    error_message="Syntax error in modified code"
                )
            
            # Run unit tests if available
            test_results = await self._run_tests(sandbox_codebase)
            
            # Performance analysis
            performance_metrics = await self._analyze_performance(sandbox_codebase, mutation)
            
            # Security analysis
            security_analysis = await self._analyze_security(target_file, mutation)
            
            test_duration = (datetime.utcnow() - test_start).total_seconds()
            
            success = (test_results.get('success', True) and 
                      syntax_check['success'] and
                      security_analysis.get('safe', True))
            
            result = TestResult(
                success=success,
                test_output=test_results.get('output', ''),
                execution_time=test_duration,
                memory_usage=test_results.get('memory_usage', 0.0),
                error_message=test_results.get('error') if not success else None,
                performance_metrics=performance_metrics,
                security_analysis=security_analysis
            )
            
            mutation.test_results = asdict(result)
            mutation.status = MutationStatus.VALIDATED if success else MutationStatus.FAILED
            
            # Cleanup sandbox
            shutil.rmtree(sandbox_codebase)
            
            logger.info(f"Testing complete for {mutation.mutation_id} - Success: {success}")
            return result
            
        except Exception as e:
            logger.error(f"Error testing mutation {mutation.mutation_id}: {e}")
            mutation.status = MutationStatus.FAILED
            
            return TestResult(
                success=False,
                test_output="",
                execution_time=0.0,
                memory_usage=0.0,
                error_message=str(e)
            )
            
    async def apply_validated_mutation(self, mutation: CodeMutation) -> bool:
        """Safely apply code changes after validation"""
        
        if mutation.status != MutationStatus.VALIDATED:
            logger.warning(f"Cannot apply unvalidated mutation: {mutation.mutation_id}")
            return False
        
        try:
            # Create backup
            backup_success = await self._create_backup(mutation.file_path)
            if not backup_success:
                logger.error(f"Failed to create backup for {mutation.file_path}")
                return False
            
            # Apply the mutation
            with open(mutation.file_path, 'w', encoding='utf-8') as f:
                f.write(mutation.modified_code)
            
            mutation.status = MutationStatus.APPLIED
            mutation.applied_at = datetime.utcnow()
            self.applied_mutations.append(mutation)
            
            # Update success rate
            self._update_success_rate()
            
            logger.info(f"Successfully applied mutation: {mutation.mutation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error applying mutation {mutation.mutation_id}: {e}")
            
            # Attempt to restore from backup
            await self._restore_from_backup(mutation.file_path)
            return False
            
    # Private helper methods
    
    async def _analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single Python file for quality metrics"""
        
        analysis = {
            'file': str(file_path.relative_to(self.codebase_path)),
            'lines': 0,
            'functions': 0,
            'classes': 0,
            'weaknesses': [],
            'opportunities': [],
            'complexity_score': 0.0
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            lines = content.splitlines()
            analysis['lines'] = len(lines)
            
            # Parse AST
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    analysis['functions'] += 1
                    func_analysis = await self._analyze_function(node, content)
                    
                    if func_analysis['weaknesses']:
                        analysis['weaknesses'].extend(func_analysis['weaknesses'])
                    if func_analysis['opportunities']:
                        analysis['opportunities'].extend(func_analysis['opportunities'])
                        
                elif isinstance(node, ast.ClassDef):
                    analysis['classes'] += 1
                    
        except Exception as e:
            logger.warning(f"Could not analyze {file_path}: {e}")
            analysis['error'] = str(e)
            
        return analysis
        
    async def _analyze_function(self, func_node: ast.FunctionDef, file_content: str) -> Dict[str, Any]:
        """Analyze a function for quality issues"""
        
        analysis = {
            'name': func_node.name,
            'weaknesses': [],
            'opportunities': [],
            'complexity': 0
        }
        
        # Count lines in function
        func_lines = func_node.end_lineno - func_node.lineno if hasattr(func_node, 'end_lineno') else 0
        
        # Check for long functions
        if func_lines > 50:
            analysis['weaknesses'].append({
                'type': 'long_function',
                'function': func_node.name,
                'lines': func_lines,
                'file': file_content
            })
        
        # Check for missing docstring
        if not ast.get_docstring(func_node):
            analysis['weaknesses'].append({
                'type': 'missing_docstring',
                'function': func_node.name,
                'file': file_content
            })
        
        # Calculate cyclomatic complexity
        complexity = self._calculate_cyclomatic_complexity(func_node)
        analysis['complexity'] = complexity
        
        if complexity > 10:
            analysis['weaknesses'].append({
                'type': 'high_complexity',
                'function': func_node.name,
                'complexity': complexity,
                'file': file_content
            })
        
        return analysis
        
    def _calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of a function"""
        
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.With, ast.AsyncWith):
                complexity += 1
            elif isinstance(child, (ast.BoolOp, ast.Compare)):
                # Count boolean operators
                if isinstance(child, ast.BoolOp):
                    complexity += len(child.values) - 1
                elif isinstance(child, ast.Compare):
                    complexity += len(child.ops)
                    
        return complexity
        
    async def _apply_hypothesis_changes(self, original_code: str, hypothesis: Hypothesis) -> str:
        """Apply changes suggested by hypothesis to code"""
        
        # This is a simplified implementation
        # In a real system, this would use more sophisticated code transformation
        
        modified_code = original_code
        
        # Example transformations based on hypothesis type
        if "docstring" in hypothesis.description.lower():
            # Add docstrings where missing
            modified_code = await self._add_missing_docstrings(modified_code)
            
        elif "refactor" in hypothesis.description.lower():
            # Simple refactoring (placeholder)
            modified_code = await self._simple_refactor(modified_code)
            
        elif "optimize" in hypothesis.description.lower():
            # Performance optimizations (placeholder)
            modified_code = await self._apply_optimizations(modified_code)
            
        return modified_code
        
    async def _add_missing_docstrings(self, code: str) -> str:
        """Add basic docstrings to functions missing them"""
        
        try:
            tree = ast.parse(code)
            lines = code.splitlines()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and not ast.get_docstring(node):
                    # Insert basic docstring
                    insert_line = node.lineno
                    indent = len(lines[insert_line - 1]) - len(lines[insert_line - 1].lstrip())
                    docstring = f'{" " * (indent + 4)}"""TODO: Add proper docstring for {node.name}"""'
                    lines.insert(insert_line, docstring)
                    
            return '\n'.join(lines)
            
        except Exception:
            return code
            
    async def _simple_refactor(self, code: str) -> str:
        """Apply simple refactoring improvements"""
        # Placeholder - would implement actual refactoring logic
        return code
        
    async def _apply_optimizations(self, code: str) -> str:
        """Apply performance optimizations"""
        # Placeholder - would implement optimization logic
        return code
        
    def _determine_mutation_type(self, hypothesis: Hypothesis) -> MutationType:
        """Determine mutation type from hypothesis"""
        
        description = hypothesis.description.lower()
        
        if "docstring" in description or "document" in description:
            return MutationType.DOCUMENTATION
        elif "refactor" in description or "break down" in description:
            return MutationType.REFACTOR
        elif "optimize" in description or "performance" in description:
            return MutationType.OPTIMIZATION
        elif "fix" in description:
            return MutationType.BUG_FIX
        elif "simplify" in description or "complexity" in description:
            return MutationType.REFACTOR
        else:
            return MutationType.ENHANCEMENT
            
    def _calculate_risk_level(self, hypothesis: Hypothesis, original: str, modified: str) -> float:
        """Calculate risk level of applying a mutation"""
        
        # Simple risk calculation based on amount of change
        diff_ratio = len(modified) / max(len(original), 1)
        
        if diff_ratio > 1.5 or diff_ratio < 0.5:
            risk = 0.8  # High risk for major changes
        elif hypothesis.confidence < 0.5:
            risk = 0.7  # High risk for low confidence
        elif "refactor" in hypothesis.description.lower():
            risk = 0.4  # Medium risk for refactoring
        else:
            risk = 0.2  # Low risk for simple changes
            
        return min(1.0, risk)
        
    async def _check_syntax(self, file_path: Path) -> Dict[str, Any]:
        """Check Python syntax of modified file"""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            ast.parse(content)
            
            return {'success': True, 'output': 'Syntax check passed'}
            
        except SyntaxError as e:
            return {
                'success': False,
                'output': f'Syntax error: {e}',
                'error': str(e)
            }
        except Exception as e:
            return {
                'success': False,
                'output': f'Error checking syntax: {e}',
                'error': str(e)
            }
            
    async def _run_tests(self, sandbox_path: Path) -> Dict[str, Any]:
        """Run tests in sandbox environment"""
        
        try:
            # Look for test files
            test_files = list(sandbox_path.rglob('test_*.py')) + list(sandbox_path.rglob('*_test.py'))
            
            if not test_files:
                return {'success': True, 'output': 'No tests found'}
            
            # Run pytest if available
            result = subprocess.run(
                ['python', '-m', 'pytest', '--tb=short'],
                cwd=sandbox_path,
                capture_output=True,
                text=True,
                timeout=self.test_timeout
            )
            
            return {
                'success': result.returncode == 0,
                'output': result.stdout + result.stderr,
                'returncode': result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'output': 'Tests timed out',
                'error': 'Timeout'
            }
        except Exception as e:
            return {
                'success': False,
                'output': f'Error running tests: {e}',
                'error': str(e)
            }
            
    async def _analyze_performance(self, sandbox_path: Path, mutation: CodeMutation) -> Dict[str, float]:
        """Analyze performance impact of mutation"""
        
        # Placeholder - would implement actual performance analysis
        return {
            'execution_time_change': 0.0,
            'memory_usage_change': 0.0,
            'complexity_change': 0.0
        }
        
    async def _analyze_security(self, file_path: Path, mutation: CodeMutation) -> Dict[str, Any]:
        """Analyze security implications of mutation"""
        
        # Basic security checks
        security_issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for common security issues
            if 'eval(' in content:
                security_issues.append('Use of eval() function')
            if 'exec(' in content:
                security_issues.append('Use of exec() function')
            if 'os.system(' in content:
                security_issues.append('Use of os.system()')
                
            return {
                'safe': len(security_issues) == 0,
                'issues': security_issues,
                'risk_level': len(security_issues) * 0.3
            }
            
        except Exception as e:
            return {
                'safe': False,
                'issues': [f'Error analyzing security: {e}'],
                'risk_level': 1.0
            }
            
    async def _create_backup(self, file_path: str) -> bool:
        """Create backup of file before mutation"""
        
        try:
            source = Path(file_path)
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            backup_name = f"{source.stem}_{timestamp}{source.suffix}"
            backup_file = self.backup_path / backup_name
            
            shutil.copy2(source, backup_file)
            
            # Clean old backups
            await self._cleanup_old_backups()
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return False
            
    async def _restore_from_backup(self, file_path: str) -> bool:
        """Restore file from most recent backup"""
        
        try:
            source = Path(file_path)
            backup_pattern = f"{source.stem}_*{source.suffix}"
            backups = list(self.backup_path.glob(backup_pattern))
            
            if not backups:
                return False
                
            # Get most recent backup
            latest_backup = max(backups, key=lambda p: p.stat().st_mtime)
            shutil.copy2(latest_backup, source)
            
            return True
            
        except Exception as e:
            logger.error(f"Error restoring from backup: {e}")
            return False
            
    async def _cleanup_old_backups(self):
        """Remove old backup files"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=self.backup_retention_days)
        
        for backup_file in self.backup_path.glob('*.py'):
            if datetime.fromtimestamp(backup_file.stat().st_mtime) < cutoff_date:
                backup_file.unlink()
                
    async def _calculate_quality_score(self, scan_results: Dict[str, Any]) -> float:
        """Calculate overall code quality score"""
        
        total_functions = scan_results.get('functions_analyzed', 1)
        total_weaknesses = len(scan_results.get('weaknesses', []))
        
        weakness_ratio = total_weaknesses / total_functions
        quality_score = max(0.0, 1.0 - weakness_ratio)
        
        return quality_score
        
    async def _calculate_complexity_metrics(self, scan_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate complexity metrics"""
        
        return {
            'average_function_length': scan_results.get('total_lines', 0) / max(scan_results.get('functions_analyzed', 1), 1),
            'functions_per_file': scan_results.get('functions_analyzed', 0) / max(scan_results.get('files_analyzed', 1), 1),
            'classes_per_file': scan_results.get('classes_analyzed', 0) / max(scan_results.get('files_analyzed', 1), 1)
        }
        
    def _update_success_rate(self):
        """Update mutation success rate"""
        
        if not self.applied_mutations:
            self.mutation_success_rate = 0.0
            return
            
        successful = sum(1 for m in self.applied_mutations 
                        if m.test_results and m.test_results.get('success', False))
        
        self.mutation_success_rate = successful / len(self.applied_mutations)
        
    async def get_evolution_statistics(self) -> Dict[str, Any]:
        """Get comprehensive evolution statistics"""
        
        return {
            'total_mutations_proposed': len(self.proposed_mutations),
            'total_mutations_applied': len(self.applied_mutations),
            'success_rate': self.mutation_success_rate,
            'active_hypotheses': len(self.hypotheses),
            'recent_activity': {
                'last_scan': datetime.utcnow().isoformat() if hasattr(self, 'last_scan') else None,
                'mutations_last_24h': len([m for m in self.applied_mutations 
                                         if m.applied_at and m.applied_at > datetime.utcnow() - timedelta(days=1)])
            },
            'mutation_types': {
                mt.value: len([m for m in self.applied_mutations if m.mutation_type == mt])
                for mt in MutationType
            }
        }
        
    async def shutdown(self):
        """Gracefully shutdown the evolution engine"""
        
        logger.info("Shutting down Code Evolution Engine...")
        
        # Save state
        evolution_state = await self.get_evolution_statistics()
        
        logger.info(f"Evolution Engine shutdown complete - Final statistics: {evolution_state}")