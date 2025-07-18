"""
Unit tests for the Orchestrator module.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import json
from typing import List, Dict, Any

from modules.orchestrator import (
    OrchestratorModule,
    WorkflowEngine,
    TaskManager,
    Workflow,
    Task,
    WorkflowStatus,
    TaskStatus,
    ExecutionMode,
    DependencyResolver
)
from config.config import AetherVeilConfig


class TestWorkflow:
    """Test Workflow data class"""
    
    def test_workflow_creation(self):
        """Test Workflow creation with default values"""
        workflow = Workflow(
            name="test_workflow",
            description="Test workflow",
            tasks=[],
            execution_mode=ExecutionMode.SEQUENTIAL
        )
        
        assert workflow.name == "test_workflow"
        assert workflow.description == "Test workflow"
        assert workflow.tasks == []
        assert workflow.execution_mode == ExecutionMode.SEQUENTIAL
        assert workflow.status == WorkflowStatus.CREATED
        assert workflow.metadata == {}
    
    def test_workflow_with_custom_values(self):
        """Test Workflow creation with custom values"""
        tasks = [
            Task(name="task1", module="reconnaissance", parameters={}),
            Task(name="task2", module="scanning", parameters={})
        ]
        metadata = {"priority": "high", "timeout": 3600}
        
        workflow = Workflow(
            name="custom_workflow",
            description="Custom workflow",
            tasks=tasks,
            execution_mode=ExecutionMode.PARALLEL,
            metadata=metadata
        )
        
        assert workflow.name == "custom_workflow"
        assert workflow.tasks == tasks
        assert workflow.execution_mode == ExecutionMode.PARALLEL
        assert workflow.metadata == metadata


class TestTask:
    """Test Task data class"""
    
    def test_task_creation(self):
        """Test Task creation"""
        task = Task(
            name="test_task",
            module="reconnaissance",
            parameters={"target": "example.com"}
        )
        
        assert task.name == "test_task"
        assert task.module == "reconnaissance"
        assert task.parameters == {"target": "example.com"}
        assert task.status == TaskStatus.PENDING
        assert task.dependencies == []
        assert task.timeout == 300
        assert task.retry_count == 0
        assert task.max_retries == 3
    
    def test_task_with_dependencies(self):
        """Test Task creation with dependencies"""
        task = Task(
            name="scan_task",
            module="scanning",
            parameters={"target": "example.com"},
            dependencies=["recon_task"],
            timeout=600,
            max_retries=5
        )
        
        assert task.name == "scan_task"
        assert task.dependencies == ["recon_task"]
        assert task.timeout == 600
        assert task.max_retries == 5


class TestDependencyResolver:
    """Test DependencyResolver functionality"""
    
    @pytest.fixture
    def resolver(self):
        """Dependency resolver fixture"""
        return DependencyResolver()
    
    def test_resolve_dependencies_linear(self, resolver):
        """Test linear dependency resolution"""
        tasks = [
            Task(name="task1", module="reconnaissance", parameters={}),
            Task(name="task2", module="scanning", parameters={}, dependencies=["task1"]),
            Task(name="task3", module="exploitation", parameters={}, dependencies=["task2"])
        ]
        
        resolved = resolver.resolve_dependencies(tasks)
        
        assert len(resolved) == 3
        assert resolved[0].name == "task1"
        assert resolved[1].name == "task2"
        assert resolved[2].name == "task3"
    
    def test_resolve_dependencies_parallel(self, resolver):
        """Test parallel dependency resolution"""
        tasks = [
            Task(name="task1", module="reconnaissance", parameters={}),
            Task(name="task2", module="osint", parameters={}),
            Task(name="task3", module="scanning", parameters={}, dependencies=["task1", "task2"])
        ]
        
        resolved = resolver.resolve_dependencies(tasks)
        
        assert len(resolved) == 3
        # task1 and task2 should be first (in any order)
        first_two = [resolved[0].name, resolved[1].name]
        assert "task1" in first_two
        assert "task2" in first_two
        # task3 should be last
        assert resolved[2].name == "task3"
    
    def test_resolve_dependencies_circular(self, resolver):
        """Test circular dependency detection"""
        tasks = [
            Task(name="task1", module="reconnaissance", parameters={}, dependencies=["task2"]),
            Task(name="task2", module="scanning", parameters={}, dependencies=["task1"])
        ]
        
        with pytest.raises(ValueError, match="Circular dependency detected"):
            resolver.resolve_dependencies(tasks)
    
    def test_resolve_dependencies_missing(self, resolver):
        """Test missing dependency detection"""
        tasks = [
            Task(name="task1", module="reconnaissance", parameters={}),
            Task(name="task2", module="scanning", parameters={}, dependencies=["nonexistent"])
        ]
        
        with pytest.raises(ValueError, match="Missing dependency"):
            resolver.resolve_dependencies(tasks)


class TestTaskManager:
    """Test TaskManager functionality"""
    
    @pytest.fixture
    def task_manager(self):
        """Task manager fixture"""
        return TaskManager()
    
    @pytest.mark.asyncio
    async def test_execute_task_success(self, task_manager):
        """Test successful task execution"""
        task = Task(
            name="test_task",
            module="reconnaissance",
            parameters={"target": "example.com"}
        )
        
        # Mock module execution
        with patch.object(task_manager, '_get_module') as mock_get_module:
            mock_module = Mock()
            mock_module.execute = AsyncMock(return_value={"status": "success", "data": {}})
            mock_get_module.return_value = mock_module
            
            result = await task_manager.execute_task(task)
            
            assert result["status"] == "success"
            assert task.status == TaskStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_execute_task_failure(self, task_manager):
        """Test task execution failure"""
        task = Task(
            name="test_task",
            module="reconnaissance",
            parameters={"target": "example.com"}
        )
        
        # Mock module execution failure
        with patch.object(task_manager, '_get_module') as mock_get_module:
            mock_module = Mock()
            mock_module.execute = AsyncMock(side_effect=Exception("Module error"))
            mock_get_module.return_value = mock_module
            
            result = await task_manager.execute_task(task)
            
            assert result["status"] == "error"
            assert task.status == TaskStatus.FAILED
    
    @pytest.mark.asyncio
    async def test_execute_task_timeout(self, task_manager):
        """Test task execution timeout"""
        task = Task(
            name="test_task",
            module="reconnaissance",
            parameters={"target": "example.com"},
            timeout=1  # Very short timeout
        )
        
        # Mock slow module execution
        with patch.object(task_manager, '_get_module') as mock_get_module:
            mock_module = Mock()
            mock_module.execute = AsyncMock(side_effect=asyncio.TimeoutError())
            mock_get_module.return_value = mock_module
            
            result = await task_manager.execute_task(task)
            
            assert result["status"] == "timeout"
            assert task.status == TaskStatus.FAILED
    
    @pytest.mark.asyncio
    async def test_execute_task_retry(self, task_manager):
        """Test task retry mechanism"""
        task = Task(
            name="test_task",
            module="reconnaissance",
            parameters={"target": "example.com"},
            max_retries=2
        )
        
        # Mock module execution with failure then success
        with patch.object(task_manager, '_get_module') as mock_get_module:
            mock_module = Mock()
            mock_module.execute = AsyncMock(side_effect=[
                Exception("First failure"),
                {"status": "success", "data": {}}
            ])
            mock_get_module.return_value = mock_module
            
            result = await task_manager.execute_task(task)
            
            assert result["status"] == "success"
            assert task.status == TaskStatus.COMPLETED
            assert task.retry_count == 1
    
    @pytest.mark.asyncio
    async def test_execute_tasks_sequential(self, task_manager):
        """Test sequential task execution"""
        tasks = [
            Task(name="task1", module="reconnaissance", parameters={}),
            Task(name="task2", module="scanning", parameters={})
        ]
        
        # Mock module execution
        with patch.object(task_manager, '_get_module') as mock_get_module:
            mock_module = Mock()
            mock_module.execute = AsyncMock(return_value={"status": "success", "data": {}})
            mock_get_module.return_value = mock_module
            
            results = await task_manager.execute_tasks(tasks, ExecutionMode.SEQUENTIAL)
            
            assert len(results) == 2
            assert all(r["status"] == "success" for r in results)
    
    @pytest.mark.asyncio
    async def test_execute_tasks_parallel(self, task_manager):
        """Test parallel task execution"""
        tasks = [
            Task(name="task1", module="reconnaissance", parameters={}),
            Task(name="task2", module="osint", parameters={}),
            Task(name="task3", module="scanning", parameters={})
        ]
        
        # Mock module execution
        with patch.object(task_manager, '_get_module') as mock_get_module:
            mock_module = Mock()
            mock_module.execute = AsyncMock(return_value={"status": "success", "data": {}})
            mock_get_module.return_value = mock_module
            
            results = await task_manager.execute_tasks(tasks, ExecutionMode.PARALLEL)
            
            assert len(results) == 3
            assert all(r["status"] == "success" for r in results)


class TestWorkflowEngine:
    """Test WorkflowEngine functionality"""
    
    @pytest.fixture
    def workflow_engine(self):
        """Workflow engine fixture"""
        return WorkflowEngine()
    
    @pytest.mark.asyncio
    async def test_create_workflow(self, workflow_engine):
        """Test workflow creation"""
        workflow_def = {
            "name": "test_workflow",
            "description": "Test workflow",
            "tasks": [
                {
                    "name": "recon_task",
                    "module": "reconnaissance",
                    "parameters": {"target": "example.com"}
                }
            ],
            "execution_mode": "sequential"
        }
        
        workflow_id = await workflow_engine.create_workflow(workflow_def)
        
        assert workflow_id is not None
        assert workflow_id in workflow_engine.workflows
        
        workflow = workflow_engine.workflows[workflow_id]
        assert workflow.name == "test_workflow"
        assert len(workflow.tasks) == 1
        assert workflow.tasks[0].name == "recon_task"
    
    @pytest.mark.asyncio
    async def test_execute_workflow(self, workflow_engine):
        """Test workflow execution"""
        workflow_def = {
            "name": "test_workflow",
            "description": "Test workflow",
            "tasks": [
                {
                    "name": "recon_task",
                    "module": "reconnaissance",
                    "parameters": {"target": "example.com"}
                }
            ],
            "execution_mode": "sequential"
        }
        
        workflow_id = await workflow_engine.create_workflow(workflow_def)
        
        # Mock task manager execution
        with patch.object(workflow_engine.task_manager, 'execute_tasks') as mock_execute:
            mock_execute.return_value = [{"status": "success", "data": {}}]
            
            result = await workflow_engine.execute_workflow(workflow_id)
            
            assert result["status"] == "completed"
            assert workflow_engine.workflows[workflow_id].status == WorkflowStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_execute_workflow_with_dependencies(self, workflow_engine):
        """Test workflow execution with task dependencies"""
        workflow_def = {
            "name": "dependency_workflow",
            "description": "Workflow with dependencies",
            "tasks": [
                {
                    "name": "recon_task",
                    "module": "reconnaissance",
                    "parameters": {"target": "example.com"}
                },
                {
                    "name": "scan_task",
                    "module": "scanning",
                    "parameters": {"target": "example.com"},
                    "dependencies": ["recon_task"]
                }
            ],
            "execution_mode": "sequential"
        }
        
        workflow_id = await workflow_engine.create_workflow(workflow_def)
        
        # Mock task manager execution
        with patch.object(workflow_engine.task_manager, 'execute_tasks') as mock_execute:
            mock_execute.return_value = [
                {"status": "success", "data": {}},
                {"status": "success", "data": {}}
            ]
            
            result = await workflow_engine.execute_workflow(workflow_id)
            
            assert result["status"] == "completed"
            
            # Verify dependency resolution
            workflow = workflow_engine.workflows[workflow_id]
            assert len(workflow.tasks) == 2
    
    @pytest.mark.asyncio
    async def test_get_workflow_status(self, workflow_engine):
        """Test workflow status retrieval"""
        workflow_def = {
            "name": "status_workflow",
            "description": "Status test workflow",
            "tasks": [
                {
                    "name": "test_task",
                    "module": "reconnaissance",
                    "parameters": {"target": "example.com"}
                }
            ],
            "execution_mode": "sequential"
        }
        
        workflow_id = await workflow_engine.create_workflow(workflow_def)
        
        status = await workflow_engine.get_workflow_status(workflow_id)
        
        assert status["workflow_id"] == workflow_id
        assert status["name"] == "status_workflow"
        assert status["status"] == "created"
        assert status["progress"] == 0.0
        assert status["total_tasks"] == 1
        assert status["completed_tasks"] == 0
    
    @pytest.mark.asyncio
    async def test_cancel_workflow(self, workflow_engine):
        """Test workflow cancellation"""
        workflow_def = {
            "name": "cancel_workflow",
            "description": "Cancellation test workflow",
            "tasks": [
                {
                    "name": "test_task",
                    "module": "reconnaissance",
                    "parameters": {"target": "example.com"}
                }
            ],
            "execution_mode": "sequential"
        }
        
        workflow_id = await workflow_engine.create_workflow(workflow_def)
        
        # Start workflow execution
        workflow_engine.workflows[workflow_id].status = WorkflowStatus.RUNNING
        
        success = await workflow_engine.cancel_workflow(workflow_id)
        
        assert success is True
        assert workflow_engine.workflows[workflow_id].status == WorkflowStatus.CANCELLED
    
    def test_list_workflows(self, workflow_engine):
        """Test listing workflows"""
        # Create test workflows
        workflow1 = Workflow(
            name="workflow1",
            description="First workflow",
            tasks=[],
            execution_mode=ExecutionMode.SEQUENTIAL
        )
        workflow2 = Workflow(
            name="workflow2",
            description="Second workflow",
            tasks=[],
            execution_mode=ExecutionMode.PARALLEL
        )
        
        workflow_engine.workflows["id1"] = workflow1
        workflow_engine.workflows["id2"] = workflow2
        
        workflows = workflow_engine.list_workflows()
        
        assert len(workflows) == 2
        assert "id1" in workflows
        assert "id2" in workflows
        assert workflows["id1"]["name"] == "workflow1"
        assert workflows["id2"]["name"] == "workflow2"
    
    def test_get_workflow_templates(self, workflow_engine):
        """Test getting workflow templates"""
        templates = workflow_engine.get_workflow_templates()
        
        assert isinstance(templates, list)
        assert len(templates) > 0
        
        # Check template structure
        for template in templates:
            assert "name" in template
            assert "description" in template
            assert "tasks" in template
            assert "execution_mode" in template


class TestOrchestratorModule:
    """Test main orchestrator module"""
    
    @pytest.fixture
    def config(self):
        """Configuration fixture"""
        return AetherVeilConfig()
    
    @pytest.fixture
    def orchestrator_module(self, config):
        """Orchestrator module fixture"""
        return OrchestratorModule(config)
    
    @pytest.mark.asyncio
    async def test_module_initialization(self, orchestrator_module):
        """Test module initialization"""
        assert orchestrator_module.module_type.value == "orchestrator"
        assert orchestrator_module.status.value == "initialized"
        assert orchestrator_module.version == "1.0.0"
        assert orchestrator_module.workflow_engine is not None
    
    @pytest.mark.asyncio
    async def test_module_start_stop(self, orchestrator_module):
        """Test module start and stop"""
        # Test start
        success = await orchestrator_module.start()
        assert success is True
        assert orchestrator_module.status.value == "running"
        
        # Test stop
        success = await orchestrator_module.stop()
        assert success is True
        assert orchestrator_module.status.value == "stopped"
    
    @pytest.mark.asyncio
    async def test_create_workflow(self, orchestrator_module):
        """Test workflow creation through module"""
        workflow_def = {
            "name": "module_workflow",
            "description": "Module test workflow",
            "tasks": [
                {
                    "name": "test_task",
                    "module": "reconnaissance",
                    "parameters": {"target": "example.com"}
                }
            ],
            "execution_mode": "sequential"
        }
        
        workflow_id = await orchestrator_module.create_workflow(workflow_def)
        
        assert workflow_id is not None
        assert workflow_id in orchestrator_module.workflow_engine.workflows
    
    @pytest.mark.asyncio
    async def test_execute_workflow(self, orchestrator_module):
        """Test workflow execution through module"""
        workflow_def = {
            "name": "execution_workflow",
            "description": "Execution test workflow",
            "tasks": [
                {
                    "name": "test_task",
                    "module": "reconnaissance",
                    "parameters": {"target": "example.com"}
                }
            ],
            "execution_mode": "sequential"
        }
        
        workflow_id = await orchestrator_module.create_workflow(workflow_def)
        
        # Mock workflow execution
        with patch.object(orchestrator_module.workflow_engine, 'execute_workflow') as mock_execute:
            mock_execute.return_value = {"status": "completed", "duration": 30.0}
            
            result = await orchestrator_module.execute_workflow(workflow_id)
            
            assert result["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_get_workflow_status(self, orchestrator_module):
        """Test workflow status retrieval through module"""
        workflow_def = {
            "name": "status_workflow",
            "description": "Status test workflow",
            "tasks": [],
            "execution_mode": "sequential"
        }
        
        workflow_id = await orchestrator_module.create_workflow(workflow_def)
        
        status = await orchestrator_module.get_workflow_status(workflow_id)
        
        assert status["workflow_id"] == workflow_id
        assert status["name"] == "status_workflow"
        assert "status" in status
        assert "progress" in status
    
    @pytest.mark.asyncio
    async def test_get_execution_history(self, orchestrator_module):
        """Test execution history retrieval"""
        history = await orchestrator_module.get_execution_history()
        
        assert isinstance(history, list)
        # History might be empty in new instance
    
    @pytest.mark.asyncio
    async def test_get_status(self, orchestrator_module):
        """Test module status reporting"""
        status = await orchestrator_module.get_status()
        
        assert status["module"] == "orchestrator"
        assert status["status"] == "initialized"
        assert status["version"] == "1.0.0"
        assert "active_workflows" in status
        assert "completed_workflows" in status
        assert "failed_workflows" in status
    
    def test_get_templates(self, orchestrator_module):
        """Test getting workflow templates"""
        templates = orchestrator_module.get_templates()
        
        assert isinstance(templates, list)
        assert len(templates) > 0
    
    def test_export_workflow(self, orchestrator_module):
        """Test workflow export"""
        workflow = Workflow(
            name="export_workflow",
            description="Export test workflow",
            tasks=[
                Task(name="test_task", module="reconnaissance", parameters={})
            ],
            execution_mode=ExecutionMode.SEQUENTIAL
        )
        
        workflow_id = "export-test-123"
        orchestrator_module.workflow_engine.workflows[workflow_id] = workflow
        
        exported = orchestrator_module.export_workflow(workflow_id)
        
        assert exported["name"] == "export_workflow"
        assert exported["description"] == "Export test workflow"
        assert len(exported["tasks"]) == 1
        assert exported["execution_mode"] == "sequential"


@pytest.mark.performance
class TestOrchestratorPerformance:
    """Performance tests for orchestrator module"""
    
    @pytest.fixture
    def config(self):
        """Configuration fixture"""
        return AetherVeilConfig()
    
    @pytest.fixture
    def orchestrator_module(self, config):
        """Orchestrator module fixture"""
        return OrchestratorModule(config)
    
    @pytest.mark.asyncio
    async def test_large_workflow_execution(self, orchestrator_module, performance_monitor):
        """Test large workflow execution performance"""
        # Create a workflow with many tasks
        tasks = []
        for i in range(20):
            tasks.append({
                "name": f"task_{i}",
                "module": "reconnaissance",
                "parameters": {"target": f"example{i}.com"}
            })
        
        workflow_def = {
            "name": "large_workflow",
            "description": "Large workflow test",
            "tasks": tasks,
            "execution_mode": "parallel"
        }
        
        workflow_id = await orchestrator_module.create_workflow(workflow_def)
        
        # Mock fast task execution
        with patch.object(orchestrator_module.workflow_engine.task_manager, 'execute_tasks') as mock_execute:
            mock_execute.return_value = [{"status": "success", "data": {}} for _ in range(20)]
            
            performance_monitor.start()
            
            result = await orchestrator_module.execute_workflow(workflow_id)
            
            performance_monitor.stop()
            
            duration = performance_monitor.get_duration()
            assert duration is not None
            assert duration < 30.0  # Should complete within 30 seconds
            assert result["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_concurrent_workflow_execution(self, orchestrator_module, performance_monitor):
        """Test concurrent workflow execution performance"""
        # Create multiple workflows
        workflow_ids = []
        for i in range(5):
            workflow_def = {
                "name": f"workflow_{i}",
                "description": f"Workflow {i}",
                "tasks": [
                    {
                        "name": f"task_{i}",
                        "module": "reconnaissance",
                        "parameters": {"target": f"example{i}.com"}
                    }
                ],
                "execution_mode": "sequential"
            }
            workflow_id = await orchestrator_module.create_workflow(workflow_def)
            workflow_ids.append(workflow_id)
        
        # Mock task execution
        with patch.object(orchestrator_module.workflow_engine.task_manager, 'execute_tasks') as mock_execute:
            mock_execute.return_value = [{"status": "success", "data": {}}]
            
            performance_monitor.start()
            
            # Execute workflows concurrently
            tasks = [orchestrator_module.execute_workflow(wid) for wid in workflow_ids]
            results = await asyncio.gather(*tasks)
            
            performance_monitor.stop()
            
            duration = performance_monitor.get_duration()
            assert duration is not None
            assert duration < 10.0  # Should complete within 10 seconds
            assert len(results) == 5
            assert all(r["status"] == "completed" for r in results)


@pytest.mark.security
class TestOrchestratorSecurity:
    """Security tests for orchestrator module"""
    
    @pytest.fixture
    def config(self):
        """Configuration fixture"""
        return AetherVeilConfig()
    
    @pytest.fixture
    def orchestrator_module(self, config):
        """Orchestrator module fixture"""
        return OrchestratorModule(config)
    
    @pytest.mark.asyncio
    async def test_workflow_input_validation(self, orchestrator_module):
        """Test workflow input validation"""
        # Test malicious workflow definitions
        malicious_workflows = [
            {
                "name": "; rm -rf /",
                "description": "Malicious workflow",
                "tasks": [],
                "execution_mode": "sequential"
            },
            {
                "name": "valid_name",
                "description": "Valid description",
                "tasks": [
                    {
                        "name": "$(whoami)",
                        "module": "reconnaissance",
                        "parameters": {"target": "example.com"}
                    }
                ],
                "execution_mode": "sequential"
            }
        ]
        
        for workflow_def in malicious_workflows:
            # Should handle malicious input gracefully
            workflow_id = await orchestrator_module.create_workflow(workflow_def)
            assert workflow_id is not None
    
    @pytest.mark.asyncio
    async def test_task_parameter_validation(self, orchestrator_module):
        """Test task parameter validation"""
        workflow_def = {
            "name": "validation_workflow",
            "description": "Parameter validation test",
            "tasks": [
                {
                    "name": "test_task",
                    "module": "reconnaissance",
                    "parameters": {
                        "target": "'; DROP TABLE users; --",
                        "malicious_param": "../../../etc/passwd"
                    }
                }
            ],
            "execution_mode": "sequential"
        }
        
        workflow_id = await orchestrator_module.create_workflow(workflow_def)
        
        # Should handle malicious parameters gracefully
        assert workflow_id is not None
    
    @pytest.mark.asyncio
    async def test_resource_limit_enforcement(self, orchestrator_module):
        """Test resource limit enforcement"""
        # Test workflow with excessive resource requirements
        tasks = []
        for i in range(1000):  # Many tasks
            tasks.append({
                "name": f"task_{i}",
                "module": "reconnaissance",
                "parameters": {"target": f"example{i}.com"}
            })
        
        workflow_def = {
            "name": "resource_intensive_workflow",
            "description": "Resource intensive workflow",
            "tasks": tasks,
            "execution_mode": "parallel"
        }
        
        workflow_id = await orchestrator_module.create_workflow(workflow_def)
        
        # Should handle resource-intensive workflows gracefully
        assert workflow_id is not None
    
    @pytest.mark.asyncio
    async def test_workflow_isolation(self, orchestrator_module):
        """Test workflow isolation"""
        # Create two workflows
        workflow1_def = {
            "name": "workflow1",
            "description": "First workflow",
            "tasks": [
                {
                    "name": "task1",
                    "module": "reconnaissance",
                    "parameters": {"target": "example1.com"}
                }
            ],
            "execution_mode": "sequential"
        }
        
        workflow2_def = {
            "name": "workflow2",
            "description": "Second workflow",
            "tasks": [
                {
                    "name": "task2",
                    "module": "reconnaissance",
                    "parameters": {"target": "example2.com"}
                }
            ],
            "execution_mode": "sequential"
        }
        
        workflow1_id = await orchestrator_module.create_workflow(workflow1_def)
        workflow2_id = await orchestrator_module.create_workflow(workflow2_def)
        
        # Workflows should be isolated
        assert workflow1_id != workflow2_id
        assert workflow1_id in orchestrator_module.workflow_engine.workflows
        assert workflow2_id in orchestrator_module.workflow_engine.workflows
        
        # Modifying one should not affect the other
        workflow1 = orchestrator_module.workflow_engine.workflows[workflow1_id]
        workflow2 = orchestrator_module.workflow_engine.workflows[workflow2_id]
        
        workflow1.status = WorkflowStatus.RUNNING
        assert workflow2.status == WorkflowStatus.CREATED