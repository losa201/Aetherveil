"""
Event system for inter-module communication in Chimera
Enables asynchronous, decoupled communication between modules
"""

import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass
from weakref import WeakSet

logger = logging.getLogger(__name__)

class EventType(Enum):
    """Types of events in the Chimera system"""
    
    # Core system events
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    SYSTEM_ERROR = "system.error"
    
    # Reasoning events
    REASONING_START = "reasoning.start"
    REASONING_COMPLETE = "reasoning.complete"
    DECISION_MADE = "reasoning.decision"
    
    # Knowledge events
    KNOWLEDGE_LEARNED = "knowledge.learned"
    KNOWLEDGE_UPDATED = "knowledge.updated"
    KNOWLEDGE_PRUNED = "knowledge.pruned"
    
    # Web events
    WEB_SEARCH_START = "web.search.start"
    WEB_SEARCH_COMPLETE = "web.search.complete"
    WEB_SCRAPE_COMPLETE = "web.scrape.complete"
    
    # LLM events
    LLM_QUERY_START = "llm.query.start"
    LLM_QUERY_COMPLETE = "llm.query.complete"
    LLM_ADVICE_RECEIVED = "llm.advice.received"
    
    # Planning events
    PLAN_CREATED = "plan.created"
    PLAN_UPDATED = "plan.updated"
    PLAN_EXECUTED = "plan.executed"
    
    # Execution events
    TASK_START = "task.start"
    TASK_COMPLETE = "task.complete"
    TASK_FAILED = "task.failed"
    
    # OPSEC events
    OPSEC_VIOLATION = "opsec.violation"
    STEALTH_MODE_CHANGE = "opsec.stealth.change"
    
    # Validation events
    VALIDATION_START = "validation.start"
    VALIDATION_COMPLETE = "validation.complete"
    VALIDATION_FAILED = "validation.failed"
    
    # Report events
    REPORT_GENERATED = "report.generated"
    FINDING_DISCOVERED = "report.finding"

@dataclass
class ChimeraEvent:
    """Event object containing event data"""
    
    event_type: EventType
    source: str
    data: Dict[str, Any]
    timestamp: datetime
    correlation_id: Optional[str] = None
    severity: str = "INFO"
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

class EventSystem:
    """
    Asynchronous event system for Chimera modules
    Handles event distribution, filtering, and lifecycle management
    """
    
    def __init__(self, max_queue_size: int = 1000):
        self.max_queue_size = max_queue_size
        self.event_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self.subscribers: Dict[EventType, WeakSet] = {}
        self.event_history: List[ChimeraEvent] = []
        self.max_history = 10000
        self.running = False
        self.processor_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start the event processing system"""
        if self.running:
            return
            
        self.running = True
        self.processor_task = asyncio.create_task(self._process_events())
        logger.info("Event system started")
        
    async def stop(self):
        """Stop the event processing system"""
        if not self.running:
            return
            
        self.running = False
        if self.processor_task:
            self.processor_task.cancel()
            try:
                await self.processor_task
            except asyncio.CancelledError:
                pass
                
        logger.info("Event system stopped")
        
    async def emit(self, event: ChimeraEvent):
        """Emit an event to the system"""
        try:
            await self.event_queue.put(event)
            logger.debug(f"Event emitted: {event.event_type.value} from {event.source}")
        except asyncio.QueueFull:
            logger.warning(f"Event queue full, dropping event: {event.event_type.value}")
            
    async def subscribe(self, event_type: EventType, callback: Callable[[ChimeraEvent], None]):
        """Subscribe to events of a specific type"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = WeakSet()
            
        self.subscribers[event_type].add(callback)
        logger.debug(f"Subscribed to event type: {event_type.value}")
        
    def unsubscribe(self, event_type: EventType, callback: Callable[[ChimeraEvent], None]):
        """Unsubscribe from events"""
        if event_type in self.subscribers:
            self.subscribers[event_type].discard(callback)
            
    async def _process_events(self):
        """Process events from the queue"""
        while self.running:
            try:
                # Wait for events with timeout to allow checking running flag
                event = await asyncio.wait_for(
                    self.event_queue.get(), 
                    timeout=1.0
                )
                
                # Add to history
                self._add_to_history(event)
                
                # Notify subscribers
                await self._notify_subscribers(event)
                
                # Mark task as done
                self.event_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")
                
    def _add_to_history(self, event: ChimeraEvent):
        """Add event to history with size management"""
        self.event_history.append(event)
        
        # Trim history if too large
        if len(self.event_history) > self.max_history:
            self.event_history = self.event_history[-self.max_history//2:]
            
    async def _notify_subscribers(self, event: ChimeraEvent):
        """Notify all subscribers of an event"""
        if event.event_type not in self.subscribers:
            return
            
        callbacks = list(self.subscribers[event.event_type])
        
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")
                
    def get_recent_events(self, event_type: Optional[EventType] = None, limit: int = 100) -> List[ChimeraEvent]:
        """Get recent events, optionally filtered by type"""
        events = self.event_history[-limit:]
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
            
        return events
        
    def get_event_stats(self) -> Dict[str, Any]:
        """Get statistics about event processing"""
        type_counts = {}
        for event in self.event_history:
            event_type = event.event_type.value
            type_counts[event_type] = type_counts.get(event_type, 0) + 1
            
        return {
            "total_events": len(self.event_history),
            "queue_size": self.event_queue.qsize(),
            "subscriber_count": sum(len(subs) for subs in self.subscribers.values()),
            "event_type_distribution": type_counts,
            "running": self.running
        }

class EventEmitter:
    """Mixin class for modules that emit events"""
    
    def __init__(self, event_system: EventSystem, module_name: str):
        self.event_system = event_system
        self.module_name = module_name
        
    async def emit_event(self, event_type: EventType, data: Dict[str, Any], 
                        severity: str = "INFO", correlation_id: Optional[str] = None):
        """Emit an event from this module"""
        event = ChimeraEvent(
            event_type=event_type,
            source=self.module_name,
            data=data,
            timestamp=datetime.utcnow(),
            severity=severity,
            correlation_id=correlation_id
        )
        
        await self.event_system.emit(event)

class EventSubscriber:
    """Mixin class for modules that subscribe to events"""
    
    def __init__(self, event_system: EventSystem):
        self.event_system = event_system
        self.subscriptions: List[tuple] = []
        
    async def subscribe_to_event(self, event_type: EventType, callback: Callable[[ChimeraEvent], None]):
        """Subscribe to an event type"""
        await self.event_system.subscribe(event_type, callback)
        self.subscriptions.append((event_type, callback))
        
    def unsubscribe_all(self):
        """Unsubscribe from all events"""
        for event_type, callback in self.subscriptions:
            self.event_system.unsubscribe(event_type, callback)
        self.subscriptions.clear()