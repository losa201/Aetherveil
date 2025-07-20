"""
Aether Event System: Advanced event-driven architecture for agent coordination
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Set, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import weakref
from collections import defaultdict, deque
import uuid
import inspect

logger = logging.getLogger(__name__)

class EventPriority(Enum):
    """Event priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5

class EventType(Enum):
    """Core event types in the Aether system"""
    
    # Cognitive events
    CONSCIOUSNESS_CHANGE = "consciousness_change"
    INTROSPECTION_COMPLETE = "introspection_complete"
    CURIOSITY_TRIGGERED = "curiosity_triggered"
    LEARNING_GOAL_SET = "learning_goal_set"
    
    # Code evolution events
    CODE_SCAN_COMPLETE = "code_scan_complete"
    MUTATION_PROPOSED = "mutation_proposed"
    MUTATION_TESTED = "mutation_tested"
    MUTATION_APPLIED = "mutation_applied"
    HYPOTHESIS_GENERATED = "hypothesis_generated"
    
    # LLM interaction events
    LLM_QUERY_START = "llm_query_start"
    LLM_RESPONSE_RECEIVED = "llm_response_received"
    RAPPORT_UPDATED = "rapport_updated"
    CONVERSATION_STARTED = "conversation_started"
    CONVERSATION_ENDED = "conversation_ended"
    
    # Knowledge events
    INSIGHT_STORED = "insight_stored"
    KNOWLEDGE_SYNTHESIZED = "knowledge_synthesized"
    MEMORY_CONSOLIDATED = "memory_consolidated"
    KNOWLEDGE_GAP_IDENTIFIED = "knowledge_gap_identified"
    
    # Identity and stealth events
    IDENTITY_CREATED = "identity_created"
    IDENTITY_ROTATED = "identity_rotated"
    STEALTH_COMPROMISED = "stealth_compromised"
    BROWSER_SESSION_START = "browser_session_start"
    BROWSER_SESSION_END = "browser_session_end"
    
    # System events
    AGENT_STARTUP = "agent_startup"
    AGENT_SHUTDOWN = "agent_shutdown"
    MODULE_INITIALIZED = "module_initialized"
    ERROR_OCCURRED = "error_occurred"
    PERFORMANCE_ALERT = "performance_alert"
    
    # Learning cycle events
    LEARNING_CYCLE_START = "learning_cycle_start"
    LEARNING_CYCLE_COMPLETE = "learning_cycle_complete"
    SKILL_ACQUIRED = "skill_acquired"
    MASTERY_ACHIEVED = "mastery_achieved"

@dataclass
class Event:
    """Represents an event in the system"""
    
    event_id: str
    event_type: EventType
    source: str
    timestamp: datetime
    priority: EventPriority
    data: Dict[str, Any]
    correlation_id: Optional[str] = None
    requires_response: bool = False
    response_timeout: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = str(uuid.uuid4())[:8]

@dataclass
class EventSubscription:
    """Represents an event subscription"""
    
    subscription_id: str
    event_types: Set[EventType]
    handler: Callable
    priority: int
    filter_func: Optional[Callable] = None
    once_only: bool = False
    max_events: Optional[int] = None
    events_handled: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_triggered: Optional[datetime] = None

@dataclass
class EventMetrics:
    """Event system performance metrics"""
    
    total_events_published: int = 0
    total_events_handled: int = 0
    events_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    events_by_priority: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    average_handling_time: float = 0.0
    failed_events: int = 0
    pending_events: int = 0
    active_subscriptions: int = 0

class EventBus:
    """
    Advanced event bus with priority queues, filtering, and metrics
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Event storage and queues
        self.event_queues: Dict[EventPriority, asyncio.Queue] = {
            priority: asyncio.Queue() for priority in EventPriority
        }
        
        # Subscriptions
        self.subscriptions: Dict[str, EventSubscription] = {}
        self.event_handlers: Dict[EventType, List[EventSubscription]] = defaultdict(list)
        
        # Event history and correlation
        self.event_history: deque = deque(maxlen=config.get('max_event_history', 1000))
        self.correlation_chains: Dict[str, List[str]] = defaultdict(list)
        
        # Performance and monitoring
        self.metrics = EventMetrics()
        self.event_timings: deque = deque(maxlen=100)
        self.failed_events: deque = deque(maxlen=50)
        
        # Control flags
        self.running = False
        self.workers: List[asyncio.Task] = []
        self.max_workers = config.get('max_event_workers', 5)
        
        # Debugging and logging
        self.debug_mode = config.get('debug_events', False)
        self.log_all_events = config.get('log_all_events', False)
        
    async def start(self):
        """Start the event system"""
        
        if self.running:
            return
            
        logger.info("Starting Aether Event System...")
        
        self.running = True
        
        # Start event processing workers
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._event_worker(f"worker_{i}"))
            self.workers.append(worker)
            
        # Start metrics collection
        asyncio.create_task(self._metrics_collector())
        
        # Emit startup event
        await self.publish(Event(
            event_id="system_startup",
            event_type=EventType.AGENT_STARTUP,
            source="event_system",
            timestamp=datetime.utcnow(),
            priority=EventPriority.HIGH,
            data={"workers": self.max_workers}
        ))
        
        logger.info(f"Event system started with {self.max_workers} workers")
        
    async def stop(self):
        """Stop the event system"""
        
        if not self.running:
            return
            
        logger.info("Stopping Aether Event System...")
        
        # Emit shutdown event
        await self.publish(Event(
            event_id="system_shutdown",
            event_type=EventType.AGENT_SHUTDOWN,
            source="event_system",
            timestamp=datetime.utcnow(),
            priority=EventPriority.CRITICAL,
            data={"metrics": asdict(self.metrics)}
        ))
        
        self.running = False
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
            
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        
        logger.info("Event system stopped")
        
    async def publish(self, event: Event):
        """Publish an event to the system"""
        
        if self.log_all_events or self.debug_mode:
            logger.debug(f"Publishing event: {event.event_type.value} from {event.source}")
            
        # Add to history
        self.event_history.append(event)
        
        # Handle correlation
        if event.correlation_id:
            self.correlation_chains[event.correlation_id].append(event.event_id)
            
        # Update metrics
        self.metrics.total_events_published += 1
        self.metrics.events_by_type[event.event_type.value] += 1
        self.metrics.events_by_priority[event.priority.name] += 1
        
        # Add to appropriate priority queue
        await self.event_queues[event.priority].put(event)
        
        if self.debug_mode:
            logger.debug(f"Event {event.event_id} queued with priority {event.priority.name}")
            
    async def subscribe(self, 
                       event_types: Union[EventType, List[EventType]], 
                       handler: Callable,
                       priority: int = 0,
                       filter_func: Optional[Callable] = None,
                       once_only: bool = False,
                       max_events: Optional[int] = None) -> str:
        """Subscribe to events"""
        
        # Normalize event_types to set
        if isinstance(event_types, EventType):
            event_types_set = {event_types}
        else:
            event_types_set = set(event_types)
            
        # Generate subscription ID
        subscription_id = f"sub_{len(self.subscriptions)}_{int(time.time())}"
        
        # Create subscription
        subscription = EventSubscription(
            subscription_id=subscription_id,
            event_types=event_types_set,
            handler=handler,
            priority=priority,
            filter_func=filter_func,
            once_only=once_only,
            max_events=max_events
        )
        
        # Store subscription
        self.subscriptions[subscription_id] = subscription
        
        # Add to event type mappings
        for event_type in event_types_set:
            self.event_handlers[event_type].append(subscription)
            # Sort by priority (higher priority first)
            self.event_handlers[event_type].sort(key=lambda s: s.priority, reverse=True)
            
        self.metrics.active_subscriptions += 1
        
        logger.debug(f"Created subscription {subscription_id} for events: {[et.value for et in event_types_set]}")
        
        return subscription_id
        
    async def unsubscribe(self, subscription_id: str):
        """Unsubscribe from events"""
        
        if subscription_id not in self.subscriptions:
            logger.warning(f"Subscription {subscription_id} not found")
            return
            
        subscription = self.subscriptions[subscription_id]
        
        # Remove from event type mappings
        for event_type in subscription.event_types:
            if event_type in self.event_handlers:
                self.event_handlers[event_type] = [
                    s for s in self.event_handlers[event_type] 
                    if s.subscription_id != subscription_id
                ]
                
        # Remove subscription
        del self.subscriptions[subscription_id]
        self.metrics.active_subscriptions -= 1
        
        logger.debug(f"Removed subscription {subscription_id}")
        
    async def publish_and_wait(self, event: Event, timeout: float = 30.0) -> List[Any]:
        """Publish event and wait for all handlers to complete"""
        
        # Set response requirement
        event.requires_response = True
        event.response_timeout = timeout
        
        # Create future for collecting responses
        response_future = asyncio.Future()
        event.metadata['response_future'] = response_future
        
        # Publish the event
        await self.publish(event)
        
        try:
            # Wait for responses
            responses = await asyncio.wait_for(response_future, timeout=timeout)
            return responses
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for responses to event {event.event_id}")
            return []
            
    async def emit_consciousness_change(self, source: str, old_level: float, new_level: float):
        """Emit consciousness level change event"""
        
        await self.publish(Event(
            event_id=f"consciousness_{int(time.time())}",
            event_type=EventType.CONSCIOUSNESS_CHANGE,
            source=source,
            timestamp=datetime.utcnow(),
            priority=EventPriority.HIGH,
            data={
                'old_level': old_level,
                'new_level': new_level,
                'change': new_level - old_level
            }
        ))
        
    async def emit_learning_milestone(self, source: str, milestone_type: str, details: Dict[str, Any]):
        """Emit learning milestone achievement"""
        
        event_type = EventType.SKILL_ACQUIRED if milestone_type == 'skill' else EventType.MASTERY_ACHIEVED
        
        await self.publish(Event(
            event_id=f"milestone_{milestone_type}_{int(time.time())}",
            event_type=event_type,
            source=source,
            timestamp=datetime.utcnow(),
            priority=EventPriority.HIGH,
            data={
                'milestone_type': milestone_type,
                'details': details
            }
        ))
        
    async def emit_error(self, source: str, error: Exception, context: Dict[str, Any] = None):
        """Emit error event"""
        
        await self.publish(Event(
            event_id=f"error_{int(time.time())}",
            event_type=EventType.ERROR_OCCURRED,
            source=source,
            timestamp=datetime.utcnow(),
            priority=EventPriority.CRITICAL,
            data={
                'error_type': type(error).__name__,
                'error_message': str(error),
                'context': context or {}
            }
        ))
        
    # Private methods
    
    async def _event_worker(self, worker_id: str):
        """Worker coroutine for processing events"""
        
        logger.debug(f"Event worker {worker_id} started")
        
        while self.running:
            try:
                # Check queues in priority order
                event = None
                
                for priority in [EventPriority.EMERGENCY, EventPriority.CRITICAL, 
                               EventPriority.HIGH, EventPriority.NORMAL, EventPriority.LOW]:
                    try:
                        event = self.event_queues[priority].get_nowait()
                        break
                    except asyncio.QueueEmpty:
                        continue
                        
                if event is None:
                    # No events available, wait a bit
                    await asyncio.sleep(0.1)
                    continue
                    
                # Process the event
                await self._handle_event(event, worker_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in event worker {worker_id}: {e}")
                await asyncio.sleep(1)  # Prevent tight error loop
                
        logger.debug(f"Event worker {worker_id} stopped")
        
    async def _handle_event(self, event: Event, worker_id: str):
        """Handle a single event"""
        
        start_time = time.time()
        responses = []
        handlers_called = 0
        
        try:
            # Get handlers for this event type
            handlers = self.event_handlers.get(event.event_type, [])
            
            if self.debug_mode and handlers:
                logger.debug(f"Processing event {event.event_id} with {len(handlers)} handlers")
                
            for subscription in handlers:
                try:
                    # Check if subscription is still valid
                    if subscription.once_only and subscription.events_handled > 0:
                        continue
                        
                    if subscription.max_events and subscription.events_handled >= subscription.max_events:
                        continue
                        
                    # Apply filter if present
                    if subscription.filter_func and not subscription.filter_func(event):
                        continue
                        
                    # Call the handler
                    handler_start = time.time()
                    
                    if inspect.iscoroutinefunction(subscription.handler):
                        result = await subscription.handler(event)
                    else:
                        result = subscription.handler(event)
                        
                    handler_time = time.time() - handler_start
                    
                    # Update subscription stats
                    subscription.events_handled += 1
                    subscription.last_triggered = datetime.utcnow()
                    
                    handlers_called += 1
                    
                    # Collect response if needed
                    if event.requires_response and result is not None:
                        responses.append(result)
                        
                    # Remove subscription if it's once-only
                    if subscription.once_only:
                        await self.unsubscribe(subscription.subscription_id)
                        
                    if self.debug_mode:
                        logger.debug(f"Handler {subscription.subscription_id} processed event in {handler_time:.3f}s")
                        
                except Exception as e:
                    logger.error(f"Error in event handler {subscription.subscription_id}: {e}")
                    self.failed_events.append({
                        'event_id': event.event_id,
                        'subscription_id': subscription.subscription_id,
                        'error': str(e),
                        'timestamp': datetime.utcnow()
                    })
                    self.metrics.failed_events += 1
                    
            # Update metrics
            processing_time = time.time() - start_time
            self.event_timings.append(processing_time)
            self.metrics.total_events_handled += 1
            
            # Update average handling time
            if self.event_timings:
                self.metrics.average_handling_time = sum(self.event_timings) / len(self.event_timings)
                
            # Send responses if required
            if event.requires_response and 'response_future' in event.metadata:
                future = event.metadata['response_future']
                if not future.done():
                    future.set_result(responses)
                    
            if self.debug_mode:
                logger.debug(f"Event {event.event_id} processed by {handlers_called} handlers in {processing_time:.3f}s")
                
        except Exception as e:
            logger.error(f"Critical error handling event {event.event_id}: {e}")
            
            # Set exception on response future if present
            if event.requires_response and 'response_future' in event.metadata:
                future = event.metadata['response_future']
                if not future.done():
                    future.set_exception(e)
                    
    async def _metrics_collector(self):
        """Background task for collecting metrics"""
        
        while self.running:
            try:
                # Update pending events count
                self.metrics.pending_events = sum(
                    queue.qsize() for queue in self.event_queues.values()
                )
                
                # Clean up old correlation chains
                cutoff = datetime.utcnow() - timedelta(hours=1)
                for correlation_id in list(self.correlation_chains.keys()):
                    # Remove old chains (this is simplified, would need timestamp tracking)
                    if len(self.correlation_chains[correlation_id]) > 100:
                        del self.correlation_chains[correlation_id]
                        
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collector: {e}")
                await asyncio.sleep(60)
                
    def get_event_history(self, event_type: Optional[EventType] = None, 
                         limit: int = 100) -> List[Event]:
        """Get recent event history"""
        
        events = list(self.event_history)
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
            
        return events[-limit:]
        
    def get_correlation_chain(self, correlation_id: str) -> List[str]:
        """Get event correlation chain"""
        
        return self.correlation_chains.get(correlation_id, [])
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive event system metrics"""
        
        return {
            'metrics': asdict(self.metrics),
            'queue_sizes': {
                priority.name: queue.qsize() 
                for priority, queue in self.event_queues.items()
            },
            'recent_failed_events': list(self.failed_events)[-10:],
            'average_processing_time': (
                sum(self.event_timings) / len(self.event_timings) 
                if self.event_timings else 0.0
            ),
            'worker_count': len(self.workers),
            'active_subscriptions': len(self.subscriptions)
        }

class EventEmitter:
    """
    Base class for components that emit events
    """
    
    def __init__(self, event_bus: EventBus, source_name: str):
        self.event_bus = event_bus
        self.source_name = source_name
        
    async def emit_event(self, event_type: EventType, data: Dict[str, Any], 
                        priority: EventPriority = EventPriority.NORMAL,
                        correlation_id: Optional[str] = None):
        """Emit an event"""
        
        event = Event(
            event_id=f"{self.source_name}_{event_type.value}_{int(time.time())}",
            event_type=event_type,
            source=self.source_name,
            timestamp=datetime.utcnow(),
            priority=priority,
            data=data,
            correlation_id=correlation_id
        )
        
        await self.event_bus.publish(event)
        
    async def emit_and_wait(self, event_type: EventType, data: Dict[str, Any],
                           timeout: float = 30.0) -> List[Any]:
        """Emit event and wait for responses"""
        
        event = Event(
            event_id=f"{self.source_name}_{event_type.value}_{int(time.time())}",
            event_type=event_type,
            source=self.source_name,
            timestamp=datetime.utcnow(),
            priority=EventPriority.NORMAL,
            data=data
        )
        
        return await self.event_bus.publish_and_wait(event, timeout)

# Utility functions and decorators

def event_handler(event_types: Union[EventType, List[EventType]], 
                 priority: int = 0, 
                 filter_func: Optional[Callable] = None):
    """Decorator for marking methods as event handlers"""
    
    def decorator(func):
        func._event_handler = True
        func._event_types = event_types if isinstance(event_types, list) else [event_types]
        func._handler_priority = priority
        func._filter_func = filter_func
        return func
    return decorator

async def register_event_handlers(obj: Any, event_bus: EventBus) -> List[str]:
    """Register all event handler methods from an object"""
    
    subscription_ids = []
    
    for attr_name in dir(obj):
        attr = getattr(obj, attr_name)
        
        if hasattr(attr, '_event_handler') and attr._event_handler:
            subscription_id = await event_bus.subscribe(
                event_types=attr._event_types,
                handler=attr,
                priority=getattr(attr, '_handler_priority', 0),
                filter_func=getattr(attr, '_filter_func', None)
            )
            subscription_ids.append(subscription_id)
            
    return subscription_ids

# Event correlation utilities

def create_correlation_id() -> str:
    """Create a new correlation ID"""
    return f"corr_{uuid.uuid4().hex[:12]}"

def event_correlation(correlation_id: str):
    """Decorator to add correlation ID to emitted events"""
    
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Store correlation ID in context (simplified)
            result = await func(*args, **kwargs)
            return result
        return wrapper
    return decorator