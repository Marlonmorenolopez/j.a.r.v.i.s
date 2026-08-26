"""Event System — Publish/subscribe event bus for P.I.P.E.

This module provides:
- Typed event classes for core system events
- EventBus singleton for publish/subscribe pattern
- Thread-safe event dispatching
- Event history and replay capability
- Integration with SecurityGate, PermissionManager, CapabilityRegistry
- Latency budgets for real-time event processing
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from collections import deque

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Core event types for P.I.P.E system."""
    # Capability lifecycle
    CAPABILITY_REGISTERED = "capability_registered"
    CAPABILITY_UNREGISTERED = "capability_unregistered"
    CAPABILITY_STATUS_CHANGED = "capability_status_changed"

    # Execution lifecycle
    EXECUTION_STARTED = "execution_started"
    EXECUTION_COMPLETED = "execution_completed"
    EXECUTION_FAILED = "execution_failed"
    EXECUTION_CANCELLED = "execution_cancelled"

    # Security & permissions
    PERMISSION_REQUESTED = "permission_requested"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_DENIED = "permission_denied"
    SECURITY_GATE_EVALUATED = "security_gate_evaluated"
    SECURITY_GATE_BLOCKED = "security_gate_blocked"

    # Intent routing
    INTENT_CLASSIFIED = "intent_classified"
    INTENT_ROUTED = "intent_routed"
    FAST_PATH_EXECUTED = "fast_path_executed"
    AGENT_PATH_STARTED = "agent_path_started"

    # Planning
    PLAN_CREATED = "plan_created"
    PLAN_STEP_STARTED = "plan_step_started"
    PLAN_STEP_COMPLETED = "plan_step_completed"
    PLAN_STEP_FAILED = "plan_step_failed"
    PLAN_REPLANNED = "plan_replanned"
    PLAN_COMPLETED = "plan_completed"

    # Context & memory
    CONTEXT_UPDATED = "context_updated"
    MEMORY_SAVED = "memory_saved"
    MEMORY_LOADED = "memory_loaded"

    # System
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    ERROR_OCCURRED = "error_occurred"
    WARNING_ISSUED = "warning_issued"

    # Voice/Audio
    VOICE_STARTED = "voice_started"
    VOICE_ENDED = "voice_ended"
    TRANSCRIPTION_RECEIVED = "transcription_received"
    AUDIO_PLAYBACK_STARTED = "audio_playback_started"
    AUDIO_PLAYBACK_ENDED = "audio_playback_ended"


class EventPriority(Enum):
    """Event priority for dispatch ordering."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass(frozen=True)
class Event:
    """Immutable event representation."""
    type: EventType
    source: str
    timestamp: float = field(default_factory=time.time)
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: EventPriority = EventPriority.NORMAL
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging/debugging."""
        return {
            "type": self.type.value,
            "source": self.source,
            "timestamp": self.timestamp,
            "payload": self.payload,
            "priority": self.priority.value,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata,
        }


# Type alias for event handlers
EventHandler = Callable[[Event], None]


@dataclass
class Subscription:
    """Event subscription with filter."""
    handler: EventHandler
    event_types: Set[EventType]
    priority: int = 0  # Higher = called first
    filter_fn: Optional[Callable[[Event], bool]] = None
    once: bool = False


class EventBus:
    """Thread-safe event bus for publish/subscribe pattern.

    Features:
    - Synchronous and asynchronous dispatch
    - Event filtering and priority ordering
    - Event history with bounded buffer
    - Generation tracking for cache invalidation
    - Latency budgets for real-time processing
    """

    _instance: Optional[EventBus] = None
    _lock = threading.RLock()

    # Latency budgets (milliseconds)
    LATENCY_BUDGET_SYNC = 10      # Sync dispatch must complete within 10ms
    LATENCY_BUDGET_ASYNC = 50     # Async dispatch queue target
    MAX_HISTORY = 10000           # Maximum events in history

    def __new__(cls) -> EventBus:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return

            self._subscriptions: Dict[str, List[Subscription]] = {}
            self._global_subscriptions: List[Subscription] = []
            self._history: deque = deque(maxlen=self.MAX_HISTORY)
            self._generation = 0
            self._dispatch_stats = {
                "total_published": 0,
                "total_dispatched": 0,
                "total_errors": 0,
                "sync_dispatches": 0,
                "async_dispatches": 0,
            }
            self._async_queue: deque = deque()
            self._async_worker: Optional[threading.Thread] = None
            self._async_running = False

            self._initialized = True
            logger.info("EventBus initialized")

    # =========================================================================
    # SUBSCRIPTION MANAGEMENT
    # =========================================================================

    def subscribe(
        self,
        event_types: List[EventType],
        handler: EventHandler,
        priority: int = 0,
        filter_fn: Optional[Callable[[Event], bool]] = None,
        once: bool = False,
        source_filter: Optional[str] = None,
    ) -> str:
        """Subscribe to event types.

        Args:
            event_types: List of event types to subscribe to
            handler: Callback function receiving Event
            priority: Higher priority handlers called first
            filter_fn: Optional predicate to filter events
            once: If True, auto-unsubscribe after first event
            source_filter: Optional source string to filter by

        Returns:
            Subscription ID for unsubscribing
        """
        import uuid
        sub_id = str(uuid.uuid4())[:8]

        event_type_set = set(event_types)

        # Wrap filter with source filter if provided
        if source_filter:
            original_filter = filter_fn
            filter_fn = lambda e: e.source == source_filter and (original_filter is None or original_filter(e))

        subscription = Subscription(
            handler=handler,
            event_types=event_type_set,
            priority=priority,
            filter_fn=filter_fn,
            once=once,
        )

        with self._lock:
            for event_type in event_type_set:
                key = event_type.value
                if key not in self._subscriptions:
                    self._subscriptions[key] = []
                self._subscriptions[key].append(subscription)
                # Sort by priority (highest first)
                self._subscriptions[key].sort(key=lambda s: -s.priority)

            self._generation += 1
            logger.debug("Subscribed %s to %s (priority=%d)", sub_id, [t.value for t in event_types], priority)

        # Attach sub_id to subscription for later lookup
        subscription._sub_id = sub_id  # type: ignore

        return sub_id

    def subscribe_all(
        self,
        handler: EventHandler,
        priority: int = 0,
        filter_fn: Optional[Callable[[Event], bool]] = None,
    ) -> str:
        """Subscribe to all events."""
        import uuid
        sub_id = str(uuid.uuid4())[:8]

        subscription = Subscription(
            handler=handler,
            event_types=set(EventType),  # All types
            priority=priority,
            filter_fn=filter_fn,
        )

        with self._lock:
            self._global_subscriptions.append(subscription)
            self._global_subscriptions.sort(key=lambda s: -s.priority)
            self._generation += 1

        subscription._sub_id = sub_id  # type: ignore
        return sub_id

    def unsubscribe(self, sub_id: str) -> bool:
        """Unsubscribe by subscription ID."""
        with self._lock:
            # Check global subscriptions
            for i, sub in enumerate(self._global_subscriptions):
                if id(sub) == id(sub_id) or (hasattr(sub, '_sub_id') and sub._sub_id == sub_id):
                    self._global_subscriptions.pop(i)
                    self._generation += 1
                    return True

            # Check per-event-type subscriptions
            for event_type, subs in self._subscriptions.items():
                for i, sub in enumerate(subs):
                    if id(sub) == id(sub_id) or (hasattr(sub, '_sub_id') and sub._sub_id == sub_id):
                        subs.pop(i)
                        self._generation += 1
                        return True

        return False

    def _attach_sub_id(self, subscription: Subscription, sub_id: str) -> None:
        """Attach subscription ID for later lookup."""
        subscription._sub_id = sub_id  # type: ignore

    # =========================================================================
    # EVENT PUBLISHING
    # =========================================================================

    def publish(self, event: Event, async_dispatch: bool = False) -> None:
        """Publish an event to all subscribers.

        Args:
            event: Event to publish
            async_dispatch: If True, dispatch asynchronously via worker thread
        """
        with self._lock:
            self._history.append(event)
            self._dispatch_stats["total_published"] += 1

        if async_dispatch:
            self._dispatch_async(event)
        else:
            self._dispatch_sync(event)

    def publish_sync(self, event: Event) -> None:
        """Publish event with synchronous dispatch (blocking)."""
        self.publish(event, async_dispatch=False)

    def publish_async(self, event: Event) -> None:
        """Publish event with asynchronous dispatch (non-blocking)."""
        self.publish(event, async_dispatch=True)

    def _dispatch_sync(self, event: Event) -> None:
        """Synchronous event dispatch with latency budget."""
        start = time.perf_counter()

        handlers = self._get_handlers(event)

        for subscription in handlers:
            try:
                if subscription.filter_fn and not subscription.filter_fn(event):
                    continue
                subscription.handler(event)
                self._dispatch_stats["total_dispatched"] += 1

                if subscription.once:
                    self.unsubscribe(subscription._sub_id)  # type: ignore

            except Exception as e:
                self._dispatch_stats["total_errors"] += 1
                logger.error("Event handler error for %s: %s", event.type.value, e)

        elapsed_ms = (time.perf_counter() - start) * 1000
        self._dispatch_stats["sync_dispatches"] += 1

        if elapsed_ms > self.LATENCY_BUDGET_SYNC:
            logger.warning(
                "Sync dispatch latency budget exceeded: %.2fms for %s (budget: %dms)",
                elapsed_ms, event.type.value, self.LATENCY_BUDGET_SYNC
            )

    def _dispatch_async(self, event: Event) -> None:
        """Queue event for asynchronous dispatch."""
        with self._lock:
            self._async_queue.append(event)
            self._dispatch_stats["async_dispatches"] += 1

        # Start worker if not running
        if not self._async_running:
            self._start_async_worker()

    def _start_async_worker(self) -> None:
        """Start the async dispatch worker thread."""
        if self._async_running:
            return

        self._async_running = True
        self._async_worker = threading.Thread(target=self._async_worker_loop, daemon=True)
        self._async_worker.start()

    def _async_worker_loop(self) -> None:
        """Async worker loop for event dispatch."""
        while self._async_running:
            event = None
            with self._lock:
                if self._async_queue:
                    event = self._async_queue.popleft()

            if event:
                self._dispatch_sync(event)
            else:
                time.sleep(0.001)  # 1ms sleep when queue empty

    def stop_async_worker(self) -> None:
        """Stop the async worker and drain queue."""
        self._async_running = False
        if self._async_worker:
            self._async_worker.join(timeout=1.0)
            self._async_worker = None

    def _get_handlers(self, event: Event) -> List[Subscription]:
        """Get all handlers for an event, ordered by priority."""
        handlers = []

        with self._lock:
            # Global subscriptions (catch all)
            handlers.extend(self._global_subscriptions)

            # Type-specific subscriptions
            key = event.type.value
            if key in self._subscriptions:
                handlers.extend(self._subscriptions[key])

        # Sort by priority (highest first)
        handlers.sort(key=lambda s: -s.priority)
        return handlers

    # =========================================================================
    # EVENT HISTORY & REPLAY
    # =========================================================================

    def get_history(
        self,
        limit: int = 100,
        event_types: Optional[List[EventType]] = None,
        source: Optional[str] = None,
        since: Optional[float] = None,
    ) -> List[Event]:
        """Get event history with optional filters."""
        with self._lock:
            events = list(self._history)

        # Apply filters
        if event_types:
            type_set = set(event_types)
            events = [e for e in events if e.type in type_set]

        if source:
            events = [e for e in events if e.source == source]

        if since:
            events = [e for e in events if e.timestamp >= since]

        # Return most recent first
        events.reverse()
        return events[:limit]

    def replay(
        self,
        event_types: Optional[List[EventType]] = None,
        source: Optional[str] = None,
        since: Optional[float] = None,
        handler: Optional[EventHandler] = None,
    ) -> int:
        """Replay historical events to a handler (or all current subscribers)."""
        events = self.get_history(
            limit=self.MAX_HISTORY,
            event_types=event_types,
            source=source,
            since=since,
        )

        # Reverse to get chronological order
        events.reverse()

        count = 0
        for event in events:
            if handler:
                try:
                    handler(event)
                    count += 1
                except Exception as e:
                    logger.error("Replay handler error: %s", e)
            else:
                # Dispatch to current subscribers
                self._dispatch_sync(event)
                count += 1

        return count

    def clear_history(self) -> None:
        """Clear event history."""
        with self._lock:
            self._history.clear()

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    def emit(
        self,
        event_type: EventType,
        source: str,
        payload: Optional[Dict[str, Any]] = None,
        priority: EventPriority = EventPriority.NORMAL,
        correlation_id: Optional[str] = None,
        async_dispatch: bool = False,
        **metadata,
    ) -> Event:
        """Convenience method to create and publish an event."""
        event = Event(
            type=event_type,
            source=source,
            payload=payload or {},
            priority=priority,
            correlation_id=correlation_id,
            metadata=metadata,
        )
        self.publish(event, async_dispatch)
        return event

    # =========================================================================
    # STATS & UTILITIES
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        with self._lock:
            return {
                "total_published": self._dispatch_stats["total_published"],
                "total_dispatched": self._dispatch_stats["total_dispatched"],
                "total_errors": self._dispatch_stats["total_errors"],
                "sync_dispatches": self._dispatch_stats["sync_dispatches"],
                "async_dispatches": self._dispatch_stats["async_dispatches"],
                "history_size": len(self._history),
                "history_capacity": self.MAX_HISTORY,
                "active_subscriptions": sum(len(s) for s in self._subscriptions.values()) + len(self._global_subscriptions),
                "subscription_types": len(self._subscriptions),
                "async_queue_size": len(self._async_queue),
                "async_worker_running": self._async_running,
                "generation": self._generation,
            }

    def get_generation(self) -> int:
        """Get generation counter for cache invalidation."""
        return self._generation

    def clear(self) -> None:
        """Clear all state (for testing)."""
        with self._lock:
            self._subscriptions.clear()
            self._global_subscriptions.clear()
            self._history.clear()
            self._async_queue.clear()
            self._dispatch_stats = {
                "total_published": 0,
                "total_dispatched": 0,
                "total_errors": 0,
                "sync_dispatches": 0,
                "async_dispatches": 0,
            }
            self._generation += 1

    def shutdown(self) -> None:
        """Graceful shutdown."""
        self.stop_async_worker()
        self.clear()
        logger.info("EventBus shutdown complete")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get the global EventBus instance."""
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus


def initialize_event_bus() -> None:
    """Explicit initialization."""
    get_event_bus()
    logger.info("EventBus explicitly initialized")


def emit_event(
    event_type: EventType,
    source: str,
    payload: Optional[Dict[str, Any]] = None,
    priority: EventPriority = EventPriority.NORMAL,
    correlation_id: Optional[str] = None,
    async_dispatch: bool = False,
    **metadata,
) -> Event:
    """Convenience function to emit an event."""
    return get_event_bus().emit(
        event_type=event_type,
        source=source,
        payload=payload,
        priority=priority,
        correlation_id=correlation_id,
        async_dispatch=async_dispatch,
        **metadata,
    )


def subscribe_to_events(
    event_types: List[EventType],
    handler: EventHandler,
    priority: int = 0,
    filter_fn: Optional[Callable[[Event], bool]] = None,
    once: bool = False,
) -> str:
    """Convenience function to subscribe to events."""
    return get_event_bus().subscribe(event_types, handler, priority, filter_fn, once)


if __name__ == "__main__":
    # Demo
    bus = get_event_bus()
    bus.clear()

    print("=== EventBus Demo ===")

    # Track received events
    received = []

    def handler(event: Event):
        received.append(event)
        print(f"Received: {event.type.value} from {event.source} - {event.payload}")

    # Subscribe to execution events
    sub_id = bus.subscribe(
        [EventType.EXECUTION_STARTED, EventType.EXECUTION_COMPLETED, EventType.EXECUTION_FAILED],
        handler,
        priority=10,
    )

    # Emit some events
    bus.emit(EventType.EXECUTION_STARTED, "executor", {"capability": "open_app", "step": 1})
    bus.emit(EventType.EXECUTION_COMPLETED, "executor", {"capability": "open_app", "result": "success"})

    # Test async
    bus.emit(EventType.ERROR_OCCURRED, "security_gate", {"error": "rate limited"}, async_dispatch=True)
    time.sleep(0.1)  # Wait for async

    print(f"\nHistory: {len(bus.get_history())} events")
    print(f"Stats: {bus.get_stats()}")

    bus.unsubscribe(sub_id)
    bus.shutdown()
    print("\n✅ EventBus demo complete")