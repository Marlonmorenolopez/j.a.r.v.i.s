"""Tests for Event System (FASE 2B B7)."""

import sys
import time
sys.path.insert(0, ".")

from core.event_system import (
    EventBus,
    Event,
    EventType,
    EventPriority,
    get_event_bus,
    initialize_event_bus,
    emit_event,
    subscribe_to_events,
)


def test_bus_creation():
    """Verifica que el EventBus se crea correctamente."""
    bus = get_event_bus()
    stats = bus.get_stats()
    assert "total_published" in stats
    assert "history_size" in stats
    assert "generation" in stats
    print(f"PASS: EventBus created, stats={stats}")
    return True


def test_emit_and_subscribe_sync():
    """Verifica emisión y suscripción síncrona."""
    bus = get_event_bus()
    bus.clear()

    received = []

    def handler(event: Event):
        received.append(event)

    sub_id = bus.subscribe([EventType.EXECUTION_STARTED], handler)

    bus.emit(EventType.EXECUTION_STARTED, "test_source", {"capability": "open_app"})

    assert len(received) == 1
    assert received[0].type == EventType.EXECUTION_STARTED
    assert received[0].source == "test_source"
    assert received[0].payload["capability"] == "open_app"
    print(f"PASS: Sync emit/subscribe works")

    bus.unsubscribe(sub_id)
    return True


def test_multiple_subscribers_priority():
    """Verifica que los suscriptores se llaman en orden de prioridad."""
    bus = get_event_bus()
    bus.clear()

    order = []

    def handler_low(event: Event):
        order.append("low")

    def handler_high(event: Event):
        order.append("high")

    bus.subscribe([EventType.EXECUTION_STARTED], handler_low, priority=0)
    bus.subscribe([EventType.EXECUTION_STARTED], handler_high, priority=10)

    bus.emit(EventType.EXECUTION_STARTED, "test", {})

    assert order == ["high", "low"], f"Expected ['high', 'low'], got {order}"
    print(f"PASS: Priority ordering works")
    return True


def test_event_filter():
    """Verifica filtrado de eventos."""
    bus = get_event_bus()
    bus.clear()

    received = []

    def handler(event: Event):
        received.append(event)

    # Subscribe with filter for specific capability
    bus.subscribe(
        [EventType.EXECUTION_STARTED],
        handler,
        filter_fn=lambda e: e.payload.get("capability") == "open_app",
    )

    bus.emit(EventType.EXECUTION_STARTED, "test", {"capability": "open_app"})
    bus.emit(EventType.EXECUTION_STARTED, "test", {"capability": "web_search"})

    assert len(received) == 1
    assert received[0].payload["capability"] == "open_app"
    print(f"PASS: Event filtering works")
    return True


def test_once_subscription():
    """Verifica suscripción de un solo uso."""
    bus = get_event_bus()
    bus.clear()

    count = [0]

    def handler(event: Event):
        count[0] += 1

    bus.subscribe([EventType.EXECUTION_STARTED], handler, once=True)

    bus.emit(EventType.EXECUTION_STARTED, "test", {})
    bus.emit(EventType.EXECUTION_STARTED, "test", {})

    assert count[0] == 1, f"Expected 1, got {count[0]}"
    print(f"PASS: Once subscription works")
    return True


def test_async_dispatch():
    """Verifica despacho asíncrono."""
    bus = get_event_bus()
    bus.clear()

    received = []

    def handler(event: Event):
        received.append(event)

    bus.subscribe([EventType.ERROR_OCCURRED], handler)

    bus.emit(EventType.ERROR_OCCURRED, "test", {"error": "test"}, async_dispatch=True)

    # Wait for async worker
    time.sleep(0.2)

    assert len(received) == 1
    assert received[0].type == EventType.ERROR_OCCURRED
    print(f"PASS: Async dispatch works")
    return True


def test_event_history():
    """Verifica historial de eventos."""
    bus = get_event_bus()
    bus.clear()

    bus.emit(EventType.EXECUTION_STARTED, "source1", {"step": 1})
    bus.emit(EventType.EXECUTION_COMPLETED, "source1", {"step": 1})
    bus.emit(EventType.EXECUTION_STARTED, "source2", {"step": 2})

    history = bus.get_history(limit=10)
    assert len(history) == 3
    # Most recent first
    assert history[0].type == EventType.EXECUTION_STARTED
    assert history[0].source == "source2"

    # Filter by source
    source1_events = bus.get_history(source="source1")
    assert len(source1_events) == 2

    # Filter by type
    started_events = bus.get_history(event_types=[EventType.EXECUTION_STARTED])
    assert len(started_events) == 2

    print(f"PASS: Event history with filters works")
    return True


def test_replay():
    """Verifica replay de eventos históricos."""
    bus = get_event_bus()
    bus.clear()

    bus.emit(EventType.EXECUTION_STARTED, "test", {"step": 1})
    bus.emit(EventType.EXECUTION_COMPLETED, "test", {"step": 1})

    replayed = []

    def handler(event: Event):
        replayed.append(event)

    count = bus.replay(handler=handler)

    assert count == 2
    assert len(replayed) == 2
    assert replayed[0].type == EventType.EXECUTION_STARTED
    assert replayed[1].type == EventType.EXECUTION_COMPLETED
    print(f"PASS: Replay works")
    return True


def test_global_subscription():
    """Verifica suscripción global (todos los eventos)."""
    bus = get_event_bus()
    bus.clear()

    received = []

    def handler(event: Event):
        received.append(event)

    bus.subscribe_all(handler)

    bus.emit(EventType.EXECUTION_STARTED, "test", {})
    bus.emit(EventType.PERMISSION_GRANTED, "test", {})
    bus.emit(EventType.SYSTEM_SHUTDOWN, "test", {})

    assert len(received) == 3
    print(f"PASS: Global subscription works")
    return True


def test_correlation_id():
    """Verifica correlation ID para trazabilidad."""
    bus = get_event_bus()
    bus.clear()

    received = []

    def handler(event: Event):
        received.append(event)

    bus.subscribe([EventType.EXECUTION_STARTED, EventType.EXECUTION_COMPLETED], handler)

    corr_id = "corr-123"
    bus.emit(EventType.EXECUTION_STARTED, "test", {}, correlation_id=corr_id)
    bus.emit(EventType.EXECUTION_COMPLETED, "test", {}, correlation_id=corr_id)

    assert len(received) == 2
    assert received[0].correlation_id == corr_id
    assert received[1].correlation_id == corr_id
    print(f"PASS: Correlation ID works")
    return True


def test_event_priority_enum():
    """Verifica prioridad de eventos."""
    bus = get_event_bus()
    bus.clear()

    received = []

    def handler(event: Event):
        received.append(event.priority)

    bus.subscribe([EventType.EXECUTION_STARTED], handler)

    bus.emit(EventType.EXECUTION_STARTED, "test", {}, priority=EventPriority.LOW)
    bus.emit(EventType.EXECUTION_STARTED, "test", {}, priority=EventPriority.HIGH)
    bus.emit(EventType.EXECUTION_STARTED, "test", {}, priority=EventPriority.CRITICAL)

    assert received[0] == EventPriority.LOW
    assert received[1] == EventPriority.HIGH
    assert received[2] == EventPriority.CRITICAL
    print(f"PASS: Event priority works")
    return True


def test_source_filter():
    """Verifica filtro por fuente en suscripción."""
    bus = get_event_bus()
    bus.clear()

    received = []

    def handler(event: Event):
        received.append(event.source)

    bus.subscribe(
        [EventType.EXECUTION_STARTED],
        handler,
        source_filter="executor",
    )

    bus.emit(EventType.EXECUTION_STARTED, "executor", {})
    bus.emit(EventType.EXECUTION_STARTED, "planner", {})
    bus.emit(EventType.EXECUTION_STARTED, "executor", {})

    assert received == ["executor", "executor"]
    print(f"PASS: Source filter works")
    return True


def test_stats():
    """Verifica estadísticas del bus."""
    bus = get_event_bus()
    bus.clear()

    # Subscribe to events so they get dispatched
    bus.subscribe([EventType.EXECUTION_STARTED, EventType.EXECUTION_COMPLETED], lambda e: None)

    bus.emit(EventType.EXECUTION_STARTED, "test", {})
    bus.emit(EventType.EXECUTION_COMPLETED, "test", {})

    stats = bus.get_stats()
    assert stats["total_published"] == 2
    assert stats["total_dispatched"] >= 2
    assert stats["history_size"] == 2
    assert "generation" in stats
    print(f"PASS: Stats work: {stats}")
    return True


def test_generation_tracking():
    """Verifica tracking de generación."""
    bus = get_event_bus()
    bus.clear()

    gen1 = bus.get_generation()
    bus.emit(EventType.EXECUTION_STARTED, "test", {})
    gen2 = bus.get_generation()
    # Publish doesn't increment generation
    assert gen2 == gen1

    # But subscribe does
    bus.subscribe([EventType.EXECUTION_STARTED], lambda e: None)
    gen3 = bus.get_generation()
    assert gen3 > gen1
    print(f"PASS: Generation tracking works: {gen1} -> {gen2} -> {gen3}")
    return True


def test_clear():
    """Verifica limpieza del bus."""
    bus = get_event_bus()
    bus.clear()

    bus.emit(EventType.EXECUTION_STARTED, "test", {})
    bus.subscribe([EventType.EXECUTION_STARTED], lambda e: None)

    bus.clear()

    stats = bus.get_stats()
    assert stats["history_size"] == 0
    assert stats["active_subscriptions"] == 0
    print(f"PASS: Clear works")
    return True


def test_convenience_functions():
    """Verifica funciones de conveniencia."""
    bus = get_event_bus()
    bus.clear()

    received = []

    def handler(event: Event):
        received.append(event)

    sub_id = subscribe_to_events([EventType.PERMISSION_GRANTED], handler)

    emit_event(EventType.PERMISSION_GRANTED, "test", {"capability": "send_message"})

    assert len(received) == 1
    assert received[0].type == EventType.PERMISSION_GRANTED

    bus.unsubscribe(sub_id)
    print(f"PASS: Convenience functions work")
    return True


def test_latency_budget_warning(caplog=None):
    """Verifica advertencia de presupuesto de latencia (simulado)."""
    bus = get_event_bus()
    bus.clear()

    # This test just verifies the budget constants exist
    assert bus.LATENCY_BUDGET_SYNC == 10
    assert bus.LATENCY_BUDGET_ASYNC == 50
    print(f"PASS: Latency budgets configured")
    return True


def test_event_serialization():
    """Verifica serialización de eventos."""
    event = Event(
        type=EventType.EXECUTION_STARTED,
        source="test",
        payload={"capability": "open_app"},
        priority=EventPriority.HIGH,
        correlation_id="corr-123",
        metadata={"user": "marlon"},
    )

    d = event.to_dict()
    assert d["type"] == "execution_started"
    assert d["source"] == "test"
    assert d["payload"]["capability"] == "open_app"
    assert d["priority"] == EventPriority.HIGH.value
    assert d["correlation_id"] == "corr-123"
    assert d["metadata"]["user"] == "marlon"
    print(f"PASS: Event serialization works")
    return True


def test_all_event_types():
    """Verifica que todos los tipos de evento están definidos."""
    # Just verify enum has expected values
    expected_types = [
        "CAPABILITY_REGISTERED", "CAPABILITY_UNREGISTERED", "CAPABILITY_STATUS_CHANGED",
        "EXECUTION_STARTED", "EXECUTION_COMPLETED", "EXECUTION_FAILED", "EXECUTION_CANCELLED",
        "PERMISSION_REQUESTED", "PERMISSION_GRANTED", "PERMISSION_DENIED",
        "SECURITY_GATE_EVALUATED", "SECURITY_GATE_BLOCKED",
        "INTENT_CLASSIFIED", "INTENT_ROUTED", "FAST_PATH_EXECUTED", "AGENT_PATH_STARTED",
        "PLAN_CREATED", "PLAN_STEP_STARTED", "PLAN_STEP_COMPLETED", "PLAN_STEP_FAILED",
        "PLAN_REPLANNED", "PLAN_COMPLETED",
        "CONTEXT_UPDATED", "MEMORY_SAVED", "MEMORY_LOADED",
        "SYSTEM_STARTUP", "SYSTEM_SHUTDOWN", "ERROR_OCCURRED", "WARNING_ISSUED",
        "VOICE_STARTED", "VOICE_ENDED", "TRANSCRIPTION_RECEIVED",
        "AUDIO_PLAYBACK_STARTED", "AUDIO_PLAYBACK_ENDED",
    ]

    for expected in expected_types:
        assert hasattr(EventType, expected), f"Missing EventType: {expected}"

    print(f"PASS: All {len(expected_types)} event types defined")
    return True


def test_shutdown():
    """Verifica apagado graceful."""
    bus = get_event_bus()
    bus.clear()

    bus.emit(EventType.EXECUTION_STARTED, "test", {}, async_dispatch=True)
    time.sleep(0.05)

    bus.shutdown()

    # Should be able to create new instance after shutdown
    bus2 = EventBus()
    bus2.clear()
    bus2.shutdown()
    print(f"PASS: Shutdown works")
    return True


def main():
    """Ejecuta todas las pruebas."""
    print("=" * 60)
    print("TESTS Event System — FASE 2B B7")
    print("=" * 60)
    print()

    tests = [
        test_bus_creation,
        test_emit_and_subscribe_sync,
        test_multiple_subscribers_priority,
        test_event_filter,
        test_once_subscription,
        test_async_dispatch,
        test_event_history,
        test_replay,
        test_global_subscription,
        test_correlation_id,
        test_event_priority_enum,
        test_source_filter,
        test_stats,
        test_generation_tracking,
        test_clear,
        test_convenience_functions,
        test_latency_budget_warning,
        test_event_serialization,
        test_all_event_types,
        test_shutdown,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__} — {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 60)
    print(f"RESULTADO: {passed} pasaron, {failed} fallaron")
    print("=" * 60)

    if failed == 0:
        print("✅ Event System B7 — ESTABLE")
        return True
    else:
        print("❌ Event System B7 — TIENE FALLOS")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)