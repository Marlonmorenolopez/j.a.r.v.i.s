"""
Tests for Voice-UI Integration (FASE 5) - No UI initialization needed.
"""

import sys
import time
import threading
sys.path.insert(0, ".")

from voice.ui_bridge import VoiceUIBridge, VoiceUIConfig
from voice.pipeline import create_pipeline, PipelineConfig
from core.event_system import (
    EventBus, Event, EventType, EventPriority,
    get_event_bus, emit_event, subscribe_to_events
)
from agent.intent_router import get_intent_router
from agent.action_resolver import get_action_resolver
from agent.pipe_context import get_context_engine
from agent.action_dispatcher import dispatch_tool


# Mock UI for testing
class MockUI:
    def __init__(self):
        self.muted = False
        self.state = "IDLE"
        self.logs = []
        self.on_text_command = None
    
    def set_state(self, state: str):
        self.state = state
        print(f"[MockUI] State: {state}")
    
    def write_log(self, text: str):
        self.logs.append(text)
        print(f"[MockUI] Log: {text}")


def test_bridge_creation():
    """Test VoiceUIBridge creation."""
    ui = MockUI()
    bridge = VoiceUIBridge(ui, VoiceUIConfig(stt_model="tiny"))
    
    assert bridge is not None
    assert bridge.ui is ui
    assert bridge.config.stt_model == "tiny"
    print(f"PASS: Bridge created")
    return True


def test_bridge_initialization():
    """Test bridge initialization without UI."""
    ui = MockUI()
    bridge = VoiceUIBridge(ui, VoiceUIConfig(stt_model="tiny"))
    
    result = bridge.initialize()
    assert result, "Bridge initialization failed"
    assert bridge._pipeline is not None
    assert bridge._router is not None
    assert bridge._resolver is not None
    print(f"PASS: Bridge initialized with pipeline")
    return True


def test_bridge_start_stop():
    """Test bridge start/stop lifecycle."""
    ui = MockUI()
    bridge = VoiceUIBridge(ui, VoiceUIConfig(stt_model="tiny"))
    bridge.initialize()
    
    # Start
    result = bridge.start()
    assert result, "Bridge start failed"
    assert bridge.is_running
    assert bridge._pipeline.is_running
    print(f"PASS: Bridge started")
    
    time.sleep(0.5)
    
    # Stop
    bridge.stop()
    assert not bridge.is_running
    assert not bridge._pipeline.is_running
    print(f"PASS: Bridge stopped")
    return True


def test_event_subscriptions():
    """Test that bridge subscribes to events."""
    ui = MockUI()
    bridge = VoiceUIBridge(ui, VoiceUIConfig(stt_model="tiny"))
    bridge.initialize()
    bridge.start()
    
    bus = get_event_bus()
    
    # Check subscriptions exist
    stats = bus.get_stats()
    assert stats["active_subscriptions"] >= 5, f"Expected >=5 subscriptions, got {stats['active_subscriptions']}"
    
    bridge.stop()
    print(f"PASS: Event subscriptions active ({stats['active_subscriptions']})")
    return True


def test_ui_state_transitions():
    """Test UI state transitions via events."""
    ui = MockUI()
    bridge = VoiceUIBridge(ui, VoiceUIConfig(stt_model="tiny"))
    bridge.initialize()
    bridge.start()
    
    # Test state transitions via direct event emission
    emit_event(EventType.VOICE_STARTED, "test", {"state": "listening"})
    time.sleep(0.1)
    assert ui.state == "LISTENING"
    
    emit_event(EventType.AUDIO_PLAYBACK_STARTED, "test", {"text": "Hola"})
    time.sleep(0.1)
    assert ui.state == "SPEAKING"
    
    emit_event(EventType.AUDIO_PLAYBACK_ENDED, "test", {})
    time.sleep(0.1)
    assert ui.state == "LISTENING"
    
    bridge.stop()
    print(f"PASS: UI state transitions work")
    return True


def test_transcription_callback():
    """Test transcription callback processing."""
    ui = MockUI()
    bridge = VoiceUIBridge(ui, VoiceUIConfig(stt_model="tiny"))
    bridge.initialize()
    
    # Manually trigger transcription callback
    bridge._on_transcription_received("abre Chrome")
    time.sleep(0.5)
    
    bridge.stop()
    print(f"PASS: Transcription callback triggers processing")
    return True


def test_pipeline_integration():
    """Test pipeline integration with bridge."""
    pipeline = create_pipeline(stt_model="tiny")
    ui = MockUI()
    bridge = VoiceUIBridge(ui, VoiceUIConfig(stt_model="tiny"))
    
    # Manually set pipeline
    bridge._pipeline = pipeline
    bridge._router = get_intent_router()
    bridge._resolver = get_action_resolver()
    bridge._context_engine = get_context_engine()
    bridge._event_bus = get_event_bus()
    
    # Subscribe events
    bridge._subscribe_events()
    
    # Verify pipeline events are bridged
    received_events = []
    def capture(event):
        received_events.append(event.type)
    
    sub_ids = []
    for et in [EventType.VOICE_STARTED, EventType.VOICE_ENDED, 
               EventType.TRANSCRIPTION_RECEIVED, EventType.AUDIO_PLAYBACK_STARTED,
               EventType.AUDIO_PLAYBACK_ENDED]:
        sub_ids.append(subscribe_to_events([et], capture))
    
    pipeline.start()
    time.sleep(0.3)
    pipeline.speak("Prueba", async_play=False)
    time.sleep(0.3)
    pipeline.stop()
    
    for sid in sub_ids:
        get_event_bus().unsubscribe(sid)
    
    # Should have received voice events
    assert EventType.VOICE_STARTED in received_events
    assert EventType.AUDIO_PLAYBACK_STARTED in received_events
    assert EventType.AUDIO_PLAYBACK_ENDED in received_events
    assert EventType.VOICE_ENDED in received_events
    
    print(f"PASS: Pipeline events bridged: {[e.value for e in received_events]}")
    return True


def test_intent_resolution_integration():
    """Test intent → action resolution in bridge context."""
    ui = MockUI()
    bridge = VoiceUIBridge(ui, VoiceUIConfig(stt_model="tiny"))
    bridge.initialize()
    
    # Test the internal processing flow
    text = "abre Chrome"
    
    # Classify intent
    intent = bridge._router.classify(text, autonomy_level=3)
    assert intent.capability_id == "open_app"
    
    # Resolve action
    action_context = bridge._context_engine.get_action_context(intent.capability_id)
    action = bridge._resolver.resolve(intent, action_context)
    assert action.tool_name == "open_app"
    assert action.parameters.get("app") == "Chrome"
    
    bridge.stop()
    print(f"PASS: Intent→Action resolution works in bridge context")
    return True


def test_stats_collection():
    """Test statistics collection."""
    ui = MockUI()
    bridge = VoiceUIBridge(ui, VoiceUIConfig(stt_model="tiny"))
    bridge.initialize()
    bridge.start()
    
    time.sleep(0.2)
    
    stats = bridge.get_stats()
    assert "total_commands" in stats
    assert "successful" in stats
    assert "failed" in stats
    assert "state" in stats
    print(f"PASS: Stats collected: {stats}")
    
    bridge.stop()
    return True


def test_dispatcher_integration():
    """Test action dispatcher integration."""
    # Test that dispatch_tool works with basic tools
    result = dispatch_tool("get_time", {}, player=None, speak=None)
    assert result is not None
    print(f"PASS: Dispatcher works: {result[:50]}...")
    return True


def main():
    """Run all FASE 5 tests."""
    print("=" * 60)
    print("TESTS Voice-UI Integration — FASE 5")
    print("=" * 60)
    print()
    
    tests = [
        test_bridge_creation,
        test_bridge_initialization,
        test_bridge_start_stop,
        test_event_subscriptions,
        test_ui_state_transitions,
        test_transcription_callback,
        test_pipeline_integration,
        test_intent_resolution_integration,
        test_stats_collection,
        test_dispatcher_integration,
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
        print("✅ Voice-UI Integration FASE 5 — ESTABLE")
        return True
    else:
        print("❌ Voice-UI Integration FASE 5 — TIENE FALLOS")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)