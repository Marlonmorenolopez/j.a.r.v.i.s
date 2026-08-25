"""
Tests for Voice Pipeline (FASE 4).
"""

import sys
import time
import threading
sys.path.insert(0, ".")

from voice.capture import VoiceCapture, AudioConfig, list_microphones
from voice.stt import VoiceSTT, STTConfig
from voice.tts import VoiceTTS, TTSConfig
from voice.pipeline import VoicePipeline, PipelineConfig, create_pipeline
from core.event_system import (
    EventBus, Event, EventType, EventPriority, 
    get_event_bus, emit_event, subscribe_to_events
)


def test_microphone_listing():
    """Test microphone enumeration."""
    mics = list_microphones()
    assert len(mics) > 0, "No microphones found"
    print(f"PASS: Found {len(mics)} microphone(s)")
    for mic in mics[:3]:
        print(f"  {mic['index']}: {mic['name']}")
    return True


def test_capture_basic():
    """Test basic audio capture."""
    capture = VoiceCapture()
    audio = capture.get_audio_buffer(1.0)  # 1 second
    assert len(audio) > 0, "No audio captured"
    expected_bytes = int(1.0 * 16000 * 2)  # 1 sec * 16kHz * 2 bytes
    assert abs(len(audio) - expected_bytes) < expected_bytes * 0.2, "Audio length mismatch"
    print(f"PASS: Captured {len(audio)} bytes (~1s)")
    return True


def test_stt_initialization():
    """Test STT model loading."""
    stt = VoiceSTT(STTConfig(model_size="tiny", language="es"))  # Use tiny for speed
    assert stt._model is not None, "Model not loaded"
    print(f"PASS: STT model loaded ({stt.config.model_size})")
    return True


def test_stt_transcribe_silence():
    """Test STT on silence (should return empty)."""
    stt = VoiceSTT(STTConfig(model_size="tiny", language="es"))
    
    # Generate silence
    import numpy as np
    import wave
    import tempfile
    import os
    
    sample_rate = 16000
    duration = 1.0
    silence = np.zeros(int(sample_rate * duration), dtype=np.int16)
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
        with wave.open(temp_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(silence.tobytes())
    
    try:
        text = stt.transcribe_file(temp_path)
        # Silence should produce empty or very short transcription
        assert len(text.strip()) == 0, f"Expected empty transcription, got: '{text}'"
        print(f"PASS: Silence transcribed as empty")
    finally:
        os.unlink(temp_path)
    return True


def test_tts_initialization():
    """Test TTS initialization."""
    tts = VoiceTTS(TTSConfig(voice="es-ES-AlvaroNeural"))
    assert tts.config.voice == "es-ES-AlvaroNeural"
    print(f"PASS: TTS initialized with {tts.config.voice}")
    return True


def test_tts_synthesis():
    """Test TTS audio generation."""
    tts = VoiceTTS(TTSConfig(voice="es-ES-AlvaroNeural"))
    import asyncio
    
    async def test():
        audio = await tts.synthesize("Hola, prueba de voz")
        assert len(audio) > 1000, f"Audio too small: {len(audio)} bytes"
        print(f"PASS: TTS synthesized {len(audio)} bytes")
    
    asyncio.run(test())
    return True


def test_event_integration():
    """Test voice pipeline integrates with event system."""
    bus = get_event_bus()
    bus.clear()
    
    received = []
    
    def handler(event: Event):
        received.append(event)
    
    sub_id = subscribe_to_events([EventType.TRANSCRIPTION_RECEIVED], handler)
    
    # Emit a transcription event
    emit_event(EventType.TRANSCRIPTION_RECEIVED, "test_stt", {
        "text": "Hola mundo",
        "success": True,
        "latency_ms": 150.0
    })
    
    time.sleep(0.05)  # Allow dispatch
    
    assert len(received) == 1, f"Expected 1 event, got {len(received)}"
    assert received[0].type == EventType.TRANSCRIPTION_RECEIVED
    assert received[0].payload["text"] == "Hola mundo"
    assert received[0].source == "test_stt"
    
    bus.unsubscribe(sub_id)
    print(f"PASS: Event integration works")
    return True


def test_pipeline_creation():
    """Test pipeline creation and configuration."""
    pipeline = create_pipeline(
        stt_model="tiny",
        tts_voice="es-ES-AlvaroNeural",
        language="es"
    )
    
    assert pipeline.config.stt_model == "tiny"
    assert pipeline.config.tts_voice == "es-ES-AlvaroNeural"
    assert pipeline.config.stt_language == "es"
    assert pipeline.state.value == "idle"
    print(f"PASS: Pipeline created with config")
    return True


def test_pipeline_start_stop():
    """Test pipeline start/stop lifecycle."""
    pipeline = create_pipeline(stt_model="tiny")
    
    # Start
    result = pipeline.start()
    assert result, "Pipeline start failed"
    assert pipeline.is_running, "Pipeline not running after start"
    assert pipeline.state.value == "listening"
    print(f"PASS: Pipeline started (state={pipeline.state.value})")
    
    # Give it a moment
    time.sleep(0.5)
    
    # Stop
    pipeline.stop()
    assert not pipeline.is_running, "Pipeline still running after stop"
    assert pipeline.state.value == "idle"
    print(f"PASS: Pipeline stopped (state={pipeline.state.value})")
    return True


def test_pipeline_metrics():
    """Test pipeline metrics collection."""
    pipeline = create_pipeline(stt_model="tiny")
    pipeline.start()
    time.sleep(0.2)
    pipeline.stop()
    
    metrics = pipeline.get_metrics()
    assert "total_utterances" in metrics
    assert "successful_transcriptions" in metrics
    assert "failed_transcriptions" in metrics
    assert "state" in metrics
    print(f"PASS: Metrics collected: {metrics}")
    return True


def test_voice_events_emitted():
    """Test that pipeline emits proper events."""
    bus = get_event_bus()
    bus.clear()
    
    received = []
    
    def handler(event: Event):
        received.append(event)
    
    # Subscribe to voice events
    sub1 = subscribe_to_events([EventType.VOICE_STARTED], handler)
    sub2 = subscribe_to_events([EventType.VOICE_ENDED], handler)
    sub3 = subscribe_to_events([EventType.AUDIO_PLAYBACK_STARTED], handler)
    sub4 = subscribe_to_events([EventType.AUDIO_PLAYBACK_ENDED], handler)
    
    pipeline = create_pipeline(stt_model="tiny")
    pipeline.start()
    time.sleep(0.3)
    pipeline.speak("Prueba de voz", async_play=False)
    time.sleep(0.5)
    pipeline.stop()
    
    # Check events were emitted
    event_types = [e.type for e in received]
    assert EventType.VOICE_STARTED in event_types, "VOICE_STARTED not emitted"
    assert EventType.AUDIO_PLAYBACK_STARTED in event_types, "AUDIO_PLAYBACK_STARTED not emitted"
    assert EventType.AUDIO_PLAYBACK_ENDED in event_types, "AUDIO_PLAYBACK_ENDED not emitted"
    
    for sub in [sub1, sub2, sub3, sub4]:
        bus.unsubscribe(sub)
    
    print(f"PASS: Voice events emitted: {[e.value for e in event_types]}")
    return True


def main():
    """Run all voice pipeline tests."""
    print("=" * 60)
    print("TESTS Voice Pipeline — FASE 4")
    print("=" * 60)
    print()
    
    tests = [
        test_microphone_listing,
        test_capture_basic,
        test_stt_initialization,
        test_stt_transcribe_silence,
        test_tts_initialization,
        test_tts_synthesis,
        test_event_integration,
        test_pipeline_creation,
        test_pipeline_start_stop,
        test_pipeline_metrics,
        test_voice_events_emitted,
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
        print("✅ Voice Pipeline FASE 4 — ESTABLE")
        return True
    else:
        print("❌ Voice Pipeline FASE 4 — TIENE FALLOS")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)