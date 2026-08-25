"""
Voice Pipeline — Complete voice processing pipeline integrating capture, STT, TTS, and event system.
"""

import asyncio
import threading
import time
import queue
from pathlib import Path
from typing import Optional, Callable, Generator, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
import logging

from core.event_system import (
    EventBus, Event, EventType, EventPriority, 
    get_event_bus, emit_event, subscribe_to_events
)

from voice.capture import VoiceCapture, AudioConfig
from voice.stt import VoiceSTT, STTConfig
from voice.tts import VoiceTTS, TTSConfig, VoiceStyle


logger = logging.getLogger(__name__)


class PipelineState(Enum):
    """Voice pipeline states."""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ERROR = "error"


@dataclass
class PipelineConfig:
    """Configuration for the voice pipeline."""
    # Audio capture
    sample_rate: int = 16000
    channels: int = 1
    blocksize: int = 1024
    vad_threshold: int = 500
    vad_silence_seconds: float = 0.85
    
    # STT
    stt_model: str = "base"
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"
    stt_language: str = "es"
    
    # TTS
    tts_voice: str = "es-ES-AlvaroNeural"
    tts_rate: str = "+0%"
    tts_volume: str = "+0%"
    
    # Behavior
    auto_start: bool = False
    min_utterance_length: int = 3  # Minimum characters to consider valid
    max_silence_before_process: float = 1.0  # Seconds of silence before processing


class VoicePipeline:
    """
    Complete voice pipeline: Capture → VAD → STT → Event → TTS
    
    Integrates with the EventBus for system-wide communication.
    
    Usage:
        pipeline = VoicePipeline(PipelineConfig())
        pipeline.start()
        # Pipeline runs in background, emits events for transcriptions
        pipeline.stop()
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None, 
                 on_transcription: Optional[Callable[[str], None]] = None):
        self.config = config or PipelineConfig()
        self.on_transcription = on_transcription
        
        # State
        self._state = PipelineState.IDLE
        self._running = False
        self._lock = threading.RLock()
        
        # Components
        self._capture: Optional[VoiceCapture] = None
        self._stt: Optional[VoiceSTT] = None
        self._tts: Optional[VoiceTTS] = None
        
        # Event bus
        self._event_bus = get_event_bus()
        self._subscription_ids: List[str] = []
        
        # Audio processing
        self._audio_queue: queue.Queue = queue.Queue(maxsize=50)
        self._vad_buffer = bytearray()
        self._silence_start: Optional[float] = None
        self._speech_active = False
        
        # Threads
        self._capture_thread: Optional[threading.Thread] = None
        self._process_thread: Optional[threading.Thread] = None
        
        # Metrics
        self._metrics = {
            "total_utterances": 0,
            "successful_transcriptions": 0,
            "failed_transcriptions": 0,
            "total_latency_ms": 0.0,
        }
    
    # =========================================================================
    # INITIALIZATION
    # =========================================================================
    
    def initialize(self) -> bool:
        """Initialize all pipeline components."""
        try:
            # Initialize capture
            audio_config = AudioConfig(
                sample_rate=self.config.sample_rate,
                channels=self.config.channels,
                blocksize=self.config.blocksize,
            )
            self._capture = VoiceCapture(config=audio_config)
            
            # Initialize STT
            stt_config = STTConfig(
                model_size=self.config.stt_model,
                device=self.config.stt_device,
                compute_type=self.config.stt_compute_type,
                language=self.config.stt_language,
            )
            self._stt = VoiceSTT(config=stt_config)
            
            # Initialize TTS
            tts_config = TTSConfig(
                voice=self.config.tts_voice,
                rate=self.config.tts_rate,
                volume=self.config.tts_volume,
            )
            self._tts = VoiceTTS(config=tts_config)
            
            # Subscribe to events
            self._subscribe_events()
            
            # Emit startup event
            emit_event(EventType.SYSTEM_STARTUP, "voice_pipeline", {
                "components": ["capture", "stt", "tts"],
                "config": {
                    "stt_model": self.config.stt_model,
                    "tts_voice": self.config.tts_voice,
                    "language": self.config.stt_language,
                }
            })
            
            logger.info("Voice pipeline initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize voice pipeline: {e}")
            emit_event(EventType.ERROR_OCCURRED, "voice_pipeline", {
                "error": str(e),
                "phase": "initialization"
            })
            return False
    
    def _subscribe_events(self) -> None:
        """Subscribe to relevant system events."""
        # Subscribe to shutdown
        sub_id = subscribe_to_events([EventType.SYSTEM_SHUTDOWN], self._on_shutdown)
        self._subscription_ids.append(sub_id)
        
        # Subscribe to voice commands (from other components)
        sub_id = subscribe_to_events([EventType.VOICE_STARTED], self._on_voice_started)
        self._subscription_ids.append(sub_id)
        
        sub_id = subscribe_to_events([EventType.VOICE_ENDED], self._on_voice_ended)
        self._subscription_ids.append(sub_id)
    
    # =========================================================================
    # LIFECYCLE
    # =========================================================================
    
    def start(self) -> bool:
        """Start the voice pipeline."""
        with self._lock:
            if self._running:
                logger.warning("Pipeline already running")
                return True
            
            if not self.initialize():
                return False
            
            self._running = True
            self._state = PipelineState.LISTENING
            
            # Start capture thread
            self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._capture_thread.start()
            
            # Start processing thread
            self._process_thread = threading.Thread(target=self._process_loop, daemon=True)
            self._process_thread.start()
            
            # Emit voice started event
            emit_event(EventType.VOICE_STARTED, "voice_pipeline", {
                "state": self._state.value
            })
            
            logger.info("Voice pipeline started")
            return True
    
    def stop(self) -> None:
        """Stop the voice pipeline."""
        with self._lock:
            if not self._running:
                return
            
            self._running = False
            self._state = PipelineState.IDLE
            
            # Stop capture
            if self._capture:
                self._capture.stop()
            
            # Wait for threads
            if self._capture_thread:
                self._capture_thread.join(timeout=2.0)
            if self._process_thread:
                self._process_thread.join(timeout=2.0)
            
            # Unsubscribe events
            for sub_id in self._subscription_ids:
                self._event_bus.unsubscribe(sub_id)
            self._subscription_ids.clear()
            
            # Emit voice ended event
            emit_event(EventType.VOICE_ENDED, "voice_pipeline", {
                "state": self._state.value,
                "metrics": self._metrics.copy()
            })
            
            logger.info("Voice pipeline stopped")
    
    def _on_shutdown(self, event: Event) -> None:
        """Handle system shutdown event."""
        self.stop()
    
    def _on_voice_started(self, event: Event) -> None:
        """Handle voice started event from other components."""
        pass
    
    def _on_voice_ended(self, event: Event) -> None:
        """Handle voice ended event from other components."""
        pass
    
    # =========================================================================
    # CAPTURE LOOP
    # =========================================================================
    
    def _capture_loop(self) -> None:
        """Main capture loop running in background thread."""
        if not self._capture:
            return
        
        self._capture.start()
        
        try:
            for chunk in self._capture.stream():
                if not self._running:
                    break
                
                # VAD processing
                self._process_vad(chunk)
                
        except Exception as e:
            logger.error(f"Capture loop error: {e}")
            emit_event(EventType.ERROR_OCCURRED, "voice_pipeline", {
                "error": str(e),
                "phase": "capture"
            })
        finally:
            self._capture.stop()
    
    def _process_vad(self, audio_chunk: bytes) -> None:
        """Process audio chunk with Voice Activity Detection."""
        # Calculate RMS
        samples = audio_chunk  # Already int16 bytes
        sample_count = len(samples) // 2
        if sample_count == 0:
            return
        
        # Convert to int16 array for RMS calculation
        import struct
        values = struct.unpack(f'<{sample_count}h', samples)
        rms = (sum(v * v for v in values) / sample_count) ** 0.5
        
        now = time.time()
        
        if rms >= self.config.vad_threshold:
            # Speech detected
            if not self._speech_active:
                self._speech_active = True
                self._vad_buffer = bytearray()
                self._silence_start = None
                self._state = PipelineState.LISTENING
                emit_event(EventType.VOICE_STARTED, "voice_pipeline_vad", {
                    "rms": rms
                })
            
            self._vad_buffer.extend(samples)
            
        elif self._speech_active:
            # In speech, but below threshold
            self._vad_buffer.extend(samples)
            
            if self._silence_start is None:
                self._silence_start = now
            elif now - self._silence_start >= self.config.vad_silence_seconds:
                # End of utterance detected
                self._speech_active = False
                self._silence_start = None
                self._state = PipelineState.PROCESSING
                
                # Queue utterance for processing
                utterance = bytes(self._vad_buffer)
                self._vad_buffer = bytearray()
                
                try:
                    self._audio_queue.put_nowait(utterance)
                except queue.Full:
                    logger.warning("Audio queue full, dropping utterance")
                
                emit_event(EventType.VOICE_ENDED, "voice_pipeline_vad", {
                    "duration_seconds": len(utterance) / 2 / self.config.sample_rate,
                    "queued": True
                })
    
    # =========================================================================
    # PROCESSING LOOP
    # =========================================================================
    
    def _process_loop(self) -> None:
        """Process queued utterances with STT."""
        while self._running:
            try:
                utterance = self._audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            
            if not self._running:
                break
            
            self._process_utterance(utterance)
    
    def _process_utterance(self, audio_bytes: bytes) -> None:
        """Process a single utterance through STT."""
        start_time = time.perf_counter()
        
        try:
            self._metrics["total_utterances"] += 1
            
            # Transcribe
            text = self._stt.transcribe_bytes(audio_bytes, self.config.sample_rate)
            
            if not text or len(text.strip()) < self.config.min_utterance_length:
                self._metrics["failed_transcriptions"] += 1
                emit_event(EventType.TRANSCRIPTION_RECEIVED, "voice_pipeline", {
                    "text": text,
                    "success": False,
                    "reason": "too_short_or_empty"
                })
                return
            
            self._metrics["successful_transcriptions"] += 1
            latency_ms = (time.perf_counter() - start_time) * 1000
            self._metrics["total_latency_ms"] += latency_ms
            
            # Emit transcription event
            event = emit_event(EventType.TRANSCRIPTION_RECEIVED, "voice_pipeline", {
                "text": text,
                "success": True,
                "latency_ms": latency_ms,
                "audio_duration": len(audio_bytes) / 2 / self.config.sample_rate,
            })
            
            # Call callback if provided
            if self.on_transcription:
                try:
                    self.on_transcription(text)
                except Exception as e:
                    logger.error(f"Transcription callback error: {e}")
            
        except Exception as e:
            self._metrics["failed_transcriptions"] += 1
            logger.error(f"STT processing error: {e}")
            emit_event(EventType.ERROR_OCCURRED, "voice_pipeline", {
                "error": str(e),
                "phase": "stt_processing"
            })
    
    # =========================================================================
    # TTS OUTPUT
    # =========================================================================
    
    def speak(self, text: str, async_play: bool = False) -> None:
        """Speak text using TTS."""
        if not self._tts:
            logger.error("TTS not initialized")
            return
        
        self._state = PipelineState.SPEAKING
        
        emit_event(EventType.AUDIO_PLAYBACK_STARTED, "voice_pipeline", {
            "text": text[:50] + "..." if len(text) > 50 else text,
        })
        
        def _speak():
            try:
                self._tts.speak(text)
            except Exception as e:
                logger.error(f"TTS error: {e}")
                emit_event(EventType.ERROR_OCCURRED, "voice_pipeline", {
                    "error": str(e),
                    "phase": "tts"
                })
            finally:
                emit_event(EventType.AUDIO_PLAYBACK_ENDED, "voice_pipeline", {})
                with self._lock:
                    if self._state == PipelineState.SPEAKING:
                        self._state = PipelineState.LISTENING
        
        if async_play:
            threading.Thread(target=_speak, daemon=True).start()
        else:
            _speak()
    
    async def speak_async(self, text: str) -> None:
        """Speak text asynchronously (for async contexts)."""
        if not self._tts:
            return
        
        self._state = PipelineState.SPEAKING
        emit_event(EventType.AUDIO_PLAYBACK_STARTED, "voice_pipeline", {"text": text[:50]})
        
        try:
            await self._tts._speak_async(text, block=True)
        except Exception as e:
            logger.error(f"TTS error: {e}")
            emit_event(EventType.ERROR_OCCURRED, "voice_pipeline", {"error": str(e), "phase": "tts"})
        finally:
            emit_event(EventType.AUDIO_PLAYBACK_ENDED, "voice_pipeline", {})
            with self._lock:
                if self._state == PipelineState.SPEAKING:
                    self._state = PipelineState.LISTENING
    
    # =========================================================================
    # STATUS & METRICS
    # =========================================================================
    
    @property
    def state(self) -> PipelineState:
        return self._state
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline metrics."""
        metrics = self._metrics.copy()
        if metrics["successful_transcriptions"] > 0:
            metrics["avg_latency_ms"] = metrics["total_latency_ms"] / metrics["successful_transcriptions"]
        else:
            metrics["avg_latency_ms"] = 0
        metrics["state"] = self._state.value
        return metrics
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


# =========================================================================
# CONVENIENCE FUNCTIONS
# =========================================================================

def create_pipeline(
    stt_model: str = "base",
    tts_voice: str = "es-ES-AlvaroNeural",
    language: str = "es",
    on_transcription: Optional[Callable[[str], None]] = None
) -> VoicePipeline:
    """Create a voice pipeline with common defaults."""
    config = PipelineConfig(
        stt_model=stt_model,
        tts_voice=tts_voice,
        stt_language=language,
    )
    return VoicePipeline(config=config, on_transcription=on_transcription)


# =========================================================================
# TESTING
# =========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("VOICE PIPELINE TEST")
    print("=" * 60)
    
    received_texts = []
    
    def on_transcription(text: str):
        print(f"[PIPELINE] 📝 Transcribed: {text}")
        received_texts.append(text)
    
    pipeline = create_pipeline(
        stt_model="base",
        tts_voice="es-ES-AlvaroNeural",
        language="es",
        on_transcription=on_transcription
    )
    
    print("\nStarting pipeline... Speak into microphone.")
    print("Press Ctrl+C to stop.\n")
    
    try:
        pipeline.start()
        
        # Keep running
        while pipeline.is_running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\nStopping pipeline...")
    finally:
        pipeline.stop()
        
        print("\n" + "=" * 60)
        print("METRICS:")
        for k, v in pipeline.get_metrics().items():
            print(f"  {k}: {v}")
        print(f"  transcriptions: {received_texts}")
        print("=" * 60)