"""
Voice-UI Integration — Connects VoicePipeline to P.I.P.E UI and Agent system.
"""

import threading
import time
from typing import Optional, Callable
from dataclasses import dataclass

from core.event_system import (
    EventBus, Event, EventType, EventPriority,
    get_event_bus, emit_event, subscribe_to_events
)
from voice.pipeline import VoicePipeline, PipelineConfig, create_pipeline
from agent.intent_router import get_intent_router
from agent.action_resolver import get_action_resolver
from agent.pipe_context import get_context_engine
from agent.executor import AgentExecutor


@dataclass
class VoiceUIConfig:
    """Configuration for voice-UI integration."""
    stt_model: str = "tiny"  # Use tiny for speed, base for accuracy
    tts_voice: str = "es-ES-AlvaroNeural"
    language: str = "es"
    auto_start: bool = True
    min_utterance_length: int = 3


class VoiceUIBridge:
    """
    Bridges VoicePipeline with P.I.P.E UI and execution system.
    
    Handles:
    - Voice pipeline lifecycle (start/stop)
    - UI state updates (LISTENING, PROCESSING, SPEAKING)
    - Transcription → Intent → Action execution
    - TTS playback coordination
    - Event system integration
    """
    
    def __init__(self, ui, config: Optional[VoiceUIConfig] = None):
        self.ui = ui
        self.config = config or VoiceUIConfig()
        
        # Core components
        self._pipeline: Optional[VoicePipeline] = None
        self._router = get_intent_router()
        self._resolver = get_action_resolver()
        self._context_engine = get_context_engine()
        self._executor = AgentExecutor()
        self._event_bus = get_event_bus()
        
        # State
        self._running = False
        self._processing_lock = threading.Lock()
        self._subscription_ids = []
        
        # Metrics
        self._stats = {
            "total_commands": 0,
            "successful": 0,
            "failed": 0,
        }
    
    def initialize(self) -> bool:
        """Initialize the voice pipeline and event subscriptions."""
        try:
            # Create pipeline
            pipeline_config = PipelineConfig(
                stt_model=self.config.stt_model,
                tts_voice=self.config.tts_voice,
                stt_language=self.config.language,
            )
            self._pipeline = VoicePipeline(
                config=pipeline_config,
                on_transcription=self._on_transcription_received
            )
            
            # Subscribe to pipeline events
            self._subscribe_events()
            
            # Subscribe to UI text commands (for fallback)
            self._subscribe_ui_commands()
            
            print("[VoiceUIBridge] ✅ Initialized")
            return True
            
        except Exception as e:
            print(f"[VoiceUIBridge] ❌ Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _subscribe_events(self) -> None:
        """Subscribe to voice pipeline events."""
        # Voice state events
        self._subscription_ids.append(
            subscribe_to_events([EventType.VOICE_STARTED], self._on_voice_started)
        )
        self._subscription_ids.append(
            subscribe_to_events([EventType.VOICE_ENDED], self._on_voice_ended)
        )
        self._subscription_ids.append(
            subscribe_to_events([EventType.TRANSCRIPTION_RECEIVED], self._on_transcription_event)
        )
        self._subscription_ids.append(
            subscribe_to_events([EventType.AUDIO_PLAYBACK_STARTED], self._on_audio_playback_started)
        )
        self._subscription_ids.append(
            subscribe_to_events([EventType.AUDIO_PLAYBACK_ENDED], self._on_audio_playback_ended)
        )
        
        # System events
        self._subscription_ids.append(
            subscribe_to_events([EventType.SYSTEM_SHUTDOWN], self._on_system_shutdown)
        )
    
    def _subscribe_ui_commands(self) -> None:
        """Set up UI text command callback."""
        # The UI already has on_text_command callback set to JarvisLive._on_text_command
        # We'll add our own handler that works in parallel
        original_callback = self.ui.on_text_command
        
        def combined_callback(text: str):
            # Call original if exists
            if original_callback:
                original_callback(text)
            # Also process through our pipeline
            self.process_text_command(text)
        
        self.ui.on_text_command = combined_callback
    
    def start(self) -> bool:
        """Start the voice pipeline."""
        if self._running:
            return True
        
        if not self._pipeline:
            if not self.initialize():
                return False
        
        success = self._pipeline.start()
        if success:
            self._running = True
            self.ui.set_state("LISTENING")
            self.ui.write_log("SYS: Voice pipeline active. Listening...")
            print("[VoiceUIBridge] 🎤 Started")
        return success
    
    def stop(self) -> None:
        """Stop the voice pipeline."""
        if not self._running:
            return
        
        self._running = False
        
        if self._pipeline:
            self._pipeline.stop()
        
        # Unsubscribe events
        for sub_id in self._subscription_ids:
            self._event_bus.unsubscribe(sub_id)
        self._subscription_ids.clear()
        
        self.ui.set_state("IDLE")
        self.ui.write_log("SYS: Voice pipeline stopped.")
        print("[VoiceUIBridge] 🛑 Stopped")
    
    # =========================================================================
    # EVENT HANDLERS
    # =========================================================================
    
    def _on_voice_started(self, event: Event) -> None:
        """Handle voice activity detected."""
        # Update UI to show listening state
        self.ui.set_state("LISTENING")
        self.ui.write_log("SYS: Voice detected...")
    
    def _on_voice_ended(self, event: Event) -> None:
        """Handle end of voice utterance."""
        self.ui.set_state("PROCESSING")
        self.ui.write_log("SYS: Processing...")
    
    def _on_transcription_event(self, event: Event) -> None:
        """Handle transcription received from pipeline."""
        payload = event.payload
        text = payload.get("text", "")
        success = payload.get("success", False)
        
        if success and text:
            self.ui.write_log(f"You: {text}")
        elif not success:
            self.ui.write_log(f"SYS: Could not understand ({payload.get('reason', 'unknown')})")
    
    def _on_audio_playback_started(self, event: Event) -> None:
        """Handle TTS playback started."""
        self.ui.set_state("SPEAKING")
        text_preview = event.payload.get("text", "")[:50]
        self.ui.write_log(f"P.I.P.E: {text_preview}...")
    
    def _on_audio_playback_ended(self, event: Event) -> None:
        """Handle TTS playback ended."""
        if not self.ui.muted:
            self.ui.set_state("LISTENING")
    
    def _on_system_shutdown(self, event: Event) -> None:
        """Handle system shutdown."""
        self.stop()
    
    # =========================================================================
    # COMMAND PROCESSING
    # =========================================================================
    
    def _on_transcription_received(self, text: str) -> None:
        """Callback from VoicePipeline when transcription is ready."""
        # This runs in the pipeline thread - process asynchronously
        threading.Thread(
            target=self._process_voice_command,
            args=(text,),
            daemon=True
        ).start()
    
    def process_text_command(self, text: str) -> None:
        """Process a text command (from UI input)."""
        threading.Thread(
            target=self._process_voice_command,
            args=(text,),
            daemon=True
        ).start()
    
    def _process_voice_command(self, text: str) -> None:
        """Process a voice/text command through intent → action → execution."""
        if not text or len(text.strip()) < self.config.min_utterance_length:
            return
        
        # Prevent concurrent processing
        if not self._processing_lock.acquire(blocking=False):
            print("[VoiceUIBridge] ⚠️ Already processing, skipping")
            return
        
        try:
            self._stats["total_commands"] += 1
            
            # 1. Classify intent
            intent = self._router.classify(text, autonomy_level=3)
            
            if not intent.capability_id:
                self.ui.write_log(f"SYS: No entiendo: \"{text}\"")
                self._speak_response("No entendí tu comando.")
                self._stats["failed"] += 1
                return
            
            # 2. Resolve action
            action_context = self._context_engine.get_action_context(intent.capability_id)
            action = self._resolver.resolve(intent, action_context)
            
            # 3. Execute action
            self.ui.write_log(f"SYS: Executing {action.tool_name}...")
            
            # Use the executor (runs in thread pool)
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self._execute_single_action(action)
                )
            finally:
                loop.close()
            
            # 4. Speak response
            response_text = str(result) if result else "Listo."
            self._speak_response(response_text)
            
            self._stats["successful"] += 1
            
        except Exception as e:
            self._stats["failed"] += 1
            error_msg = f"Error: {str(e)[:100]}"
            self.ui.write_log(f"ERR: {error_msg}")
            self._speak_response(f"Hubo un error: {error_msg}")
            import traceback
            traceback.print_exc()
        finally:
            self._processing_lock.release()
    
    def _speak_response(self, text: str) -> None:
        """Speak response via TTS."""
        if self._pipeline:
            self._pipeline.speak(text, async_play=True)
    
    async def _execute_single_action(self, action) -> str:
        """Execute a single action using the dispatcher."""
        from agent.action_dispatcher import dispatch_tool
        return dispatch_tool(
            action.tool_name,
            action.parameters,
            player=None,
            speak=None
        )
    
    # =========================================================================
    # STATUS
    # =========================================================================
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    def get_stats(self) -> dict:
        stats = self._stats.copy()
        if self._pipeline:
            stats.update(self._pipeline.get_metrics())
        return stats


# Convenience function for easy integration
def create_voice_bridge(ui, **kwargs) -> VoiceUIBridge:
    """Create a VoiceUIBridge with common defaults."""
    config = VoiceUIConfig(**kwargs)
    return VoiceUIBridge(ui, config)


if __name__ == "__main__":
    # Test bridge creation
    from ui import JarvisUI
    
    print("Testing VoiceUIBridge...")
    ui = JarvisUI("face.png")
    
    bridge = create_voice_bridge(ui, stt_model="tiny")
    
    if bridge.initialize():
        print("✅ Bridge initialized")
    else:
        print("❌ Bridge initialization failed")