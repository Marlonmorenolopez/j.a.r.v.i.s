"""
Voice STT — Speech-to-Text using Faster-Whisper (local, privacy-preserving).
"""

import sys
import os
from pathlib import Path
from typing import Optional, Generator, Dict, Any
from dataclasses import dataclass
import tempfile
import numpy as np

@dataclass
class STTConfig:
    model_size: str = "base"  # tiny, base, small, medium, large-v3
    device: str = "cpu"       # cpu, cuda
    compute_type: str = "int8"  # int8, int16, float16, float32
    language: str = "es"      # es, en, auto
    beam_size: int = 5
    vad_filter: bool = True
    vad_parameters: Optional[Dict[str, Any]] = None
    condition_on_previous_text: bool = False
    no_speech_threshold: float = 0.6
    log_prob_threshold: float = -1.0


class VoiceSTT:
    """
    Local Speech-to-Text using Faster-Whisper.
    
    Usage:
        stt = VoiceSTT(STTConfig(model_size="base", language="es"))
        for text in stt.transcribe_stream(audio_generator):
            print(text)
        # Or for single audio:
        text = stt.transcribe_file("audio.wav")
    """
    
    def __init__(self, config: Optional[STTConfig] = None):
        self.config = config or STTConfig()
        self._model = None
        self._init_model()
    
    def _init_model(self):
        """Initialize the Faster-Whisper model."""
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise RuntimeError("faster-whisper not installed. Run: pip install faster-whisper")
        
        print(f"[VoiceSTT] 🔄 Loading model: {self.config.model_size} ({self.config.device}, {self.config.compute_type})")
        
        # Auto-detect CUDA
        if self.config.device == "auto":
            import torch
            self.config.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.config.device == "cuda":
                self.config.compute_type = "float16"
            else:
                self.config.compute_type = "int8"
        
        try:
            self._model = WhisperModel(
                self.config.model_size,
                device=self.config.device,
                compute_type=self.config.compute_type,
            )
            print(f"[VoiceSTT] ✅ Model loaded on {self.config.device}")
        except Exception as e:
            # Fallback to CPU
            print(f"[VoiceSTT] ⚠️ Failed on {self.config.device}, falling back to CPU: {e}")
            self.config.device = "cpu"
            self.config.compute_type = "int8"
            self._model = WhisperModel(
                self.config.model_size,
                device="cpu",
                compute_type="int8",
            )
            print("[VoiceSTT] ✅ Model loaded on CPU (fallback)")
    
    def transcribe_file(self, audio_path: str) -> str:
        """Transcribe an audio file and return the text."""
        segments, info = self._model.transcribe(
            audio_path,
            language=self.config.language if self.config.language != "auto" else None,
            beam_size=self.config.beam_size,
            vad_filter=self.config.vad_filter,
            vad_parameters=self.config.vad_parameters or {
                "threshold": 0.5,
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 400,
            },
            condition_on_previous_text=self.config.condition_on_previous_text,
            log_prob_threshold=self.config.log_prob_threshold,
        )
        
        text = " ".join([seg.text.strip() for seg in segments]).strip()
        
        print(f"[VoiceSTT] 📝 Language: {info.language} (p={info.language_probability:.2f})")
        print(f"[VoiceSTT] 📝 Duration: {info.duration:.1f}s")
        print(f"[VoiceSTT] 📝 Transcription: {text[:100]}..." if len(text) > 100 else f"[VoiceSTT] 📝 Transcription: {text}")
        
        return text
    
    def transcribe_bytes(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        """Transcribe raw audio bytes (int16)."""
        # Save to temp WAV file
        import wave
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            with wave.open(temp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # int16 = 2 bytes
                wf.setframerate(sample_rate)
                wf.writeframes(audio_bytes)
        
        try:
            return self.transcribe_file(temp_path)
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
    
    def transcribe_stream(self, audio_chunks: Generator[bytes, None, None], 
                          chunk_duration: float = 3.0,
                          sample_rate: int = 16000) -> Generator[str, None, None]:
        """
        Transcribe a stream of audio chunks.
        
        Args:
            audio_chunks: Generator yielding raw int16 audio bytes
            chunk_duration: Seconds of audio to accumulate before transcribing
            sample_rate: Sample rate of incoming audio
            
        Yields:
            Transcribed text for each chunk
        """
        bytes_per_second = sample_rate * 2  # int16 = 2 bytes per sample
        target_bytes = int(chunk_duration * bytes_per_second)
        
        buffer = bytearray()
        
        for chunk in audio_chunks:
            buffer.extend(chunk)
            
            if len(buffer) >= target_bytes:
                # Transcribe this chunk
                audio_bytes = bytes(buffer[:target_bytes])
                buffer = buffer[target_bytes:]  # Keep remainder
                
                text = self.transcribe_bytes(audio_bytes, sample_rate)
                if text:
                    yield text


if __name__ == "__main__":
    # Test with a sample
    stt = VoiceSTT(STTConfig(model_size="base", language="es"))
    
    # Create a test WAV file
    import wave
    import numpy as np
    
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Generate a tone
    tone = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
        with wave.open(temp_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(tone.tobytes())
    
    try:
        result = stt.transcribe_file(temp_path)
        print(f"Test result: '{result}'")
    finally:
        os.unlink(temp_path)