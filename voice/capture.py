"""
Voice Capture — Microphone audio capture using sounddevice (cross-platform, no FFmpeg needed).
"""

import sys
import queue
import threading
import numpy as np
import sounddevice as sd
from typing import Optional, Callable, Generator
from dataclasses import dataclass
from pathlib import Path

@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "int16"
    blocksize: int = 1024
    device: Optional[int] = None  # None = default


class VoiceCapture:
    """
    Real-time microphone capture with VAD (Voice Activity Detection).
    
    Usage:
        capture = VoiceCapture(config=AudioConfig())
        for chunk in capture.stream():
            process(chunk)
    """
    
    def __init__(
        self,
        config: Optional[AudioConfig] = None,
        on_audio_chunk: Optional[Callable[[bytes], None]] = None,
    ):
        self.config = config or AudioConfig()
        self.on_audio_chunk = on_audio_chunk
        self._running = False
        self._stream = None
        self._queue: queue.Queue = queue.Queue(maxsize=100)
        self._thread: Optional[threading.Thread] = None
        
    def _audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice for each audio block."""
        if status:
            print(f"[VoiceCapture] ⚠️ Status: {status}")
        if self._running:
            # Convert to bytes
            audio_bytes = indata.tobytes()
            try:
                self._queue.put_nowait(audio_bytes)
            except queue.Full:
                pass  # Drop if queue full
            if self.on_audio_chunk:
                self.on_audio_chunk(audio_bytes)
    
    def start(self) -> None:
        """Start capturing audio."""
        if self._running:
            return
            
        self._running = True
        
        # Find input device if not specified
        if self.config.device is None:
            devices = sd.query_devices()
            for i, d in enumerate(devices):
                if d['max_input_channels'] > 0 and 'micrófono' in d['name'].lower():
                    self.config.device = i
                    print(f"[VoiceCapture] 🎤 Using microphone: {d['name']} (device {i})")
                    break
            else:
                # Use default input
                self.config.device = sd.default.device[0]
                print(f"[VoiceCapture] 🎤 Using default input device: {self.config.device}")
        
        self._stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype=self.config.dtype,
            blocksize=self.config.blocksize,
            device=self.config.device,
            callback=self._audio_callback,
        )
        self._stream.start()
        print("[VoiceCapture] ✅ Started")
    
    def stop(self) -> None:
        """Stop capturing audio."""
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        print("[VoiceCapture] 🛑 Stopped")
    
    def stream(self) -> Generator[bytes, None, None]:
        """Generator that yields audio chunks."""
        self.start()
        try:
            while self._running:
                try:
                    chunk = self._queue.get(timeout=0.1)
                    yield chunk
                except queue.Empty:
                    continue
        finally:
            self.stop()
    
    def get_audio_buffer(self, duration_seconds: float) -> bytes:
        """Capture a fixed duration of audio and return as bytes."""
        self.start()
        try:
            chunks_needed = int(duration_seconds * self.config.sample_rate / self.config.blocksize)
            buffer = bytearray()
            for _ in range(chunks_needed):
                try:
                    chunk = self._queue.get(timeout=1.0)
                    buffer.extend(chunk)
                except queue.Empty:
                    break
            return bytes(buffer)
        finally:
            self.stop()
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


def list_microphones() -> list:
    """List available microphone devices."""
    devices = sd.query_devices()
    mics = []
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            mics.append({
                'index': i,
                'name': d['name'],
                'channels': d['max_input_channels'],
                'sample_rate': d['default_samplerate'],
            })
    return mics


if __name__ == "__main__":
    print("Available microphones:")
    for mic in list_microphones():
        print(f"  {mic['index']}: {mic['name']} ({mic['channels']}ch @ {mic['sample_rate']}Hz)")
    
    print("\nTesting 3-second capture...")
    with VoiceCapture() as capture:
        audio = capture.get_audio_buffer(3.0)
        print(f"Captured {len(audio)} bytes ({len(audio)/2/16000:.1f}s of int16 audio)")