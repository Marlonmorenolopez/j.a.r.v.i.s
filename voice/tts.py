"""
Voice TTS — Text-to-Speech using Edge TTS (local via Microsoft Edge engine).
"""

import asyncio
import sys
import os
import tempfile
from pathlib import Path
from typing import Optional, List, Callable
from dataclasses import dataclass
from enum import Enum

try:
    import edge_tts
except ImportError:
    edge_tts = None

class VoiceStyle(Enum):
    """Available voice styles for Edge TTS."""
    NEUTRAL = "neutral"
    CHEERFUL = "cheerful"
    SAD = "sad"
    ANGRY = "angry"
    FRIENDLY = "friendly"
    SHOUTING = "shouting"
    WHISPERING = "whispering"
    TERRIFIED = "terrified"
    UNFRIENDLY = "unfriendly"
    ASSISTANT = "assistant"
    NEWSCAST = "newscast"
    CUSTOMER_SERVICE = "customerservice"
    CHAT = "chat"
    EMPATHY = "empathy"
    ENTHUSIASTIC = "enthusiastic"


@dataclass
class TTSConfig:
    voice: str = "es-ES-AlvaroNeural"  # Default Spanish male voice
    rate: str = "+0%"                  # Speed: +50%, -50%, etc.
    volume: str = "+0%"                # Volume: +50%, -50%, etc.
    pitch: str = "+0Hz"                # Pitch: +50Hz, -50Hz, etc.
    style: Optional[VoiceStyle] = None # Voice style (requires compatible voice)
    
    # Common Spanish voices:
    # es-ES-AlvaroNeural (male)
    # es-ES-ElviraNeural (female)
    # es-ES-ArnauNeural (male)
    # es-ES-AbrilNeural (female)
    # es-MX-JorgeNeural (male, Mexican)
    # es-MX-DaliaNeural (female, Mexican)
    # en-US-GuyNeural (male, US)
    # en-US-JennyNeural (female, US)


class VoiceTTS:
    """
    Text-to-Speech using Edge TTS (Microsoft Edge neural voices).
    
    Usage:
        tts = VoiceTTS(TTSConfig(voice="es-ES-AlvaroNeural"))
        await tts.speak("Hola, ¿cómo estás?")
        # Or save to file:
        await tts.synthesize_to_file("Hola", "output.mp3")
    """
    
    # Voice cache to avoid re-downloading
    _voice_list_cache: Optional[List[dict]] = None
    
    def __init__(self, config: Optional[TTSConfig] = None):
        if edge_tts is None:
            raise RuntimeError("edge-tts not installed. Run: pip install edge-tts")
        
        self.config = config or TTSConfig()
        print(f"[VoiceTTS] 🔊 Initialized with voice: {self.config.voice}")
    
    @classmethod
    async def list_voices(cls) -> List[dict]:
        """Get all available Edge TTS voices."""
        if cls._voice_list_cache is None:
            cls._voice_list_cache = await edge_tts.list_voices()
        return cls._voice_list_cache
    
    @classmethod
    def find_voice(cls, language: str = "es", gender: str = "male") -> Optional[str]:
        """Find a suitable voice for language and gender."""
        # This is sync, so we can't call async list_voices directly
        # Return known good defaults
        voices = {
            ("es", "male"): "es-ES-AlvaroNeural",
            ("es", "female"): "es-ES-ElviraNeural",
            ("es-MX", "male"): "es-MX-JorgeNeural",
            ("es-MX", "female"): "es-MX-DaliaNeural",
            ("en", "male"): "en-US-GuyNeural",
            ("en", "female"): "en-US-JennyNeural",
            ("en-GB", "male"): "en-GB-RyanNeural",
            ("en-GB", "female"): "en-GB-SoniaNeural",
        }
        return voices.get((language, gender))
    
    async def synthesize(self, text: str) -> bytes:
        """Synthesize text to audio bytes (MP3)."""
        communicate = edge_tts.Communicate(
            text=text,
            voice=self.config.voice,
            rate=self.config.rate,
            volume=self.config.volume,
            pitch=self.config.pitch,
        )
        
        audio_chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])
        
        return b"".join(audio_chunks)
    
    async def synthesize_to_file(self, text: str, output_path: str) -> str:
        """Synthesize text to audio file."""
        communicate = edge_tts.Communicate(
            text=text,
            voice=self.config.voice,
            rate=self.config.rate,
            volume=self.config.volume,
            pitch=self.config.pitch,
        )
        
        await communicate.save(output_path)
        print(f"[VoiceTTS] 💾 Saved audio to: {output_path}")
        return output_path
    
    def speak(self, text: str, block: bool = True) -> None:
        """Synthesize and play text (blocking)."""
        asyncio.run(self._speak_async(text, block))
    
    async def _speak_async(self, text: str, block: bool = True) -> None:
        """Internal async speak method."""
        # Save to temp file and play
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            temp_path = f.name
        
        try:
            await self.synthesize_to_file(text, temp_path)
            
            # Play using system player
            if sys.platform == "win32":
                os.startfile(temp_path)
            elif sys.platform == "darwin":
                os.system(f"afplay '{temp_path}'")
            else:
                os.system(f"mpg123 '{temp_path}'")
            
            if block:
                # Wait for playback (rough estimate based on text length)
                await asyncio.sleep(len(text) * 0.08)
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
    
    async def speak_stream(self, text: str, on_chunk: Callable[[bytes], None]) -> None:
        """Stream audio chunks to callback (for real-time playback)."""
        communicate = edge_tts.Communicate(
            text=text,
            voice=self.config.voice,
            rate=self.config.rate,
            volume=self.config.volume,
            pitch=self.config.pitch,
        )
        
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                on_chunk(chunk["data"])


# Synchronous wrapper for easier integration
def speak(text: str, voice: str = "es-ES-AlvaroNeural", rate: str = "+0%") -> None:
    """Quick synchronous TTS."""
    tts = VoiceTTS(TTSConfig(voice=voice, rate=rate))
    tts.speak(text)


if __name__ == "__main__":
    # Quick test
    async def test():
        tts = VoiceTTS(TTSConfig(voice="es-ES-AlvaroNeural"))
        
        print("[VoiceTTS] Testing synthesis...")
        audio = await tts.synthesize("Hola, soy JARVIS. ¿En qué puedo ayudarte?")
        print(f"Generated {len(audio)} bytes of audio")
        
        # Save test file
        await tts.synthesize_to_file("Prueba de voz completada.", "test_output.mp3")
        
        # Test voice listing
        voices = await VoiceTTS.list_voices()
        spanish_voices = [v for v in voices if v["Locale"].startswith("es")]
        print(f"\nAvailable Spanish voices ({len(spanish_voices)}):")
        for v in spanish_voices[:5]:
            print(f"  {v['ShortName']} - {v['Gender']} - {v['Locale']}")
    
    asyncio.run(test())