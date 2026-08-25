"""Intent Router — Maps user input to structured intents for JARVIS.

This module provides deterministic intent classification for the Fast Path.
For the Agent Path, it provides structured intent output for the Planner.

Key design principles:
- Fast Path: Zero LLM calls, pure rule-based matching
- Agent Path: Rich intent objects with entities for Planner consumption
- Extensible: New intents registered via CapabilityRegistry
- Deterministic: Same input → same intent (for Fast Path)
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Pattern, Tuple

from core.capability_registry import get_registry

logger = logging.getLogger(__name__)


class IntentType(Enum):
    """High-level intent categories for routing."""
    SYSTEM_CONTROL = "system_control"      # open_app, volume, screenshot
    MEDIA_CONTROL = "media_control"        # youtube, spotify playback
    FILE_OPERATION = "file_operation"      # read, write, copy, delete, search
    WEB_ACTION = "web_action"              # search, navigate, scrape
    CODE_TASK = "code_task"                # execute, git, test, build
    VOICE_ACTION = "voice_action"          # stt, tts, wake word
    QUERY = "query"                        # time, weather, facts
    CONVERSATION = "conversation"          # chat, clarify, help
    AUTOMATION = "automation"              # agent_task, complex workflows
    UNKNOWN = "unknown"


class RoutingPath(Enum):
    """Execution path determined by intent classification."""
    FAST_PATH = "fast_path"       # Direct capability execution, no LLM
    AGENT_PATH = "agent_path"     # Requires Planner + Agent Coordinator
    CLARIFY = "clarify"           # Ambiguous, need user clarification


@dataclass(frozen=True)
class Entity:
    """Extracted entity from user input."""
    name: str
    value: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Intent:
    """Structured intent representation."""
    name: str                    # Canonical intent name (e.g., "OPEN_APP")
    type: IntentType             # High-level category
    path: RoutingPath            # Fast Path vs Agent Path
    confidence: float            # 0.0 - 1.0
    entities: Tuple[Entity, ...] = ()  # Extracted parameters
    raw_text: str = ""           # Original user input
    capability_id: str = ""      # Mapped capability ID (if any)
    requires_confirmation: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging/debugging."""
        return {
            "name": self.name,
            "type": self.type.value,
            "path": self.path.value,
            "confidence": self.confidence,
            "entities": [{"name": e.name, "value": e.value, "confidence": e.confidence} for e in self.entities],
            "raw_text": self.raw_text,
            "capability_id": self.capability_id,
            "requires_confirmation": self.requires_confirmation,
        }


# Pattern types for matching
PatternFn = Callable[[str], Optional[Dict[str, str]]]  # text -> extracted entities dict


@dataclass
class IntentPattern:
    """A pattern that matches an intent."""
    intent_name: str
    patterns: List[Pattern] = field(default_factory=list)
    pattern_fns: List[PatternFn] = field(default_factory=list)
    intent_type: IntentType = IntentType.UNKNOWN
    path: RoutingPath = RoutingPath.AGENT_PATH
    priority: int = 0  # Higher = checked first
    required_entities: List[str] = field(default_factory=list)
    entity_extractors: Dict[str, PatternFn] = field(default_factory=dict)


class IntentRouter:
    """Routes user input to structured intents.

    Two modes:
    1. Fast Path: Rule-based, deterministic, zero LLM
    2. Agent Path: Produces rich Intent objects for Planner
    """

    _instance: Optional[IntentRouter] = None
    _lock = threading.RLock()

    def __new__(cls) -> IntentRouter:
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
            self._patterns: List[IntentPattern] = []
            self._intent_to_pattern: Dict[str, IntentPattern] = {}
            self._capability_registry = get_registry()
            self._generation = 0
            self._initialized = True

    def register_pattern(self, pattern: IntentPattern) -> None:
        """Register an intent pattern. Higher priority checked first."""
        with self._lock:
            self._patterns.append(pattern)
            self._patterns.sort(key=lambda p: -p.priority)
            self._intent_to_pattern[pattern.intent_name] = pattern
            self._generation += 1
            logger.debug("Registered intent pattern: %s (priority=%d)", pattern.intent_name, pattern.priority)

    def register_simple(self,
                        intent_name: str,
                        regex_patterns: List[str],
                        intent_type: IntentType = IntentType.UNKNOWN,
                        path: RoutingPath = RoutingPath.AGENT_PATH,
                        priority: int = 0,
                        entity_extractors: Optional[Dict[str, PatternFn]] = None,
                        required_entities: Optional[List[str]] = None) -> None:
        """Convenience method to register simple regex-based patterns."""
        compiled = [re.compile(p, re.IGNORECASE) for p in regex_patterns]
        pattern = IntentPattern(
            intent_name=intent_name,
            patterns=compiled,
            intent_type=intent_type,
            path=path,
            priority=priority,
            entity_extractors=entity_extractors or {},
            required_entities=required_entities or [],
        )
        self.register_pattern(pattern)

    def _match_patterns(self, text: str) -> List[Tuple[IntentPattern, Dict[str, str]]]:
        """Match text against all patterns, return list of (pattern, extracted_entities)."""
        matches = []
        for pattern in self._patterns:
            # Try regex patterns first
            for regex in pattern.patterns:
                match = regex.search(text)
                if match:
                    entities = match.groupdict() if match.groupdict() else {}
                    # Apply custom extractors
                    for entity_name, extractor_fn in pattern.entity_extractors.items():
                        if entity_name not in entities:
                            extracted = extractor_fn(text)
                            if extracted:
                                entities.update(extracted)
                    matches.append((pattern, entities))
                    break  # First matching regex per pattern

            # Try pattern functions
            if not any(m[0] is pattern for m in matches):
                for fn in pattern.pattern_fns:
                    entities = fn(text)
                    if entities:
                        matches.append((pattern, entities))
                        break

        return matches

    def _resolve_capability(self, intent_name: str) -> Optional[str]:
        """Map intent name to capability ID via CapabilityRegistry."""
        return self._capability_registry._intent_to_capability.get(intent_name.lower())

    def classify(self, text: str, *, autonomy_level: int = 3) -> Intent:
        """Classify user input into an Intent.

        Args:
            text: User input (voice transcription or typed text)
            autonomy_level: Current autonomy level (0-5) for confirmation gating

        Returns:
            Intent with name, type, path, confidence, entities
        """
        text = text.strip()
        if not text:
            return Intent(
                name="EMPTY",
                type=IntentType.UNKNOWN,
                path=RoutingPath.CLARIFY,
                confidence=0.0,
                raw_text=text,
            )

        matches = self._match_patterns(text)

        if not matches:
            return Intent(
                name="UNKNOWN",
                type=IntentType.UNKNOWN,
                path=RoutingPath.CLARIFY,
                confidence=0.1,
                raw_text=text,
            )

        # Take highest priority match (patterns already sorted by priority)
        best_pattern, entities = matches[0]

        # Calculate confidence based on match quality
        confidence = self._calculate_confidence(best_pattern, text, entities)

        # Get capability mapping
        capability_id = self._resolve_capability(best_pattern.intent_name)

        # Determine confirmation requirement
        requires_confirmation = False
        if capability_id:
            cap = self._capability_registry.get(capability_id)
            if cap:
                requires_confirmation = cap.need_explicit_permission(autonomy_level)

        # Build entity objects
        entity_objects = tuple(
            Entity(name=k, value=v, confidence=confidence)
            for k, v in entities.items()
        )

        return Intent(
            name=best_pattern.intent_name,
            type=best_pattern.intent_type,
            path=best_pattern.path,
            confidence=confidence,
            entities=entity_objects,
            raw_text=text,
            capability_id=capability_id or "",
            requires_confirmation=requires_confirmation,
        )

    def _calculate_confidence(self, pattern: IntentPattern, text: str, entities: Dict[str, str]) -> float:
        """Calculate confidence score for a match."""
        base_confidence = 0.8

        # Boost for required entities present
        required_present = sum(1 for e in pattern.required_entities if e in entities)
        if pattern.required_entities:
            base_confidence += 0.15 * (required_present / len(pattern.required_entities))

        # Boost for exact phrase matches
        for regex in pattern.patterns:
            if regex.search(text) and regex.pattern.lower() in text.lower():
                base_confidence += 0.05
                break

        return min(1.0, base_confidence)

    def get_intent_pattern(self, intent_name: str) -> Optional[IntentPattern]:
        """Get registered pattern by intent name."""
        return self._intent_to_pattern.get(intent_name)

    def list_intents(self) -> List[str]:
        """List all registered intent names."""
        return list(self._intent_to_pattern.keys())

    def get_generation(self) -> int:
        """Return current router generation for cache invalidation."""
        return self._generation

    def clear(self) -> None:
        """Clear all patterns (mainly for testing)."""
        with self._lock:
            self._patterns.clear()
            self._intent_to_pattern.clear()
            self._generation += 1


# Global instance
_router: Optional[IntentRouter] = None


def get_intent_router() -> IntentRouter:
    """Get the global IntentRouter instance."""
    global _router
    if _router is None:
        _router = IntentRouter()
    return _router


# ============================================================================
# BUILT-IN ENTITY EXTRACTORS (adapted for JARVIS Spanish-first)
# ============================================================================

def extract_app_name(text: str) -> Optional[Dict[str, str]]:
    """Extract application name from 'abre X', 'inicia X', 'ejecuta X'."""
    patterns = [
        r'\b(?:abre|abrir|open|launch|iniciar|ejecuta|ejecutar)\s+([a-zA-Z0-9\s\-\.]+?)(?:\s+(?:por favor|please))?$',
        r'\b(?:abre|abrir|open|launch|iniciar|ejecuta|ejecutar)\s+([a-zA-Z0-9\s\-\.]+)',
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            return {"app_name": match.group(1).strip()}
    return None


def extract_media_query(text: str) -> Optional[Dict[str, str]]:
    """Extract search query from 'pon X en youtube', 'reproduce X'."""
    patterns = [
        r'(?:pon|reproduce|play|busca|search)\s+(.+?)\s+(?:en|on)\s+(?:youtube|spotify)',
        r'(?:youtube|spotify)\s+(?:pon|reproduce|play|busca|search)\s+(.+)',
        r'(?:pon|reproduce|play)\s+(.+)',
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            return {"query": match.group(1).strip()}
    return None


def extract_code_query(text: str) -> Optional[Dict[str, str]]:
    """Extract code description from 'escribe un script', 'crea código', etc."""
    patterns = [
        r'(?:escribe|escribir|write|crea|crear|create)\s+(?:un\s+)?(?:script|c[oó]digo|code|función|function|programa)\s+(?:en|para|in)\s+(.+)',
        r'(?:escribe|escribir|write|crea|crear|create)\s+(?:un\s+)?(?:script|c[oó]digo|code|función|function|programa)\s+(.+)',
        r'(?:c[oó]digo|code|programa|script|función|function)\s+(.+)',
        r'(?:crea|crear|create)\s+(?:una\s+)?(?:aplicaci[oó]n|app|proyecto|project)\s+(.+)',
        r'(?:crea|crear|create)\s+(?:un\s+)?(?:proyecto|project)\b',  # Just "crea un proyecto"
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            # If no capture group, use the whole matched text
            if match.groups():
                return {"query": match.group(1).strip()}
            else:
                # For "crea un proyecto", use the whole phrase as query
                return {"query": text.strip()}
    return None


def extract_file_path(text: str) -> Optional[Dict[str, str]]:
    """Extract file path from file operations."""
    patterns = [
        r'(?:archivo|file|documento)\s+["\']?([^"\']+\.(?:txt|py|js|json|md|csv|pdf|docx?|xlsx?))["\']?',
        r'["\']([^"\']+\.(?:txt|py|js|json|md|csv|pdf|docx?|xlsx?))["\']',
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            return {"file_path": match.group(1).strip()}
    return None


def extract_url(text: str) -> Optional[Dict[str, str]]:
    """Extract URL from navigate/open commands."""
    patterns = [
        r'(?:abre|open|navega|navigate|ve|go to)\s+(https?://[^\s]+)',
        r'(https?://[^\s]+)',
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            return {"url": match.group(1).strip()}
    return None


def extract_search_query(text: str) -> Optional[Dict[str, str]]:
    """Extract search query from 'busca X', 'search for X'."""
    patterns = [
        r'(?:busca|buscar|search|search for|googlea)\s+(.+)',
        r'(?:qué es|que es|what is|who is|quien es)\s+(.+)',
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            return {"query": match.group(1).strip()}
    return None


def extract_volume_direction(text: str) -> Optional[Dict[str, str]]:
    """Extract volume direction: up/down/mute."""
    if re.search(r'\b(sub[eí]|up|aumenta|m[aá]s|arriba)\b', text, re.IGNORECASE):
        return {"direction": "up"}
    if re.search(r'\b(baj[aé]|down|disminuye|menos|abajo)\b', text, re.IGNORECASE):
        return {"direction": "down"}
    if re.search(r'\b(silencia|mute|callar|quieto)\b', text, re.IGNORECASE):
        return {"direction": "mute"}
    return None


def extract_reminder_time(text: str) -> Optional[Dict[str, str]]:
    """Extract time from reminder commands."""
    patterns = [
        r'(?:a las|para las|at)\s+(\d{1,2}:\d{2})',
        r'(?:en|in)\s+(\d+)\s*(?:minutos?|mins?|horas?)',
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            return {"time": match.group(1).strip()}
    return None


def extract_weather_city(text: str) -> Optional[Dict[str, str]]:
    """Extract city from weather queries."""
    patterns = [
        r'(?:tiempo|clima|weather)\s+(?:en|in|de|of)\s+([a-zA-ZáéíóúÁÉÍÓÚñÑ\s]+)',
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            return {"city": match.group(1).strip()}
    return None


def extract_memory_content(text: str) -> Optional[Dict[str, str]]:
    """Extract content from memory save commands."""
    patterns = [
        r'(?:guarda|guardar|save|recuerda|recordar)\s+(?:esto|this|en\s+memoria|esto\s+en\s+memoria)\s+(.+)',
        r'(?:guarda|guardar|save|recuerda|recordar)\s+(.+)',
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            return {"content": match.group(1).strip()}
    return None


def extract_message_receiver(text: str) -> Optional[Dict[str, str]]:
    """Extract message receiver from 'envía un mensaje a Juan', etc."""
    patterns = [
        r'(?:env[ií]a|enviar|send|mandar)\s+(?:un\s+)?(?:mensaje|message)\s+(?:a|to)\s+(\w+)',
        r'(?:escribe|escribir|write)\s+(?:un\s+)?(?:mensaje|message)\s+(?:a|to)\s+(\w+)',
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            return {"receiver": match.group(1).strip()}
    return {"receiver": "unknown"}  # Default fallback


def extract_message_content(text: str) -> Optional[Dict[str, str]]:
    """Extract message content."""
    patterns = [
        r'(?:env[ií]a|enviar|send|mandar)\s+(?:un\s+)?(?:mensaje|message)\s+(?:a|to)\s+\w+\s+(?:que\s+diga|diciendo|contenido)\s+(.+)',
        r'(?:escribe|escribir|write)\s+(?:un\s+)?(?:mensaje|message)\s+(?:a|to)\s+\w+\s+(.+)',
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            return {"message": match.group(1).strip()}
    return {"message": "mensaje automático"}  # Default fallback


def extract_message_platform(text: str) -> Optional[Dict[str, str]]:
    """Extract platform from message."""
    patterns = [
        r'(?:por|via|en)\s+(whatsapp|telegram|discord|slack|email|sms)',
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            return {"platform": match.group(1).strip()}
    return {"platform": "whatsapp"}  # Default fallback


def extract_computer_action(text: str) -> Optional[Dict[str, str]]:
    """Extract computer control action from commands."""
    if re.search(r'\b(?:clic|click)\b', text, re.IGNORECASE):
        return {"action": "click"}
    if re.search(r'\b(?:escribe|type)\b', text, re.IGNORECASE):
        return {"action": "type"}
    if re.search(r'\b(?:tecla|hotkey)\b', text, re.IGNORECASE):
        return {"action": "hotkey"}
    if re.search(r'\b(?:scroll)\b', text, re.IGNORECASE):
        return {"action": "scroll"}
    if re.search(r'\b(?:captura|screenshot|pantallazo)\b', text, re.IGNORECASE):
        return {"action": "screenshot"}
    # Default fallback for general "controla la computadora"
    return {"action": "click"}  # Default to click for general control commands


# ============================================================================
# INITIALIZATION - Register built-in patterns for JARVIS's 24 capabilities
# ============================================================================

def initialize_intent_router() -> None:
    """Initialize router with built-in intent patterns for JARVIS capabilities."""
    router = get_intent_router()

    # Clear any existing (for re-initialization)
    router.clear()

    # ---- SYSTEM CONTROL (Fast Path) ----

    router.register_simple(
        intent_name="OPEN_APP",
        regex_patterns=[
            r'\b(?:abre|abrir|open|launch|iniciar|ejecuta|ejecutar)\s+(?:chrome|firefox|edge|notepad|calc|calculator|terminal|cmd|powershell|vscode|code|spotify|discord|steam|word|excel|powerpoint|outlook|teams|zoom|slack|whatsapp|telegram|vlc|photoshop|blender|unity|android|studio|docker|kubernetes|navegador|browser|explorador)\b',
            r'\b(?:abre|abrir|open|launch|iniciar|ejecuta|ejecutar)\s+(?:el\s+)?(?:navegador|browser|explorador)\b',
        ],
        intent_type=IntentType.SYSTEM_CONTROL,
        path=RoutingPath.FAST_PATH,
        priority=100,
        entity_extractors={"app_name": extract_app_name},
    )

    router.register_simple(
        intent_name="CLOSE_APP",
        regex_patterns=[
            r'\b(?:cierra|cerrar|close|mata|kill|termina|terminar)\s+\w+',
        ],
        intent_type=IntentType.SYSTEM_CONTROL,
        path=RoutingPath.FAST_PATH,
        priority=90,
        entity_extractors={"app_name": extract_app_name},
    )

    router.register_simple(
        intent_name="SYSTEM_VOLUME_UP",
        regex_patterns=[
            r'\b(?:sub[eí]|aumenta|m[aá]s|arriba)\s+(?:el\s+)?volumen',
            r'\bvolumen\s+(?:sub[eí]|aumenta|m[aá]s|arriba)',
        ],
        intent_type=IntentType.SYSTEM_CONTROL,
        path=RoutingPath.FAST_PATH,
        priority=100,
        entity_extractors={"direction": extract_volume_direction},
    )

    router.register_simple(
        intent_name="SYSTEM_VOLUME_DOWN",
        regex_patterns=[
            r'\b(baj[aé]|disminuye|menos|abajo)\s+(?:el\s+)?volumen',
            r'\bvolumen\s+(?:baj[aé]|disminuye|menos|abajo)',
        ],
        intent_type=IntentType.SYSTEM_CONTROL,
        path=RoutingPath.FAST_PATH,
        priority=100,
        entity_extractors={"direction": extract_volume_direction},
    )

    router.register_simple(
        intent_name="SYSTEM_MUTE",
        regex_patterns=[
            r'\b(?:silencia|mute|callar|quieto)\s+(?:el\s+)?volumen',
            r'\bvolumen\s+(?:silencia|mute|callar|quieto)',
            r'^silencia$',
        ],
        intent_type=IntentType.SYSTEM_CONTROL,
        path=RoutingPath.FAST_PATH,
        priority=100,
        entity_extractors={"direction": extract_volume_direction},
    )

    router.register_simple(
        intent_name="TAKE_SCREENSHOT",
        regex_patterns=[
            r'\b(?:captura|capturar|screenshot|pantallazo|foto\s+pantalla)\b',
        ],
        intent_type=IntentType.SYSTEM_CONTROL,
        path=RoutingPath.FAST_PATH,
        priority=100,
    )

    router.register_simple(
        intent_name="GET_TIME",
        regex_patterns=[
            r'\b(?:qu[eé]\s+hora|que\s+hora|hora|time|qu[eé]\s+hora\s+es)\b',
        ],
        intent_type=IntentType.QUERY,
        path=RoutingPath.FAST_PATH,
        priority=100,
    )

    router.register_simple(
        intent_name="OPEN_FILE",
        regex_patterns=[
            r'\b(?:abre|abrir|open)\s+(?:el\s+)?(?:archivo|file|documento)\b',
        ],
        intent_type=IntentType.FILE_OPERATION,
        path=RoutingPath.FAST_PATH,
        priority=90,
        entity_extractors={"file_path": extract_file_path},
    )

    router.register_simple(
        intent_name="CLOSE_WINDOW",
        regex_patterns=[
            r'\b(?:cierra|cerrar|close)\s+(?:la\s+)?(?:ventana|window)\b',
        ],
        intent_type=IntentType.SYSTEM_CONTROL,
        path=RoutingPath.FAST_PATH,
        priority=90,
    )

    # ---- MEDIA CONTROL (Fast Path) ----

    router.register_simple(
        intent_name="YOUTUBE_PLAY",
        regex_patterns=[
            r'\b(?:pon|reproduce|play|busca|search)\s+.+\s+(?:en|on)\s+youtube\b',
            r'\byoutube\s+(?:pon|reproduce|play|busca|search)\b',
            r'\b(?:pon|reproduce)\s+m[uú]sica\s+(?:en|on)?\s*youtube\b',
        ],
        intent_type=IntentType.MEDIA_CONTROL,
        path=RoutingPath.FAST_PATH,
        priority=100,
        entity_extractors={"query": extract_media_query},
    )

    router.register_simple(
        intent_name="YOUTUBE_PAUSE",
        regex_patterns=[
            r'\b(?:pausa|pausar|pause|para|para\s+la)\s+(?:m[uú]sica|video|youtube|reproducci[oó]n)\b',
            r'^pausa\s*youtube\b',
        ],
        intent_type=IntentType.MEDIA_CONTROL,
        path=RoutingPath.FAST_PATH,
        priority=100,
    )

    router.register_simple(
        intent_name="YOUTUBE_RESUME",
        regex_patterns=[
            r'\b(?:reanuda|reanudar|resume|continua|continuar)\s+(?:m[uú]sica|video|youtube|reproducci[oó]n)\b',
            r'^reanuda\s*youtube\b',
        ],
        intent_type=IntentType.MEDIA_CONTROL,
        path=RoutingPath.FAST_PATH,
        priority=100,
    )

    router.register_simple(
        intent_name="YOUTUBE_VOLUME",
        regex_patterns=[
            r'\bvolumen\s+youtube\b',
            r'\byoutube\s+volumen\b',
        ],
        intent_type=IntentType.MEDIA_CONTROL,
        path=RoutingPath.FAST_PATH,
        priority=90,
        entity_extractors={"direction": extract_volume_direction},
    )

    router.register_simple(
        intent_name="YOUTUBE_NEXT",
        regex_patterns=[
            r'\b(?:siguiente|next|salta|skip)\s+(?:video|canci[oó]n|youtube)\b',
            r'^siguiente\s*youtube\b',
        ],
        intent_type=IntentType.MEDIA_CONTROL,
        path=RoutingPath.FAST_PATH,
        priority=90,
    )

    router.register_simple(
        intent_name="YOUTUBE_PREVIOUS",
        regex_patterns=[
            r'\b(?:anterior|previous|atrás|back)\s+(?:video|canci[oó]n|youtube)\b',
            r'^anterior\s*youtube\b',
        ],
        intent_type=IntentType.MEDIA_CONTROL,
        path=RoutingPath.FAST_PATH,
        priority=90,
    )

    # ---- FILE OPERATIONS (Fast Path for simple, Agent Path for complex) ----

    router.register_simple(
        intent_name="FILE_READ",
        regex_patterns=[
            r'\b(?:lee|leer|read|muestra|mostrar)\s+(?:el\s+)?(?:archivo|file|documento)\b',
        ],
        intent_type=IntentType.FILE_OPERATION,
        path=RoutingPath.FAST_PATH,
        priority=90,
        entity_extractors={"file_path": extract_file_path},
    )

    router.register_simple(
        intent_name="FILE_WRITE",
        regex_patterns=[
            r'\b(?:escribe|escribir|write|crea|crear|crea\s+archivo)\s+(?:el\s+)?(?:archivo|file|documento)\b',
        ],
        intent_type=IntentType.FILE_OPERATION,
        path=RoutingPath.AGENT_PATH,
        priority=80,
        entity_extractors={"file_path": extract_file_path},
        required_entities=["file_path"],
    )

    router.register_simple(
        intent_name="FILE_LIST",
        regex_patterns=[
            r'\b(?:lista|listar|list|muestra|mostrar)\s+(?:archivos|files|carpeta|folder|directorio)\b',
        ],
        intent_type=IntentType.FILE_OPERATION,
        path=RoutingPath.FAST_PATH,
        priority=90,
    )

    # ---- WEB ACTIONS (Mixed) ----

    router.register_simple(
        intent_name="WEB_SEARCH",
        regex_patterns=[
            r'\b(?:busca|buscar|search|search for|googlea)\s+(.+)',
            r'\b(?:qu[eé]\s+es|que\s+es|what is|who is|qui[eé]n\s+es)\s+(.+)',
        ],
        intent_type=IntentType.WEB_ACTION,
        path=RoutingPath.FAST_PATH,
        priority=90,
        entity_extractors={"query": extract_search_query},
    )

    router.register_simple(
        intent_name="WEB_NAVIGATE",
        regex_patterns=[
            r'\b(?:abre|open|navega|navigate|ve|go to)\s+(https?://[^\s]+)',
            r'(https?://[^\s]+)',
        ],
        intent_type=IntentType.WEB_ACTION,
        path=RoutingPath.FAST_PATH,
        priority=95,
        entity_extractors={"url": extract_url},
    )

    # ---- QUERIES (Fast Path) ----

    router.register_simple(
        intent_name="WEATHER_REPORT",
        regex_patterns=[
            r'\b(?:tiempo|clima|weather|temperatura)\b',
        ],
        intent_type=IntentType.QUERY,
        path=RoutingPath.FAST_PATH,
        priority=90,
        entity_extractors={"city": extract_weather_city},
    )

    router.register_simple(
        intent_name="FLIGHT_SEARCH",
        regex_patterns=[
            r'\b(?:vuelos?|flights?|busca\s+vuelo)\b',
        ],
        intent_type=IntentType.QUERY,
        path=RoutingPath.AGENT_PATH,
        priority=80,
    )

    # ---- CODING (Agent Path) ----

    router.register_simple(
        intent_name="CODE_ASSISTANCE",
        regex_patterns=[
            r'\b(?:c[oó]digo|code|programa|script|función|function|clase|class|debug|error)\b',
            r'\b(?:escribe|escribir|write|crea|crear|create)\s+(?:un\s+)?(?:script|c[oó]digo|code|función|function|programa)\b',
        ],
        intent_type=IntentType.CODE_TASK,
        path=RoutingPath.AGENT_PATH,
        priority=80,
        entity_extractors={"query": extract_code_query},
    )

    router.register_simple(
        intent_name="PROJECT_DEVELOPMENT",
        regex_patterns=[
            r'\b(?:proyecto|project|crear?\s+(?:una\s+)?aplicaci[oó]n|crear?\s+(?:una\s+)?app|desarrollar|build)\b',
        ],
        intent_type=IntentType.CODE_TASK,
        path=RoutingPath.AGENT_PATH,
        priority=70,
        entity_extractors={"query": extract_code_query, "description": extract_code_query},
    )

    # ---- COMMUNICATION (Agent Path) ----

    router.register_simple(
        intent_name="SEND_MESSAGE",
        regex_patterns=[
            r'\b(?:env[ií]a|enviar|send|mandar)\s+(?:un\s+)?(?:mensaje|message)\b',
            r'\b(?:escribe|escribir|write)\s+(?:un\s+)?(?:mensaje|message)\b',
        ],
        intent_type=IntentType.CONVERSATION,
        path=RoutingPath.AGENT_PATH,
        priority=70,
        entity_extractors={"receiver": extract_message_receiver, "message": extract_message_content, "platform": extract_message_platform},
    )

    # ---- PERSONAL/MEMORY (Fast Path) ----

    router.register_simple(
        intent_name="MEMORY_SAVE",
        regex_patterns=[
            r'\b(?:guarda|guardar|save|recuerda|recordar)\s+(?:esto|this|en\s+memoria|esto\s+en\s+memoria)\b',
        ],
        intent_type=IntentType.CONVERSATION,
        path=RoutingPath.FAST_PATH,
        priority=80,
        entity_extractors={"content": extract_memory_content},
    )

    router.register_simple(
        intent_name="REMINDER",
        regex_patterns=[
            r'\b(?:recordatorio|reminder|alarma|alarm)\s+(?:para|a las|en)\b',
            r'\b(?:pon|ponme|set)\s+(?:un\s+)?(?:recordatorio|reminder|alarma|alarm)\b',
        ],
        intent_type=IntentType.SYSTEM_CONTROL,
        path=RoutingPath.AGENT_PATH,
        priority=70,
        entity_extractors={"time": extract_reminder_time},
    )

    # ---- AUTOMATION (Agent Path) ----

    router.register_simple(
        intent_name="AGENT_TASK",
        regex_patterns=[
            r'\b(?:ejecuta|ejecutar|run|haz|hacer|realiza|realizar)\s+(?:la\s+)?(?:tarea|task|plan|workflow)\b',
            r'\b(?:agente|agent)\s+(?:ejecuta|haz|run)\b',
        ],
        intent_type=IntentType.AUTOMATION,
        path=RoutingPath.AGENT_PATH,
        priority=70,
    )

    # ---- SYSTEM CONTROL (Extended) ----

    router.register_simple(
        intent_name="COMPUTER_SETTINGS",
        regex_patterns=[
            r'\b(?:configura|configuración|settings|ajustes)\b',
        ],
        intent_type=IntentType.SYSTEM_CONTROL,
        path=RoutingPath.AGENT_PATH,
        priority=60,
    )

    router.register_simple(
        intent_name="COMPUTER_CONTROL",
        regex_patterns=[
            r'\b(?:clic|click|escribe|type|tecla|hotkey|scroll|captura)\b',
            r'\b(?:controla|controlar|manejar)\s+(?:la\s+)?(?:computadora|computador|pc|ordenador)\b',
        ],
        intent_type=IntentType.SYSTEM_CONTROL,
        path=RoutingPath.AGENT_PATH,
        priority=60,
        entity_extractors={"action": extract_computer_action},
    )

    # ---- WEB ACTIONS (Extended) ----

    router.register_simple(
        intent_name="WEB_NAVIGATE",
        regex_patterns=[
            r'\b(?:abre|open|navega|navigate|ve|go to)\s+(https?://[^\s]+)',
            r'(https?://[^\s]+)',
        ],
        intent_type=IntentType.WEB_ACTION,
        path=RoutingPath.FAST_PATH,
        priority=95,
        entity_extractors={"url": extract_url},
    )

    # ---- FILE OPERATIONS (Extended) ----

    router.register_simple(
        intent_name="FILE_OPERATIONS",
        regex_patterns=[
            r'\b(?:archivo|file|carpeta|folder|directorio)\s+(?:crea|crear|create|borra|delete|elimina|move|mover|copia|copy)\b',
        ],
        intent_type=IntentType.FILE_OPERATION,
        path=RoutingPath.AGENT_PATH,
        priority=60,
    )

    # ---- DESKTOP MANAGEMENT ----

    router.register_simple(
        intent_name="DESKTOP_MANAGEMENT",
        regex_patterns=[
            r'\b(?:escritorio|desktop)\s+(?:organiza|organizar|limpia|clean|fondo|wallpaper)\b',
        ],
        intent_type=IntentType.SYSTEM_CONTROL,
        path=RoutingPath.AGENT_PATH,
        priority=60,
    )

    # ---- GAME UPDATES ----

    router.register_simple(
        intent_name="GAME_UPDATES",
        regex_patterns=[
            r'\b(?:actualiza|actualizar|update|juego|game|steam|epic)\s+(?:juego|game|update|actualiza)\b',
        ],
        intent_type=IntentType.MEDIA_CONTROL,
        path=RoutingPath.AGENT_PATH,
        priority=60,
    )

    # ---- SYSTEM SHUTDOWN (Always clarify/confirm) ----

    router.register_simple(
        intent_name="SYSTEM_SHUTDOWN",
        regex_patterns=[
            r'\b(?:apaga|apagar|shutdown|apaga\s+jarvis|cierra\s+jarvis|exit|salir)\b',
        ],
        intent_type=IntentType.SYSTEM_CONTROL,
        path=RoutingPath.CLARIFY,
        priority=50,
    )

    # ---- CONVERSATION / HELP ----

    router.register_simple(
        intent_name="HELP",
        regex_patterns=[
            r'\b(?:ayuda|help|qu[eé]\s+puedes|que\s+puedes|what\s+can\s+you)\b',
            r'^\s*(?:hola|hello|hi|buenos|buenas)\s*$',
        ],
        intent_type=IntentType.CONVERSATION,
        path=RoutingPath.CLARIFY,
        priority=50,
    )

    logger.info("IntentRouter initialized with %d patterns", len(router._patterns))


# Auto-initialize on import
initialize_intent_router()