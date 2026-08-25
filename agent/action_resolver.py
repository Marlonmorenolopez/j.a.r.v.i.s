"""Action Resolver — Converts intents to concrete executable actions for JARVIS.

This module sits between Intent Router and Capability/Tool execution.
It resolves an abstract intent into a specific tool call with parameters.

Key responsibilities:
- Intent → Capability mapping (via CapabilityRegistry)
- Capability → Tool selection (which tool from the capability's tool list)
- Parameter extraction from intent entities + context
- Parameter validation and default handling
- Action metadata for execution (timeout, retries, etc.)

This is deliberately separate from Intent Router to allow:
- Different resolution strategies per capability
- Context-aware parameter resolution
- Testing resolution logic independently
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from core.capability_registry import (
    Capability,
    RiskLevel,
    get_registry,
)
from agent.intent_router import (
    Intent,
    IntentType,
    RoutingPath,
    Entity,
    get_intent_router,
)

logger = logging.getLogger(__name__)


class ParameterSource(Enum):
    """Where a parameter value comes from."""
    ENTITY = "entity"           # From intent entities
    CONTEXT = "context"         # From execution context (session, user prefs)
    DEFAULT = "default"         # Hardcoded default
    PROMPT = "prompt"           # Need to ask user
    COMPUTED = "computed"       # Computed from other params


@dataclass(frozen=True)
class ParameterSpec:
    """Specification for a single action parameter."""
    name: str
    type: str                    # "string", "int", "float", "bool", "path", "url"
    required: bool = True
    source: ParameterSource = ParameterSource.ENTITY
    entity_name: str = ""        # Which entity maps to this param
    default: Any = None
    validator: Optional[Callable[[Any], bool]] = None
    description: str = ""
    choices: List[Any] = field(default_factory=list)  # For enum-like params


@dataclass(frozen=True)
class ActionSpec:
    """Concrete action specification ready for execution."""
    capability_id: str
    tool_name: str
    parameters: Dict[str, Any]
    intent: Intent
    timeout_ms: int = 30000
    retries: int = 0
    verification: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResolutionError(Exception):
    """Raised when action resolution fails."""
    def __init__(self, message: str, intent: Intent, missing_params: List[str] = None):
        super().__init__(message)
        self.intent = intent
        self.missing_params = missing_params or []


class ActionResolver:
    """Resolves intents to concrete action specifications."""

    _instance: Optional[ActionResolver] = None
    _lock = threading.RLock()

    def __new__(cls) -> ActionResolver:
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
            self._capability_registry = get_registry()
            self._intent_router = get_intent_router()
            self._tool_param_specs: Dict[str, List[ParameterSpec]] = {}  # capability_id -> param specs
            self._tool_selectors: Dict[str, Callable[[Intent, Dict], str]] = {}  # capability_id -> selector fn
            self._generation = 0
            self._initialized = True

    def register_tool_params(self, capability_id: str, params: List[ParameterSpec]) -> None:
        """Register parameter specifications for a capability."""
        with self._lock:
            self._tool_param_specs[capability_id] = params
            self._generation += 1
            logger.debug("Registered params for capability: %s (%d params)", capability_id, len(params))

    def register_tool_selector(self, capability_id: str, selector: Callable[[Intent, Dict], str]) -> None:
        """Register a custom tool selector for a capability with multiple tools."""
        with self._lock:
            self._tool_selectors[capability_id] = selector
            self._generation += 1

    def resolve(self, intent: Intent, context: Optional[Dict[str, Any]] = None) -> ActionSpec:
        """Resolve an intent to an executable action specification.

        Args:
            intent: The classified intent from IntentRouter
            context: Optional execution context (session data, user prefs, etc.)

        Returns:
            ActionSpec with tool_name, parameters, and execution metadata

        Raises:
            ResolutionError: If intent cannot be resolved (missing capability, params, etc.)
        """
        context = context or {}

        # 1. Get capability from intent
        capability_id = intent.capability_id
        if not capability_id:
            # Try to resolve from intent name via CapabilityRegistry
            capability_id = self._capability_registry._intent_to_capability.get(intent.name.lower())

        if not capability_id:
            raise ResolutionError(
                f"No capability mapped for intent: {intent.name}",
                intent,
            )

        capability = self._capability_registry.get(capability_id)
        if not capability:
            raise ResolutionError(
                f"Capability not registered: {capability_id}",
                intent,
            )

        # 2. Select tool (first tool by default, or use custom selector)
        tool_name = self._select_tool(capability, intent, context)

        # 3. Resolve parameters
        parameters = self._resolve_parameters(capability, tool_name, intent, context)

        # 4. Build verification plan
        verification = self._build_verification_plan(capability)

        # 5. Get timeout estimate
        timeout_ms = self._estimate_timeout(capability)

        return ActionSpec(
            capability_id=capability_id,
            tool_name=tool_name,
            parameters=parameters,
            intent=intent,
            timeout_ms=timeout_ms,
            retries=0 if capability.risk_level == RiskLevel.CRITICAL else 1,
            verification=verification,
            metadata={
                "domain": capability.domain,
                "risk_level": capability.risk_level.value,
                "latency_hint": capability.latency_hint,
                "local_or_remote": capability.local_or_remote.value,
                "fast_path_eligible": self._is_fast_path_eligible(capability, intent),
            },
        )

    def _select_tool(self, capability: Capability, intent: Intent, context: Dict) -> str:
        """Select which tool to use for this capability."""
        # Custom selector takes precedence
        if capability.id in self._tool_selectors:
            return self._tool_selectors[capability.id](intent, context)

        # Default: first tool in the capability's tool list
        if capability.tools:
            return capability.tools[0]

        raise ResolutionError(
            f"Capability {capability.id} has no tools defined",
            intent,
        )

    def _resolve_parameters(
        self,
        capability: Capability,
        tool_name: str,
        intent: Intent,
        context: Dict,
    ) -> Dict[str, Any]:
        """Resolve parameters for the selected tool."""
        param_specs = self._tool_param_specs.get(capability.id, [])
        if not param_specs:
            # No registered specs - try to infer from entities
            return self._infer_parameters_from_entities(intent, tool_name)

        parameters = {}
        missing_required = []

        for spec in param_specs:
            value = self._resolve_single_parameter(spec, intent, context)
            if value is not None:
                parameters[spec.name] = value
            elif spec.required:
                missing_required.append(spec.name)

        if missing_required:
            raise ResolutionError(
                f"Missing required parameters for {capability.id}: {missing_required}",
                intent,
                missing_required,
            )

        return parameters

    def _resolve_single_parameter(
        self,
        spec: ParameterSpec,
        intent: Intent,
        context: Dict,
    ) -> Any:
        """Resolve a single parameter value."""
        # 1. Try entity extraction
        if spec.source == ParameterSource.ENTITY and spec.entity_name:
            for entity in intent.entities:
                if entity.name == spec.entity_name:
                    return self._convert_value(entity.value, spec.type)

        # 2. Try context
        if spec.source == ParameterSource.CONTEXT and spec.entity_name:
            if spec.entity_name in context:
                return context[spec.entity_name]

        # 3. Try default
        if spec.default is not None:
            return spec.default

        # 4. Try entity name matching param name (fallback)
        if spec.source == ParameterSource.ENTITY:
            for entity in intent.entities:
                if entity.name == spec.name:
                    return self._convert_value(entity.value, spec.type)

        return None

    def _infer_parameters_from_entities(self, intent: Intent, tool_name: str) -> Dict[str, Any]:
        """Infer parameters from intent entities when no specs registered."""
        parameters = {}
        for entity in intent.entities:
            # Map common entity names to tool parameter names
            param_name = self._map_entity_to_param(entity.name, tool_name)
            if param_name:
                parameters[param_name] = entity.value
        return parameters

    def _map_entity_to_param(self, entity_name: str, tool_name: str) -> Optional[str]:
        """Map entity name to tool parameter name."""
        # Common mappings for JARVIS tools
        mappings = {
            "app_name": "app",
            "app": "app",
            "query": "query",
            "file_path": "path",
            "path": "path",
            "url": "url",
            "direction": "direction",
            "city": "city",
            "time": "time",
            "content": "content",
        }
        return mappings.get(entity_name, entity_name)

    def _convert_value(self, value: str, target_type: str) -> Any:
        """Convert string value to target type."""
        try:
            if target_type == "int":
                return int(value)
            elif target_type == "float":
                return float(value)
            elif target_type == "bool":
                return value.lower() in ("true", "yes", "sí", "si", "1", "on")
            elif target_type == "path":
                from pathlib import Path
                return str(Path(value).expanduser())
            else:
                return value
        except (ValueError, TypeError):
            return value

    def _build_verification_plan(self, capability: Capability) -> Dict[str, Any]:
        """Build verification plan from capability metadata."""
        return {
            "method": capability.verification_method,
            "cost": capability.verification_cost.value,
            "latency": capability.verification_latency,
        }

    def _estimate_timeout(self, capability: Capability) -> int:
        """Estimate timeout based on capability latency hint."""
        timeouts = {
            "0.5-2s": 10000,
            "1-2s": 10000,
            "0.1s": 5000,
            "0.2s": 5000,
            "0.2-0.5s": 5000,
            "0.3-1s": 5000,
            "0.5-1s": 10000,
            "1-3s": 15000,
            "2-5s": 20000,
            "2-8s": 30000,
            "2-10s": 30000,
            "3-10s": 30000,
            "5-30s": 60000,
            "5-60s": 120000,
            "30-120s": 180000,
            "30-300s": 360000,
            "variable": 120000,
            "0.1-0.3s": 5000,
            "0.1-2s": 10000,
            "0.2-1s": 5000,
            "unknown": 30000,
        }
        return timeouts.get(capability.verification_latency, 30000)

    def _is_fast_path_eligible(self, capability: Capability, intent: Intent) -> bool:
        """Check if this action is eligible for Fast Path execution."""
        # Fast Path requires:
        # 1. Capability is LOW or NONE risk
        # 2. No confirmation required at current autonomy level
        # 3. Intent path is FAST_PATH
        return (
            capability.risk_level in (RiskLevel.NONE, RiskLevel.LOW)
            and not intent.requires_confirmation
            and intent.path == RoutingPath.FAST_PATH
        )

    def get_generation(self) -> int:
        return self._generation

    def clear(self) -> None:
        with self._lock:
            self._tool_param_specs.clear()
            self._tool_selectors.clear()
            self._generation += 1


# Global instance
_resolver: Optional[ActionResolver] = None


def get_action_resolver() -> ActionResolver:
    """Get the global ActionResolver instance."""
    global _resolver
    if _resolver is None:
        _resolver = ActionResolver()
    return _resolver


# ============================================================================
# BUILT-IN PARAMETER SPECIFICATIONS FOR JARVIS CAPABILITIES
# ============================================================================

def initialize_action_resolver() -> None:
    """Initialize resolver with built-in parameter specifications for JARVIS."""
    resolver = get_action_resolver()
    resolver.clear()

    # OPEN_APP - parameter: app name
    resolver.register_tool_params("open_app", [
        ParameterSpec(
            name="app",
            type="string",
            required=True,
            source=ParameterSource.ENTITY,
            entity_name="app_name",
            description="Application name to launch",
        ),
    ])

    # SYSTEM_VOLUME_UP - no parameters (direction implied)
    resolver.register_tool_params("system_volume_up", [])

    # SYSTEM_VOLUME_DOWN - no parameters (direction implied)
    resolver.register_tool_params("system_volume_down", [])

    # SYSTEM_MUTE - no parameters (direction implied)
    resolver.register_tool_params("system_mute", [])

    # GET_TIME - no parameters needed
    resolver.register_tool_params("get_time", [])

    # YOUTUBE_PLAY - parameter: search query
    resolver.register_tool_params("youtube_play", [
        ParameterSpec(
            name="query",
            type="string",
            required=True,
            source=ParameterSource.ENTITY,
            entity_name="query",
            description="Search query for YouTube video",
        ),
    ])

    # YOUTUBE_PAUSE - no parameters needed
    resolver.register_tool_params("youtube_pause", [])

    # YOUTUBE_RESUME - no parameters needed
    resolver.register_tool_params("youtube_resume", [])

    # YOUTUBE_VOLUME - parameter: direction
    resolver.register_tool_params("youtube_volume", [
        ParameterSpec(
            name="direction",
            type="string",
            required=True,
            source=ParameterSource.ENTITY,
            entity_name="direction",
            choices=["up", "down", "mute"],
            description="Volume direction: up, down, or mute",
        ),
    ])

    # YOUTUBE_NEXT - no parameters needed
    resolver.register_tool_params("youtube_next", [])

    # YOUTUBE_PREVIOUS - no parameters needed
    resolver.register_tool_params("youtube_previous", [])

    # FILE_READ - parameters: path, offset, limit
    resolver.register_tool_params("file_read", [
        ParameterSpec(
            name="path",
            type="path",
            required=True,
            source=ParameterSource.ENTITY,
            entity_name="file_path",
            description="Path to file to read",
        ),
        ParameterSpec(
            name="offset",
            type="int",
            required=False,
            source=ParameterSource.DEFAULT,
            default=1,
            description="Line offset to start reading",
        ),
        ParameterSpec(
            name="limit",
            type="int",
            required=False,
            source=ParameterSource.DEFAULT,
            default=500,
            description="Maximum lines to read",
        ),
    ])

    # FILE_WRITE - parameters: path, content
    resolver.register_tool_params("file_write", [
        ParameterSpec(
            name="path",
            type="path",
            required=True,
            source=ParameterSource.ENTITY,
            entity_name="file_path",
            description="Path to file to write",
        ),
        ParameterSpec(
            name="content",
            type="string",
            required=True,
            source=ParameterSource.PROMPT,
            description="Content to write to file",
        ),
    ])

    # FILE_LIST - parameter: path (optional)
    resolver.register_tool_params("file_list", [
        ParameterSpec(
            name="path",
            type="path",
            required=False,
            source=ParameterSource.DEFAULT,
            default=".",
            description="Directory path to list",
        ),
    ])

    # WEB_SEARCH - parameter: query
    resolver.register_tool_params("web_search", [
        ParameterSpec(
            name="query",
            type="string",
            required=True,
            source=ParameterSource.ENTITY,
            entity_name="query",
            description="Search query",
        ),
        ParameterSpec(
            name="max_results",
            type="int",
            required=False,
            source=ParameterSource.DEFAULT,
            default=10,
            description="Maximum results to return",
        ),
    ])

    # WEB_NAVIGATE - parameter: URL
    resolver.register_tool_params("web_navigate", [
        ParameterSpec(
            name="url",
            type="url",
            required=True,
            source=ParameterSource.ENTITY,
            entity_name="url",
            description="URL to navigate to",
        ),
    ])

    # WEATHER_REPORT - parameter: city (optional, defaults to user location)
    resolver.register_tool_params("weather_report", [
        ParameterSpec(
            name="city",
            type="string",
            required=False,
            source=ParameterSource.ENTITY,
            entity_name="city",
            default="Bogotá",
            description="City for weather report",
        ),
    ])

    # FLIGHT_SEARCH - parameters: origin, destination, date
    resolver.register_tool_params("flight_search", [
        ParameterSpec(
            name="origin",
            type="string",
            required=True,
            source=ParameterSource.PROMPT,
            description="Origin city/airport",
        ),
        ParameterSpec(
            name="destination",
            type="string",
            required=True,
            source=ParameterSource.PROMPT,
            description="Destination city/airport",
        ),
        ParameterSpec(
            name="date",
            type="string",
            required=True,
            source=ParameterSource.PROMPT,
            description="Travel date (YYYY-MM-DD)",
        ),
    ])

    # CODE_ASSISTANCE - parameters: description, language, file_path
    resolver.register_tool_params("code_assistance", [
        ParameterSpec(
            name="description",
            type="string",
            required=True,
            source=ParameterSource.ENTITY,
            entity_name="query",
            description="Description of coding task",
        ),
        ParameterSpec(
            name="language",
            type="string",
            required=False,
            source=ParameterSource.DEFAULT,
            default="python",
            description="Programming language",
        ),
        ParameterSpec(
            name="file_path",
            type="path",
            required=False,
            source=ParameterSource.CONTEXT,
            entity_name="file_path",
            description="Optional file to edit",
        ),
    ])

    # PROJECT_DEVELOPMENT - parameter: description
    resolver.register_tool_params("project_development", [
        ParameterSpec(
            name="description",
            type="string",
            required=True,
            source=ParameterSource.ENTITY,
            entity_name="query",
            description="Project description",
        ),
    ])

    # AGENT_TASK - parameter: description
    resolver.register_tool_params("agent_task", [
        ParameterSpec(
            name="description",
            type="string",
            required=True,
            source=ParameterSource.ENTITY,
            entity_name="query",
            description="Task description for agent",
        ),
    ])

    # COMPUTER_CONTROL - parameter: action, coordinates, text
    resolver.register_tool_params("computer_control", [
        ParameterSpec(
            name="action",
            type="string",
            required=True,
            source=ParameterSource.ENTITY,
            entity_name="action",
            description="Action type: click, type, hotkey, scroll, screenshot",
        ),
        ParameterSpec(
            name="x",
            type="int",
            required=False,
            source=ParameterSource.DEFAULT,
            default=0,
            description="X coordinate for click",
        ),
        ParameterSpec(
            name="y",
            type="int",
            required=False,
            source=ParameterSource.DEFAULT,
            default=0,
            description="Y coordinate for click",
        ),
        ParameterSpec(
            name="text",
            type="string",
            required=False,
            source=ParameterSource.ENTITY,
            entity_name="text",
            description="Text to type",
        ),
        ParameterSpec(
            name="keys",
            type="string",
            required=False,
            source=ParameterSource.ENTITY,
            entity_name="keys",
            description="Hotkey combination (e.g., ctrl+c)",
        ),
    ])

    # SYSTEM_SHUTDOWN - no parameters (handled specially)
    resolver.register_tool_params("system_shutdown", [])

    # MEMORY_SAVE - parameter: content
    resolver.register_tool_params("memory_save", [
        ParameterSpec(
            name="content",
            type="string",
            required=True,
            source=ParameterSource.ENTITY,
            entity_name="content",
            description="Content to save in memory",
        ),
    ])

    # SEND_MESSAGE - parameters: receiver, message, platform
    resolver.register_tool_params("send_message", [
        ParameterSpec(
            name="receiver",
            type="string",
            required=True,
            source=ParameterSource.ENTITY,
            entity_name="receiver",
            description="Message recipient",
        ),
        ParameterSpec(
            name="message",
            type="string",
            required=True,
            source=ParameterSource.ENTITY,
            entity_name="message",
            description="Message content",
        ),
        ParameterSpec(
            name="platform",
            type="string",
            required=True,
            source=ParameterSource.ENTITY,
            entity_name="platform",
            description="Platform (WhatsApp, Telegram, etc.)",
        ),
    ])

    # REMINDER - parameters: date, time, message
    resolver.register_tool_params("reminder", [
        ParameterSpec(
            name="date",
            type="string",
            required=True,
            source=ParameterSource.ENTITY,
            entity_name="date",
            description="Date (YYYY-MM-DD)",
        ),
        ParameterSpec(
            name="time",
            type="string",
            required=True,
            source=ParameterSource.ENTITY,
            entity_name="time",
            description="Time (HH:MM)",
        ),
        ParameterSpec(
            name="message",
            type="string",
            required=True,
            source=ParameterSource.PROMPT,
            description="Reminder message",
        ),
    ])

    # GAME_UPDATES - parameters: action, platform, game_name
    resolver.register_tool_params("game_updates", [
        ParameterSpec(
            name="action",
            type="string",
            required=True,
            source=ParameterSource.DEFAULT,
            default="update",
            description="Action: update, install, list, download_status, schedule",
        ),
        ParameterSpec(
            name="platform",
            type="string",
            required=False,
            source=ParameterSource.DEFAULT,
            default="both",
            description="Platform: steam, epic, both",
        ),
        ParameterSpec(
            name="game_name",
            type="string",
            required=False,
            source=ParameterSource.ENTITY,
            entity_name="query",
            description="Game name",
        ),
    ])

    # DESKTOP_MANAGEMENT - parameter: action
    resolver.register_tool_params("desktop_management", [
        ParameterSpec(
            name="action",
            type="string",
            required=True,
            source=ParameterSource.ENTITY,
            entity_name="action",
            description="Action: wallpaper, organize, clean, list, task",
        ),
        ParameterSpec(
            name="path",
            type="path",
            required=False,
            source=ParameterSource.DEFAULT,
            default=".",
            description="Directory path",
        ),
    ])

    # COMPUTER_SETTINGS - parameter: action, value
    resolver.register_tool_params("computer_settings", [
        ParameterSpec(
            name="action",
            type="string",
            required=True,
            source=ParameterSource.ENTITY,
            entity_name="action",
            description="Settings action",
        ),
        ParameterSpec(
            name="value",
            type="string",
            required=False,
            source=ParameterSource.ENTITY,
            entity_name="value",
            description="Value to set",
        ),
    ])

    # SCREEN_CAPTURE - no parameters needed
    resolver.register_tool_params("screen_capture", [])

    # BROWSER_NAVIGATION - parameter: action, url/query
    resolver.register_tool_params("browser_navigation", [
        ParameterSpec(
            name="action",
            type="string",
            required=True,
            source=ParameterSource.ENTITY,
            entity_name="action",
            description="Action: go_to, search, click, type, scroll, get_text, press, close",
        ),
        ParameterSpec(
            name="url",
            type="url",
            required=False,
            source=ParameterSource.ENTITY,
            entity_name="url",
            description="URL for go_to action",
        ),
        ParameterSpec(
            name="query",
            type="string",
            required=False,
            source=ParameterSource.ENTITY,
            entity_name="query",
            description="Query for search action",
        ),
        ParameterSpec(
            name="text",
            type="string",
            required=False,
            source=ParameterSource.ENTITY,
            entity_name="text",
            description="Text for click/type actions",
        ),
    ])

    # FILE_OPERATIONS - parameter: action, path, content
    resolver.register_tool_params("file_operations", [
        ParameterSpec(
            name="action",
            type="string",
            required=True,
            source=ParameterSource.ENTITY,
            entity_name="action",
            description="Action: write, create_file, read, list, delete, move, copy, find, disk_usage",
        ),
        ParameterSpec(
            name="path",
            type="path",
            required=False,
            source=ParameterSource.ENTITY,
            entity_name="file_path",
            default=".",
            description="File/directory path",
        ),
        ParameterSpec(
            name="name",
            type="string",
            required=False,
            source=ParameterSource.ENTITY,
            entity_name="name",
            description="Filename",
        ),
        ParameterSpec(
            name="content",
            type="string",
            required=False,
            source=ParameterSource.PROMPT,
            description="Content for write/create_file",
        ),
    ])

    logger.info("ActionResolver initialized with %d capability parameter specs", len(resolver._tool_param_specs))


# Auto-initialize on import
initialize_action_resolver()