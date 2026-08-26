"""Context Engine — Conversation context management for P.I.P.E.

This module provides the context layer for the Agent Path:
- Session management (history, state, preferences)
- Context retrieval for Planner and capabilities
- Memory integration (short-term + long-term)
- Context window management

Distinct from context_compressor.py (token compression) and context_engine.py (abstract base).
This is the operational context engine for P.I.P.E runtime.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Deque

logger = logging.getLogger(__name__)


class ContextScope(Enum):
    """Scope of context data."""
    SESSION = "session"           # Current conversation session
    USER = "user"                 # User preferences, long-term memory
    SYSTEM = "system"             # System state, capabilities, config
    TEMPORARY = "temporary"       # Transient data for current task


@dataclass
class Message:
    """A single conversation message."""
    id: str
    role: str                    # "user", "assistant", "system", "tool"
    content: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_call_id: Optional[str] = None


@dataclass
class SessionState:
    """Current session state."""
    session_id: str
    created_at: float
    updated_at: float
    message_count: int = 0
    user_id: Optional[str] = None
    language: str = "es"
    autonomy_level: int = 3
    active_capability: Optional[str] = None
    pending_confirmation: Optional[Dict[str, Any]] = None
    context_data: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class ContextEntry:
    """A context entry with scope and TTL."""
    key: str
    value: Any
    scope: ContextScope
    created_at: float
    expires_at: Optional[float] = None
    tags: List[str] = field(default_factory=list)

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


class ContextEngine:
    """Operational context engine for P.I.P.E.

    Manages:
    - Conversation history (messages)
    - Session state
    - Scoped context data (session, user, system, temporary)
    - Context retrieval for Planner/ActionResolver
    - Memory integration points
    """

    _instance: Optional[ContextEngine] = None
    _lock = threading.RLock()

    def __new__(cls) -> ContextEngine:
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

            # Message history (bounded deque for memory efficiency)
            self._messages: Deque[Message] = deque(maxlen=1000)
            self._message_index: Dict[str, Message] = {}

            # Session state
            self._session: Optional[SessionState] = None

            # Scoped context storage
            self._context: Dict[ContextScope, Dict[str, ContextEntry]] = {
                ContextScope.SESSION: {},
                ContextScope.USER: {},
                ContextScope.SYSTEM: {},
                ContextScope.TEMPORARY: {},
            }

            # Configuration
            self._max_history_messages = 100
            self._session_ttl_seconds = 3600  # 1 hour default
            self._temporary_ttl_seconds = 300  # 5 minutes

            # Callbacks for external integrations
            self._on_session_start_callbacks: List[Callable[[SessionState], None]] = []
            self._on_session_end_callbacks: List[Callable[[SessionState], None]] = []
            self._on_message_added_callbacks: List[Callable[[Message], None]] = []

            self._generation = 0
            self._initialized = True

            logger.info("ContextEngine initialized")

    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================

    def start_session(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        language: str = "es",
        autonomy_level: int = 3,
    ) -> SessionState:
        """Start a new conversation session."""
        with self._lock:
            session_id = session_id or str(uuid.uuid4())[:8]
            now = time.time()

            self._session = SessionState(
                session_id=session_id,
                created_at=now,
                updated_at=now,
                user_id=user_id,
                language=language,
                autonomy_level=autonomy_level,
            )

            self._messages.clear()
            self._message_index.clear()
            self._clear_scope(ContextScope.SESSION)
            self._clear_scope(ContextScope.TEMPORARY)

            self._generation += 1

            # Fire callbacks
            for cb in self._on_session_start_callbacks:
                try:
                    cb(self._session)
                except Exception as e:
                    logger.warning("Session start callback failed: %s", e)

            logger.info("Session started: %s", session_id)
            return self._session

    def end_session(self, reason: str = "completed") -> Optional[SessionState]:
        """End the current session."""
        with self._lock:
            if not self._session:
                return None

            session = self._session
            session.context_data["end_reason"] = reason
            session.updated_at = time.time()

            # Fire callbacks
            for cb in self._on_session_end_callbacks:
                try:
                    cb(session)
                except Exception as e:
                    logger.warning("Session end callback failed: %s", e)

            self._session = None
            self._generation += 1

            logger.info("Session ended: %s (reason: %s)", session.session_id, reason)
            return session

    def get_session(self) -> Optional[SessionState]:
        """Get current session state."""
        return self._session

    def update_session(self, **kwargs) -> None:
        """Update session fields."""
        with self._lock:
            if not self._session:
                return
            for key, value in kwargs.items():
                if hasattr(self._session, key):
                    setattr(self._session, key, value)
            self._session.updated_at = time.time()
            self._generation += 1

    # =========================================================================
    # MESSAGE HISTORY
    # =========================================================================

    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_call_id: Optional[str] = None,
    ) -> Message:
        """Add a message to history."""
        with self._lock:
            if not self._session:
                self.start_session()

            msg = Message(
                id=str(uuid.uuid4())[:8],
                role=role,
                content=content,
                timestamp=time.time(),
                metadata=metadata or {},
                tool_calls=tool_calls or [],
                tool_call_id=tool_call_id,
            )

            self._messages.append(msg)
            self._message_index[msg.id] = msg
            self._session.message_count += 1
            self._session.updated_at = time.time()
            self._generation += 1

            # Fire callbacks
            for cb in self._on_message_added_callbacks:
                try:
                    cb(msg)
                except Exception as e:
                    logger.warning("Message callback failed: %s", e)

            return msg

    def get_messages(
        self,
        limit: Optional[int] = None,
        role: Optional[str] = None,
        since: Optional[float] = None,
    ) -> List[Message]:
        """Get messages from history with optional filters."""
        with self._lock:
            messages = list(self._messages)

            if role:
                messages = [m for m in messages if m.role == role]
            if since:
                messages = [m for m in messages if m.timestamp >= since]
            if limit:
                messages = messages[-limit:]

            return messages

    def get_recent_context(self, max_messages: int = 20) -> List[Dict[str, Any]]:
        """Get recent messages formatted for LLM context."""
        messages = self.get_messages(limit=max_messages)
        return [
            {
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp,
                "metadata": m.metadata,
            }
            for m in messages
        ]

    def get_last_user_message(self) -> Optional[Message]:
        """Get the most recent user message."""
        for msg in reversed(self._messages):
            if msg.role == "user":
                return msg
        return None

    def get_last_assistant_message(self) -> Optional[Message]:
        """Get the most recent assistant message."""
        for msg in reversed(self._messages):
            if msg.role == "assistant":
                return msg
        return None

    def clear_history(self) -> None:
        """Clear message history (keep session)."""
        with self._lock:
            self._messages.clear()
            self._message_index.clear()
            self._generation += 1

    # =========================================================================
    # SCOPED CONTEXT STORAGE
    # =========================================================================

    def set_context(
        self,
        key: str,
        value: Any,
        scope: ContextScope = ContextScope.SESSION,
        ttl_seconds: Optional[float] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Store a context value with scope and optional TTL."""
        with self._lock:
            now = time.time()
            expires_at = None
            if ttl_seconds is not None:
                expires_at = now + ttl_seconds
            elif scope == ContextScope.TEMPORARY:
                expires_at = now + self._temporary_ttl_seconds

            entry = ContextEntry(
                key=key,
                value=value,
                scope=scope,
                created_at=now,
                expires_at=expires_at,
                tags=tags or [],
            )

            self._context[scope][key] = entry
            self._generation += 1

    def get_context(
        self,
        key: str,
        scope: Optional[ContextScope] = None,
        default: Any = None,
    ) -> Any:
        """Get a context value, checking scopes in priority order."""
        with self._lock:
            scopes_to_check = [scope] if scope else [
                ContextScope.TEMPORARY,
                ContextScope.SESSION,
                ContextScope.USER,
                ContextScope.SYSTEM,
            ]

            for s in scopes_to_check:
                entry = self._context[s].get(key)
                if entry and not entry.is_expired():
                    return entry.value

            return default

    def get_all_context(self, scope: Optional[ContextScope] = None) -> Dict[str, Any]:
        """Get all non-expired context for a scope (or all scopes)."""
        with self._lock:
            result = {}
            scopes = [scope] if scope else list(ContextScope)

            for s in scopes:
                for key, entry in self._context[s].items():
                    if not entry.is_expired():
                        result[key] = entry.value

            return result

    def delete_context(self, key: str, scope: Optional[ContextScope] = None) -> bool:
        """Delete a context key from specified scope or all scopes."""
        with self._lock:
            scopes = [scope] if scope else list(ContextScope)
            deleted = False

            for s in scopes:
                if key in self._context[s]:
                    del self._context[s][key]
                    deleted = True

            if deleted:
                self._generation += 1

            return deleted

    def _clear_scope(self, scope: ContextScope) -> None:
        """Clear all entries in a scope."""
        self._context[scope].clear()

    def cleanup_expired(self) -> int:
        """Remove expired entries from all scopes. Returns count removed."""
        with self._lock:
            removed = 0
            for scope in ContextScope:
                expired_keys = [
                    k for k, e in self._context[scope].items()
                    if e.is_expired()
                ]
                for k in expired_keys:
                    del self._context[scope][k]
                    removed += 1

            if removed:
                self._generation += 1

            return removed

    # =========================================================================
    # CONTEXT FOR PLANNER / AGENT PATH
    # =========================================================================

    def get_planner_context(self) -> Dict[str, Any]:
        """Get context package for Planner (Agent Path)."""
        with self._lock:
            session = self._session
            recent_messages = self.get_recent_context(max_messages=10)

            return {
                "session": asdict(session) if session else None,
                "recent_messages": recent_messages,
                "session_context": self.get_all_context(ContextScope.SESSION),
                "user_context": self.get_all_context(ContextScope.USER),
                "system_context": self.get_all_context(ContextScope.SYSTEM),
                "temporary_context": self.get_all_context(ContextScope.TEMPORARY),
                "capabilities_summary": self._get_capabilities_summary(),
                "timestamp": time.time(),
            }

    def get_action_context(self, capability_id: str) -> Dict[str, Any]:
        """Get context package for ActionResolver/Capability execution."""
        with self._lock:
            session = self._session
            last_user_msg = self.get_last_user_message()

            return {
                "session_id": session.session_id if session else None,
                "user_id": session.user_id if session else None,
                "language": session.language if session else "es",
                "autonomy_level": session.autonomy_level if session else 3,
                "last_user_input": last_user_msg.content if last_user_msg else None,
                "last_user_metadata": last_user_msg.metadata if last_user_msg else {},
                "capability_context": self.get_all_context(ContextScope.TEMPORARY),
                "session_context": self.get_all_context(ContextScope.SESSION),
                "system_context": self.get_all_context(ContextScope.SYSTEM),
            }

    def _get_capabilities_summary(self) -> List[Dict[str, Any]]:
        """Get summary of available capabilities for Planner."""
        try:
            from core.capability_registry import get_registry
            registry = get_registry()
            caps = registry.get_all()
            return [
                {
                    "id": cap.id,
                    "name_es": cap.name_es,
                    "domain": cap.domain,
                    "risk_level": cap.risk_level.value,
                    "tools": cap.tools,
                    "latency_hint": cap.latency_hint,
                }
                for cap in caps
            ]
        except Exception:
            return []

    # =========================================================================
    # MEMORY INTEGRATION
    # =========================================================================

    def store_memory(self, key: str, value: Any, tags: Optional[List[str]] = None) -> None:
        """Store in long-term user memory (USER scope, no TTL)."""
        self.set_context(key, value, scope=ContextScope.USER, tags=tags or ["memory"])

    def recall_memory(self, key: str, default: Any = None) -> Any:
        """Recall from long-term user memory."""
        return self.get_context(key, scope=ContextScope.USER, default=default)

    def search_memory(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search memory by query (simple text match)."""
        with self._lock:
            results = []
            user_ctx = self._context[ContextScope.USER]

            for key, entry in user_ctx.items():
                if query.lower() in str(entry.value).lower():
                    results.append({
                        "key": key,
                        "value": entry.value,
                        "tags": entry.tags,
                        "created_at": entry.created_at,
                    })

            results.sort(key=lambda x: x["created_at"], reverse=True)
            return results[:limit]

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    def export_session(self) -> Dict[str, Any]:
        """Export current session to dictionary (for persistence)."""
        with self._lock:
            return {
                "session": asdict(self._session) if self._session else None,
                "messages": [
                    {
                        "id": m.id,
                        "role": m.role,
                        "content": m.content,
                        "timestamp": m.timestamp,
                        "metadata": m.metadata,
                        "tool_calls": m.tool_calls,
                        "tool_call_id": m.tool_call_id,
                    }
                    for m in self._messages
                ],
                "context": {
                    scope.value: {k: {
                        "key": e.key,
                        "value": e.value,
                        "created_at": e.created_at,
                        "expires_at": e.expires_at,
                        "tags": e.tags,
                    } for k, e in v.items() if not e.is_expired()}
                    for scope, v in self._context.items()
                },
                "generation": self._generation,
            }

    def import_session(self, data: Dict[str, Any]) -> None:
        """Import session from dictionary."""
        with self._lock:
            if data.get("session"):
                s = data["session"]
                self._session = SessionState(
                    session_id=s["session_id"],
                    created_at=s["created_at"],
                    updated_at=s["updated_at"],
                    message_count=s["message_count"],
                    user_id=s.get("user_id"),
                    language=s.get("language", "es"),
                    autonomy_level=s.get("autonomy_level", 3),
                    active_capability=s.get("active_capability"),
                    pending_confirmation=s.get("pending_confirmation"),
                    context_data=s.get("context_data", {}),
                    tags=s.get("tags", []),
                )

            self._messages.clear()
            self._message_index.clear()
            for m in data.get("messages", []):
                msg = Message(
                    id=m["id"],
                    role=m["role"],
                    content=m["content"],
                    timestamp=m["timestamp"],
                    metadata=m.get("metadata", {}),
                    tool_calls=m.get("tool_calls", []),
                    tool_call_id=m.get("tool_call_id"),
                )
                self._messages.append(msg)
                self._message_index[msg.id] = msg

            for scope_str, entries in data.get("context", {}).items():
                scope = ContextScope(scope_str)
                for k, e in entries.items():
                    entry = ContextEntry(
                        key=e["key"],
                        value=e["value"],
                        scope=scope,
                        created_at=e["created_at"],
                        expires_at=e.get("expires_at"),
                        tags=e.get("tags", []),
                    )
                    if not entry.is_expired():
                        self._context[scope][k] = entry

            self._generation = data.get("generation", 0)
            logger.info("Session imported: %s", self._session.session_id if self._session else "none")

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def on_session_start(self, callback: Callable[[SessionState], None]) -> None:
        """Register session start callback."""
        self._on_session_start_callbacks.append(callback)

    def on_session_end(self, callback: Callable[[SessionState], None]) -> None:
        """Register session end callback."""
        self._on_session_end_callbacks.append(callback)

    def on_message_added(self, callback: Callable[[Message], None]) -> None:
        """Register message added callback."""
        self._on_message_added_callbacks.append(callback)

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        with self._lock:
            session = self._session
            context_counts = {
                scope.value: len([e for e in entries.values() if not e.is_expired()])
                for scope, entries in self._context.items()
            }

            return {
                "session_id": session.session_id if session else None,
                "session_active": session is not None,
                "message_count": len(self._messages),
                "context_counts": context_counts,
                "generation": self._generation,
            }


# Global instance
_engine: Optional[ContextEngine] = None


def get_context_engine() -> ContextEngine:
    """Get the global ContextEngine instance."""
    global _engine
    if _engine is None:
        _engine = ContextEngine()
    return _engine


def initialize_context_engine() -> None:
    """Initialize context engine (explicit initialization)."""
    get_context_engine()
    logger.info("ContextEngine explicitly initialized")