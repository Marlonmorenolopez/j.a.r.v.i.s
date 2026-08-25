"""Permission Manager — Manages user permissions and consent for JARVIS.

This module handles:
- Permission requests for capabilities requiring confirmation
- Consent tracking (granted/denied/pending)
- Autonomy level integration
- Persistent permission storage
- Session-scoped temporary permissions
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from core.capability_registry import (
    Capability,
    RiskLevel,
    get_registry,
)

logger = logging.getLogger(__name__)


class PermissionStatus(Enum):
    """Status of a permission request."""
    PENDING = "pending"
    GRANTED = "granted"
    DENIED = "denied"
    EXPIRED = "expired"


class PermissionScope(Enum):
    """Scope of a permission grant."""
    SESSION = "session"           # Valid for current session only
    PERMANENT = "permanent"       # Persisted across sessions
    ONESHOT = "oneshot"           # Valid for single use only


@dataclass
class PermissionRequest:
    """A request for permission to execute a capability."""
    id: str
    capability_id: str
    capability_name: str
    risk_level: RiskLevel
    reason: str
    requested_at: float
    scope: PermissionScope = PermissionScope.SESSION
    status: PermissionStatus = PermissionStatus.PENDING
    decided_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PermissionGrant:
    """A granted permission."""
    capability_id: str
    granted_at: float
    scope: PermissionScope
    expires_at: Optional[float] = None
    conditions: Dict[str, Any] = field(default_factory=dict)
    source: str = "user"  # "user", "policy", "default"


class PermissionManager:
    """Manages permissions and consent for JARVIS capabilities.

    Integrates with CapabilityRegistry for risk-based gating.
    Provides both synchronous (blocking) and async (callback) APIs.
    """

    _instance: Optional[PermissionManager] = None
    _lock = threading.RLock()

    def __new__(cls) -> PermissionManager:
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

            self._registry = get_registry()
            self._pending_requests: Dict[str, PermissionRequest] = {}
            self._grants: Dict[str, PermissionGrant] = {}
            self._denials: Dict[str, float] = {}  # capability_id -> timestamp
            self._callbacks: Dict[str, List[Callable[[PermissionRequest], None]]] = {}
            self._generation = 0

            # Persistence
            self._persistence_path = Path(__file__).resolve().parent.parent / "config" / "permissions.json"
            self._load_persistent_grants()

            self._initialized = True
            logger.info("PermissionManager initialized")

    # =========================================================================
    # PERMISSION CHECKING (Core API)
    # =========================================================================

    def check_permission(
        self,
        capability_id: str,
        autonomy_level: int = 3,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Check if a capability can execute without explicit permission.

        Returns True if:
        - Capability doesn't need explicit permission at this autonomy level
        - Permission was previously granted (and not expired)
        - Capability is LOW/NONE risk and autonomy >= 2

        Returns False if explicit permission is needed.
        """
        capability = self._registry.get(capability_id)
        if not capability:
            logger.warning("Permission check for unknown capability: %s", capability_id)
            return False

        # CRITICAL always needs permission
        if capability.risk_level == RiskLevel.CRITICAL:
            return False

        # Check if capability needs explicit permission at this autonomy level
        if not capability.need_explicit_permission(autonomy_level):
            return True

        # Check for existing grant
        grant = self._grants.get(capability_id)
        if grant and not self._is_grant_expired(grant):
            return True

        # Check for recent denial (don't re-prompt immediately)
        if capability_id in self._denials:
            if time.time() - self._denials[capability_id] < 300:  # 5 min cooldown
                return False

        return False

    def request_permission(
        self,
        capability_id: str,
        reason: str = "",
        scope: PermissionScope = PermissionScope.SESSION,
        autonomy_level: int = 3,
        callback: Optional[Callable[[PermissionRequest], None]] = None,
    ) -> PermissionRequest:
        """Request permission for a capability.

        Can be used synchronously (blocking) or asynchronously (with callback).
        """
        capability = self._registry.get(capability_id)
        if not capability:
            raise ValueError(f"Unknown capability: {capability_id}")

        import uuid
        request_id = str(uuid.uuid4())[:8]

        request = PermissionRequest(
            id=request_id,
            capability_id=capability_id,
            capability_name=capability.name_es,
            risk_level=capability.risk_level,
            reason=reason or f"Execute {capability.name_es}",
            requested_at=time.time(),
            scope=scope,
        )

        with self._lock:
            self._pending_requests[request_id] = request
            if callback:
                self._callbacks.setdefault(request_id, []).append(callback)
            self._generation += 1

        logger.info("Permission requested: %s (%s)", capability_id, request_id)
        return request

    def grant_permission(
        self,
        request_id: str,
        scope: Optional[PermissionScope] = None,
        conditions: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Grant a pending permission request."""
        with self._lock:
            request = self._pending_requests.get(request_id)
            if not request:
                logger.warning("Grant failed: request %s not found", request_id)
                return False

            if request.status != PermissionStatus.PENDING:
                logger.warning("Grant failed: request %s already %s", request_id, request.status.value)
                return False

            request.status = PermissionStatus.GRANTED
            request.decided_at = time.time()

            grant_scope = scope or request.scope
            grant = PermissionGrant(
                capability_id=request.capability_id,
                granted_at=time.time(),
                scope=grant_scope,
                conditions=conditions or {},
            )

            if grant_scope == PermissionScope.PERMANENT:
                self._grants[request.capability_id] = grant
                self._save_persistent_grants()
            elif grant_scope == PermissionScope.SESSION:
                self._grants[request.capability_id] = grant
            # ONESHOT: don't store, just allow once

            # Fire callbacks
            for cb in self._callbacks.get(request_id, []):
                try:
                    cb(request)
                except Exception as e:
                    logger.warning("Permission callback failed: %s", e)

            del self._pending_requests[request_id]
            if request_id in self._callbacks:
                del self._callbacks[request_id]

            self._generation += 1
            logger.info("Permission granted: %s (scope=%s)", request.capability_id, grant_scope.value)
            return True

    def deny_permission(self, request_id: str) -> bool:
        """Deny a pending permission request."""
        with self._lock:
            request = self._pending_requests.get(request_id)
            if not request:
                return False

            if request.status != PermissionStatus.PENDING:
                return False

            request.status = PermissionStatus.DENIED
            request.decided_at = time.time()

            # Record denial with cooldown
            self._denials[request.capability_id] = time.time()

            # Fire callbacks
            for cb in self._callbacks.get(request_id, []):
                try:
                    cb(request)
                except Exception as e:
                    logger.warning("Permission callback failed: %s", e)

            del self._pending_requests[request_id]
            if request_id in self._callbacks:
                del self._callbacks[request_id]

            self._generation += 1
            logger.info("Permission denied: %s", request.capability_id)
            return True

    def get_pending_requests(self) -> List[PermissionRequest]:
        """Get all pending permission requests."""
        with self._lock:
            return list(self._pending_requests.values())

    def get_request(self, request_id: str) -> Optional[PermissionRequest]:
        """Get a specific permission request by ID."""
        with self._lock:
            return self._pending_requests.get(request_id)

    # =========================================================================
    # GRANT MANAGEMENT
    # =========================================================================

    def has_grant(self, capability_id: str) -> bool:
        """Check if a capability has an active grant."""
        grant = self._grants.get(capability_id)
        return grant is not None and not self._is_grant_expired(grant)

    def get_grant(self, capability_id: str) -> Optional[PermissionGrant]:
        """Get the grant for a capability."""
        with self._lock:
            grant = self._grants.get(capability_id)
            if grant and self._is_grant_expired(grant):
                del self._grants[capability_id]
                self._save_persistent_grants()
                return None
            return grant

    def revoke_grant(self, capability_id: str, permanent: bool = False) -> bool:
        """Revoke a permission grant."""
        with self._lock:
            if capability_id in self._grants:
                del self._grants[capability_id]
                if permanent:
                    self._save_persistent_grants()
                self._generation += 1
                return True
            return False

    def list_grants(self) -> Dict[str, PermissionGrant]:
        """List all active grants."""
        with self._lock:
            # Clean expired
            expired = [cid for cid, g in self._grants.items() if self._is_grant_expired(g)]
            for cid in expired:
                del self._grants[cid]
            if expired:
                self._save_persistent_grants()
            return dict(self._grants)

    def _is_grant_expired(self, grant: PermissionGrant) -> bool:
        if grant.expires_at is None:
            return False
        return time.time() > grant.expires_at

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    def _load_persistent_grants(self) -> None:
        """Load permanent grants from disk."""
        if not self._persistence_path.exists():
            return

        try:
            with open(self._persistence_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for cap_id, grant_data in data.get("grants", {}).items():
                grant = PermissionGrant(
                    capability_id=grant_data["capability_id"],
                    granted_at=grant_data["granted_at"],
                    scope=PermissionScope(grant_data["scope"]),
                    expires_at=grant_data.get("expires_at"),
                    conditions=grant_data.get("conditions", {}),
                    source=grant_data.get("source", "user"),
                )
                if not self._is_grant_expired(grant):
                    self._grants[cap_id] = grant

            logger.info("Loaded %d persistent permission grants", len(self._grants))
        except Exception as e:
            logger.warning("Failed to load permissions: %s", e)

    def _save_persistent_grants(self) -> None:
        """Save permanent grants to disk."""
        try:
            permanent_grants = {
                cap_id: {
                    "capability_id": g.capability_id,
                    "granted_at": g.granted_at,
                    "scope": g.scope.value,
                    "expires_at": g.expires_at,
                    "conditions": g.conditions,
                    "source": g.source,
                }
                for cap_id, g in self._grants.items()
                if g.scope == PermissionScope.PERMANENT and not self._is_grant_expired(g)
            }

            data = {
                "grants": permanent_grants,
                "saved_at": time.time(),
            }

            self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._persistence_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.warning("Failed to save permissions: %s", e)

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "pending_requests": len(self._pending_requests),
                "active_grants": len([g for g in self._grants.values() if not self._is_grant_expired(g)]),
                "permanent_grants": len([g for g in self._grants.values() if g.scope == PermissionScope.PERMANENT]),
                "recent_denials": len(self._denials),
                "generation": self._generation,
            }

    def get_generation(self) -> int:
        return self._generation

    def clear(self) -> None:
        """Clear all state (for testing)."""
        with self._lock:
            self._pending_requests.clear()
            self._grants.clear()
            self._denials.clear()
            self._callbacks.clear()
            self._generation += 1


# Global instance
_manager: Optional[PermissionManager] = None


def get_permission_manager() -> PermissionManager:
    """Get the global PermissionManager instance."""
    global _manager
    if _manager is None:
        _manager = PermissionManager()
    return _manager


def initialize_permission_manager() -> None:
    """Explicit initialization."""
    get_permission_manager()
    logger.info("PermissionManager explicitly initialized")


if __name__ == "__main__":
    # Demo
    pm = get_permission_manager()
    pm.clear()

    # Check permission for open_app (should be auto-granted at autonomy 3)
    print(f"open_app at autonomy 3: {pm.check_permission('open_app', autonomy_level=3)}")
    print(f"open_app at autonomy 1: {pm.check_permission('open_app', autonomy_level=1)}")

    # Check permission for send_message (needs confirmation at autonomy < 4)
    print(f"send_message at autonomy 3: {pm.check_permission('send_message', autonomy_level=3)}")
    print(f"send_message at autonomy 4: {pm.check_permission('send_message', autonomy_level=4)}")

    # Check permission for system_shutdown (CRITICAL - never auto)
    print(f"system_shutdown at autonomy 5: {pm.check_permission('system_shutdown', autonomy_level=5)}")

    # Request permission
    req = pm.request_permission("send_message", "User wants to send WhatsApp")
    print(f"Requested: {req.id}")

    # Grant it
    pm.grant_permission(req.id, PermissionScope.PERMANENT)
    print(f"After grant: {pm.check_permission('send_message', autonomy_level=1)}")

    print(f"Stats: {pm.get_stats()}")