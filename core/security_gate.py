"""Security Gate — Safety validation and policy enforcement for P.I.P.E.

This module provides:
- Pre-execution safety checks for capabilities
- Policy-based allow/deny decisions
- Rate limiting and quota enforcement
- Dangerous operation detection
- Audit logging for security events
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from core.capability_registry import (
    Capability,
    RiskLevel,
    get_registry,
)
from core.permission_manager import get_permission_manager, PermissionScope

logger = logging.getLogger(__name__)


class GateDecision(Enum):
    """Result of a security gate check."""
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_PERMISSION = "require_permission"
    REQUIRE_CONFIRMATION = "require_confirmation"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXCEEDED = "quota_exceeded"


@dataclass
class GateResult:
    """Result of a security gate evaluation."""
    decision: GateDecision
    reason: str
    capability_id: str
    risk_level: RiskLevel
    required_permissions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        return self.decision == GateDecision.ALLOW


@dataclass
class SecurityPolicy:
    """Security policy configuration."""
    # Capability IDs that are always blocked
    blocked_capabilities: Set[str] = field(default_factory=set)

    # Capability IDs that always require explicit confirmation
    always_confirm: Set[str] = field(default_factory=set)

    # Rate limits: capability_id -> (max_calls, time_window_seconds)
    rate_limits: Dict[str, tuple] = field(default_factory=dict)

    # Daily quotas: capability_id -> max_calls_per_day
    daily_quotas: Dict[str, int] = field(default_factory=dict)

    # Allowed autonomy levels per capability
    min_autonomy: Dict[str, int] = field(default_factory=dict)

    # Time-based restrictions: capability_id -> (start_hour, end_hour) UTC
    time_restrictions: Dict[str, tuple] = field(default_factory=dict)


class SecurityGate:
    """Security gate for validating capability executions.

    Integrates with:
    - CapabilityRegistry for risk assessment
    - PermissionManager for consent tracking
    - Rate limiting and quota enforcement
    """

    _instance: Optional[SecurityGate] = None
    _lock = threading.RLock()

    def __new__(cls) -> SecurityGate:
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
            self._permission_manager = get_permission_manager()
            self._policy = SecurityPolicy()
            self._call_history: Dict[str, List[float]] = {}  # capability_id -> timestamps
            self._daily_counts: Dict[str, int] = {}  # capability_id -> count
            self._daily_reset_time = self._get_next_midnight()
            self._audit_log: List[Dict[str, Any]] = []
            self._generation = 0

            # Initialize default policy
            self._init_default_policy()

            self._initialized = True
            logger.info("SecurityGate initialized")

    def _init_default_policy(self) -> None:
        """Initialize default security policies."""
        # Block dangerous capabilities by default
        self._policy.blocked_capabilities = {
            "system_shutdown",  # Requires explicit user action
        }

        # Always require confirmation for critical operations
        self._policy.always_confirm = {
            "system_shutdown",
            "computer_control",
            "file_operations",  # File writes
            "send_message",
        }

        # Rate limits (max_calls, window_seconds)
        self._policy.rate_limits = {
            "web_search": (30, 60),      # 30 searches per minute
            "youtube_play": (10, 60),    # 10 plays per minute
            "send_message": (5, 60),     # 5 messages per minute
            "open_app": (20, 60),        # 20 app opens per minute
        }

        # Daily quotas
        self._policy.daily_quotas = {
            "web_search": 500,
            "send_message": 100,
            "file_operations": 200,
        }

        # Minimum autonomy levels
        self._policy.min_autonomy = {
            "system_shutdown": 5,
            "computer_control": 4,
            "send_message": 1,
            "file_operations": 2,
        }

    def _get_next_midnight(self) -> float:
        """Get timestamp of next midnight UTC."""
        from datetime import datetime, timedelta
        now = datetime.utcnow()
        midnight = datetime(now.year, now.month, now.day) + timedelta(days=1)
        return midnight.timestamp()

    def _reset_daily_counts_if_needed(self) -> None:
        """Reset daily counters at midnight."""
        if time.time() >= self._daily_reset_time:
            self._daily_counts.clear()
            self._daily_reset_time = self._get_next_midnight()
            logger.info("Daily quotas reset")

    def evaluate(
        self,
        capability_id: str,
        autonomy_level: int = 3,
        context: Optional[Dict[str, Any]] = None,
    ) -> GateResult:
        """Evaluate if a capability execution is allowed.

        Returns GateResult with decision and metadata.
        """
        self._reset_daily_counts_if_needed()

        capability = self._registry.get(capability_id)
        if not capability:
            return GateResult(
                decision=GateDecision.DENY,
                reason=f"Unknown capability: {capability_id}",
                capability_id=capability_id,
                risk_level=RiskLevel.NONE,
            )

        # Check blocked list
        if capability_id in self._policy.blocked_capabilities:
            return self._log_and_return(GateResult(
                decision=GateDecision.DENY,
                reason=f"Capability blocked by policy: {capability_id}",
                capability_id=capability_id,
                risk_level=capability.risk_level,
            ))

        # Check minimum autonomy
        min_autonomy = self._policy.min_autonomy.get(capability_id, 0)
        if autonomy_level < min_autonomy:
            return self._log_and_return(GateResult(
                decision=GateDecision.DENY,
                reason=f"Insufficient autonomy level: need {min_autonomy}, have {autonomy_level}",
                capability_id=capability_id,
                risk_level=capability.risk_level,
                metadata={"required_autonomy": min_autonomy, "current_autonomy": autonomy_level},
            ))

        # Check time restrictions
        time_restriction = self._policy.time_restrictions.get(capability_id)
        if time_restriction:
            from datetime import datetime
            now = datetime.utcnow().hour
            start, end = time_restriction
            if not (start <= now < end):
                return self._log_and_return(GateResult(
                    decision=GateDecision.DENY,
                    reason=f"Time restriction: allowed {start}:00-{end}:00 UTC",
                    capability_id=capability_id,
                    risk_level=capability.risk_level,
                ))

        # Check rate limit
        rate_limit = self._policy.rate_limits.get(capability_id)
        if rate_limit:
            max_calls, window = rate_limit
            recent_calls = self._get_recent_calls(capability_id, window)
            if len(recent_calls) >= max_calls:
                return self._log_and_return(GateResult(
                    decision=GateDecision.RATE_LIMITED,
                    reason=f"Rate limit exceeded: {max_calls} calls per {window}s",
                    capability_id=capability_id,
                    risk_level=capability.risk_level,
                    metadata={"recent_calls": len(recent_calls), "limit": max_calls, "window": window},
                ))

        # Check daily quota
        quota = self._policy.daily_quotas.get(capability_id)
        if quota:
            today_count = self._daily_counts.get(capability_id, 0)
            if today_count >= quota:
                return self._log_and_return(GateResult(
                    decision=GateDecision.QUOTA_EXCEEDED,
                    reason=f"Daily quota exceeded: {quota} calls per day",
                    capability_id=capability_id,
                    risk_level=capability.risk_level,
                    metadata={"used_today": today_count, "quota": quota},
                ))

        # Check always_confirm FIRST - these always require explicit confirmation
        if capability_id in self._policy.always_confirm:
            return self._log_and_return(GateResult(
                decision=GateDecision.REQUIRE_CONFIRMATION,
                reason=f"Explicit confirmation required for {capability.name_es}",
                capability_id=capability_id,
                risk_level=capability.risk_level,
            ))

        # Check permission manager
        has_permission = self._permission_manager.check_permission(
            capability_id, autonomy_level, context
        )

        if not has_permission:
            return self._log_and_return(GateResult(
                decision=GateDecision.REQUIRE_PERMISSION,
                reason=f"Permission required for {capability.name_es}",
                capability_id=capability_id,
                risk_level=capability.risk_level,
            ))

        # All checks passed
        return self._log_and_return(GateResult(
            decision=GateDecision.ALLOW,
            reason="All security checks passed",
            capability_id=capability_id,
            risk_level=capability.risk_level,
        ))

    def _get_recent_calls(self, capability_id: str, window_seconds: int) -> List[float]:
        """Get timestamps of recent calls within window."""
        now = time.time()
        cutoff = now - window_seconds
        calls = self._call_history.get(capability_id, [])
        return [t for t in calls if t > cutoff]

    def record_execution(self, capability_id: str, success: bool = True) -> None:
        """Record a capability execution for rate limiting and quotas."""
        with self._lock:
            now = time.time()
            if capability_id not in self._call_history:
                self._call_history[capability_id] = []
            self._call_history[capability_id].append(now)

            # Update daily count
            self._daily_counts[capability_id] = self._daily_counts.get(capability_id, 0) + 1

            # Clean old call history (keep last 24 hours)
            cutoff = now - 86400
            for cap_id in list(self._call_history.keys()):
                self._call_history[cap_id] = [t for t in self._call_history[cap_id] if t > cutoff]
                if not self._call_history[cap_id]:
                    del self._call_history[cap_id]

            self._generation += 1

    def _log_and_return(self, result: GateResult) -> GateResult:
        """Log audit entry and return result."""
        audit_entry = {
            "timestamp": time.time(),
            "capability_id": result.capability_id,
            "decision": result.decision.value,
            "reason": result.reason,
            "risk_level": result.risk_level.value if hasattr(result.risk_level, 'value') else str(result.risk_level),
        }
        self._audit_log.append(audit_entry)

        # Keep audit log bounded
        if len(self._audit_log) > 10000:
            self._audit_log = self._audit_log[-5000:]

        logger.debug("SecurityGate: %s - %s", result.capability_id, result.decision.value)
        return result

    # =========================================================================
    # POLICY MANAGEMENT
    # =========================================================================

    def get_policy(self) -> SecurityPolicy:
        """Get current security policy."""
        return self._policy

    def update_policy(self, **kwargs) -> None:
        """Update security policy."""
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self._policy, key):
                    setattr(self._policy, key, value)
                    logger.info("Security policy updated: %s", key)
            self._generation += 1

    def add_blocked(self, capability_id: str) -> None:
        """Add capability to blocked list."""
        with self._lock:
            self._policy.blocked_capabilities.add(capability_id)
            self._generation += 1

    def remove_blocked(self, capability_id: str) -> None:
        """Remove capability from blocked list."""
        with self._lock:
            self._policy.blocked_capabilities.discard(capability_id)
            self._generation += 1

    def set_rate_limit(self, capability_id: str, max_calls: int, window_seconds: int) -> None:
        """Set rate limit for a capability."""
        with self._lock:
            self._policy.rate_limits[capability_id] = (max_calls, window_seconds)
            self._generation += 1

    def set_daily_quota(self, capability_id: str, max_calls: int) -> None:
        """Set daily quota for a capability."""
        with self._lock:
            self._policy.daily_quotas[capability_id] = max_calls
            self._generation += 1

    def set_min_autonomy(self, capability_id: str, level: int) -> None:
        """Set minimum autonomy level for a capability."""
        with self._lock:
            self._policy.min_autonomy[capability_id] = level
            self._generation += 1

    # =========================================================================
    # AUDIT & STATS
    # =========================================================================

    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit log entries."""
        with self._lock:
            return self._audit_log[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """Get security gate statistics."""
        with self._lock:
            self._reset_daily_counts_if_needed()
            return {
                "blocked_capabilities": len(self._policy.blocked_capabilities),
                "always_confirm": len(self._policy.always_confirm),
                "rate_limited": len(self._policy.rate_limits),
                "daily_quotas": len(self._policy.daily_quotas),
                "min_autonomy_rules": len(self._policy.min_autonomy),
                "recent_audit_entries": len(self._audit_log),
                "tracked_calls": sum(len(v) for v in self._call_history.values()),
                "daily_counts": dict(self._daily_counts),
                "generation": self._generation,
            }

    def get_generation(self) -> int:
        return self._generation

    def clear(self) -> None:
        """Clear all state (for testing)."""
        with self._lock:
            self._call_history.clear()
            self._daily_counts.clear()
            self._audit_log.clear()
            # Reset policy to defaults
            self._policy = SecurityPolicy()
            self._init_default_policy()
            self._generation += 1


# Global instance
_gate: Optional[SecurityGate] = None


def get_security_gate() -> SecurityGate:
    """Get the global SecurityGate instance."""
    global _gate
    if _gate is None:
        _gate = SecurityGate()
    return _gate


def initialize_security_gate() -> None:
    """Explicit initialization."""
    get_security_gate()
    logger.info("SecurityGate explicitly initialized")


if __name__ == "__main__":
    # Demo
    gate = get_security_gate()
    gate.clear()

    # Test basic evaluation
    print("=== SecurityGate Demo ===")

    # Test 1: Allow - low risk, autonomy sufficient
    result = gate.evaluate("open_app", autonomy_level=3)
    print(f"open_app (autonomy 3): {result.decision.value} - {result.reason}")

    # Test 2: Require permission - send_message at low autonomy
    result = gate.evaluate("send_message", autonomy_level=1)
    print(f"send_message (autonomy 1): {result.decision.value} - {result.reason}")

    # Test 3: Allow - send_message at high autonomy
    result = gate.evaluate("send_message", autonomy_level=5)
    print(f"send_message (autonomy 5): {result.decision.value} - {result.reason}")

    # Test 4: Blocked capability
    result = gate.evaluate("system_shutdown", autonomy_level=5)
    print(f"system_shutdown (autonomy 5): {result.decision.value} - {result.reason}")

    # Test 5: Rate limiting
    gate.set_rate_limit("test_cap", 2, 60)
    print(f"test_cap (1st): {gate.evaluate('test_cap', autonomy_level=3).decision.value}")
    gate.record_execution("test_cap")
    print(f"test_cap (2nd): {gate.evaluate('test_cap', autonomy_level=3).decision.value}")
    gate.record_execution("test_cap")
    print(f"test_cap (3rd): {gate.evaluate('test_cap', autonomy_level=3).decision.value}")

    print(f"\nStats: {gate.get_stats()}")