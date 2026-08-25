"""Tests for Security Gate (FASE 2B B6)."""

import sys
sys.path.insert(0, ".")

from core.security_gate import (
    SecurityGate,
    GateDecision,
    SecurityPolicy,
    get_security_gate,
    initialize_security_gate,
)
from core.capability_registry import get_registry, RiskLevel
from core.permission_manager import get_permission_manager, PermissionScope


def test_gate_creation():
    """Verifica que el SecurityGate se crea correctamente."""
    gate = get_security_gate()
    stats = gate.get_stats()
    assert "blocked_capabilities" in stats
    assert "always_confirm" in stats
    print(f"PASS: SecurityGate created, stats={stats}")
    return True


def test_evaluate_allow():
    """Verifica evaluación ALLOW para capacidades permitidas."""
    gate = get_security_gate()
    gate.clear()

    # open_app: LOW risk, no confirmation needed, autonomy 3
    result = gate.evaluate("open_app", autonomy_level=3)
    assert result.decision == GateDecision.ALLOW
    assert result.allowed is True
    print(f"PASS: open_app allowed at autonomy 3")

    # youtube_play: NONE risk, autonomy 2+
    result = gate.evaluate("youtube_play", autonomy_level=2)
    assert result.decision == GateDecision.ALLOW
    print(f"PASS: youtube_play allowed at autonomy 2")

    # web_search: NONE risk
    result = gate.evaluate("web_search", autonomy_level=1)
    assert result.decision == GateDecision.ALLOW
    print(f"PASS: web_search allowed at autonomy 1")

    return True


def test_evaluate_require_permission():
    """Verifica evaluación REQURE_PERMISSION para capacidades que necesitan permiso."""
    gate = get_security_gate()
    pm = get_permission_manager()
    gate.clear()
    pm.clear()

    # send_message is in always_confirm by default, remove it to test REQUIRE_PERMISSION
    gate._policy.always_confirm.discard("send_message")
    
    # send_message: LOW risk, requires_confirmation=True, autonomy 1 (< 4)
    result = gate.evaluate("send_message", autonomy_level=1)
    assert result.decision == GateDecision.REQUIRE_PERMISSION
    assert result.allowed is False
    print(f"PASS: send_message requires permission at autonomy 1 (not in always_confirm)")

    # send_message at autonomy 3
    result = gate.evaluate("send_message", autonomy_level=3)
    assert result.decision == GateDecision.REQUIRE_PERMISSION
    print(f"PASS: send_message requires permission at autonomy 3 (not in always_confirm)")

    return True


def test_evaluate_require_confirmation():
    """Verifica evaluación REQUIRE_CONFIRMATION para always_confirm."""
    gate = get_security_gate()
    gate.clear()

    # file_operations is in always_confirm by default
    result = gate.evaluate("file_operations", autonomy_level=5)
    assert result.decision == GateDecision.REQUIRE_CONFIRMATION
    assert result.allowed is False
    print(f"PASS: file_operations requires confirmation at autonomy 5")

    # send_message is also in always_confirm by default
    result = gate.evaluate("send_message", autonomy_level=5)
    assert result.decision == GateDecision.REQUIRE_CONFIRMATION
    print(f"PASS: send_message requires confirmation at autonomy 5")

    return True


def test_evaluate_deny_blocked():
    """Verifica DENY para capacidades bloqueadas."""
    gate = get_security_gate()
    gate.clear()

    # system_shutdown is blocked by default
    result = gate.evaluate("system_shutdown", autonomy_level=5)
    # Actually system_shutdown is in both blocked and always_confirm
    # The blocked check comes first
    assert result.decision == GateDecision.DENY
    print(f"PASS: system_shutdown blocked by policy")

    return True


def test_evaluate_deny_autonomy():
    """Verifica DENY por autonomía insuficiente."""
    gate = get_security_gate()
    gate.clear()

    # system_shutdown requires autonomy 5 (but it's blocked, so test computer_control)
    result = gate.evaluate("computer_control", autonomy_level=3)
    assert result.decision == GateDecision.DENY
    assert "autonomy" in result.reason.lower()
    print(f"PASS: computer_control denied at autonomy 3 (needs 4)")

    result = gate.evaluate("computer_control", autonomy_level=4)
    assert result.decision != GateDecision.DENY or "autonomy" not in result.reason.lower()
    print(f"PASS: computer_control allowed at autonomy 4")

    # system_shutdown is blocked, so test send_message with min_autonomy
    gate.set_min_autonomy("send_message", 5)
    result = gate.evaluate("send_message", autonomy_level=4)
    assert result.decision == GateDecision.DENY
    assert "autonomy" in result.reason.lower()
    print(f"PASS: send_message denied at autonomy 4 (needs 5)")

    return True


def test_rate_limiting():
    """Verifica rate limiting."""
    gate = get_security_gate()
    gate.clear()

    # Use an existing capability for testing - add a custom rate limit
    gate.set_rate_limit("open_app", 2, 60)

    # First call - allowed
    result = gate.evaluate("open_app", autonomy_level=3)
    assert result.decision == GateDecision.ALLOW
    gate.record_execution("open_app")

    # Second call - allowed
    result = gate.evaluate("open_app", autonomy_level=3)
    assert result.decision == GateDecision.ALLOW
    gate.record_execution("open_app")

    # Third call - rate limited
    result = gate.evaluate("open_app", autonomy_level=3)
    assert result.decision == GateDecision.RATE_LIMITED
    print(f"PASS: Rate limiting works (2 calls per 60s)")

    return True


def test_daily_quota():
    """Verifica cuota diaria."""
    gate = get_security_gate()
    gate.clear()

    # Use an existing capability for testing
    gate.set_daily_quota("open_app", 2)

    # First call - allowed
    result = gate.evaluate("open_app", autonomy_level=3)
    assert result.decision == GateDecision.ALLOW
    gate.record_execution("open_app")

    # Second call - allowed
    result = gate.evaluate("open_app", autonomy_level=3)
    assert result.decision == GateDecision.ALLOW
    gate.record_execution("open_app")

    # Third call - quota exceeded
    result = gate.evaluate("open_app", autonomy_level=3)
    assert result.decision == GateDecision.QUOTA_EXCEEDED
    print(f"PASS: Daily quota works (2 calls per day)")

    return True


def test_permission_grant_bypasses():
    """Verifica que otorgar permiso permite la ejecución."""
    gate = get_security_gate()
    pm = get_permission_manager()
    gate.clear()
    pm.clear()

    # Remove send_message from always_confirm to test REQUIRE_PERMISSION path
    gate._policy.always_confirm.discard("send_message")

    # send_message at autonomy 1 - needs permission
    result = gate.evaluate("send_message", autonomy_level=1)
    assert result.decision == GateDecision.REQUIRE_PERMISSION

    # Grant permission
    req = pm.request_permission("send_message", "Test grant", scope=PermissionScope.SESSION)
    pm.grant_permission(req.id, PermissionScope.SESSION)

    # Now should be allowed
    result = gate.evaluate("send_message", autonomy_level=1)
    assert result.decision == GateDecision.ALLOW
    print(f"PASS: Permission grant bypasses REQUIRE_PERMISSION")

    return True


def test_always_confirm_bypasses_with_confirmation():
    """Verifica que always_confirm requiere confirmación explícita."""
    gate = get_security_gate()
    gate.clear()

    # file_operations is in always_confirm
    result = gate.evaluate("file_operations", autonomy_level=5)
    assert result.decision == GateDecision.REQUIRE_CONFIRMATION
    print(f"PASS: always_confirm requires confirmation even at high autonomy")

    return True


def test_record_execution():
    """Verifica registro de ejecuciones para rate limiting."""
    gate = get_security_gate()
    gate.clear()

    gate.set_rate_limit("open_app", 1, 60)

    # First evaluation - allowed
    result = gate.evaluate("open_app", autonomy_level=3)
    assert result.decision == GateDecision.ALLOW

    # Record execution
    gate.record_execution("open_app")

    # Second evaluation - rate limited
    result = gate.evaluate("open_app", autonomy_level=3)
    assert result.decision == GateDecision.RATE_LIMITED
    print(f"PASS: record_execution updates rate limit counter")

    return True


def test_audit_log():
    """Verifica registro de auditoría."""
    gate = get_security_gate()
    gate.clear()

    gate.evaluate("open_app", autonomy_level=3)
    gate.evaluate("send_message", autonomy_level=1)

    audit = gate.get_audit_log(10)
    assert len(audit) >= 2
    assert audit[-2]["capability_id"] == "open_app"
    assert audit[-2]["decision"] == "allow"
    assert audit[-1]["capability_id"] == "send_message"
    assert audit[-1]["decision"] == "require_confirmation"  # send_message is in always_confirm by default
    print(f"PASS: Audit log records evaluations")

    return True


def test_policy_management():
    """Verifica gestión de políticas."""
    gate = get_security_gate()
    gate.clear()

    # Add to blocked - use an existing capability
    gate.add_blocked("open_app")
    result = gate.evaluate("open_app", autonomy_level=5)
    assert result.decision == GateDecision.DENY
    print(f"PASS: add_blocked works")

    # Remove from blocked
    gate.remove_blocked("open_app")
    result = gate.evaluate("open_app", autonomy_level=5)
    assert result.decision != GateDecision.DENY  # Not blocked anymore
    print(f"PASS: remove_blocked works")

    # Set min autonomy on an existing capability
    gate.set_min_autonomy("open_app", 4)
    result = gate.evaluate("open_app", autonomy_level=3)
    assert result.decision == GateDecision.DENY
    result = gate.evaluate("open_app", autonomy_level=4)
    assert result.decision != GateDecision.DENY
    print(f"PASS: set_min_autonomy works")

    return True


def test_stats():
    """Verifica estadísticas."""
    gate = get_security_gate()
    gate.clear()

    stats = gate.get_stats()
    assert "blocked_capabilities" in stats
    assert "always_confirm" in stats
    assert "rate_limited" in stats
    assert "daily_quotas" in stats
    assert "generation" in stats
    print(f"PASS: Stats: {stats}")

    return True


def test_integration_with_registry():
    """Verifica integración completa con CapabilityRegistry."""
    gate = get_security_gate()
    gate.clear()

    registry = get_registry()
    for cap in registry.get_all():
        # Test at different autonomy levels
        for autonomy in [0, 2, 3, 5]:
            result = gate.evaluate(cap.id, autonomy_level=autonomy)
            # Should not crash
            assert result.decision in GateDecision
            assert result.capability_id == cap.id
            assert result.risk_level == cap.risk_level

    print(f"PASS: Integration with all {len(registry.get_all())} capabilities verified")
    return True


def test_generation_tracking():
    """Verifica tracking de generación."""
    gate = get_security_gate()
    gate.clear()

    gen1 = gate.get_generation()
    gate.evaluate("open_app", autonomy_level=3)
    gen2 = gate.get_generation()
    # Evaluation doesn't increment generation (only policy changes and record_execution)
    # Let's check record_execution
    gate.record_execution("open_app")
    gen3 = gate.get_generation()
    assert gen3 > gen1
    print(f"PASS: Generation tracking works: {gen1} -> {gen2} -> {gen3}")

    return True


def main():
    """Ejecuta todas las pruebas."""
    print("=" * 60)
    print("TESTS Security Gate — FASE 2B B6")
    print("=" * 60)
    print()

    tests = [
        test_gate_creation,
        test_evaluate_allow,
        test_evaluate_require_permission,
        test_evaluate_require_confirmation,
        test_evaluate_deny_blocked,
        test_evaluate_deny_autonomy,
        test_rate_limiting,
        test_daily_quota,
        test_permission_grant_bypasses,
        test_always_confirm_bypasses_with_confirmation,
        test_record_execution,
        test_audit_log,
        test_policy_management,
        test_stats,
        test_integration_with_registry,
        test_generation_tracking,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__} — {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 60)
    print(f"RESULTADO: {passed} pasaron, {failed} fallaron")
    print("=" * 60)

    if failed == 0:
        print("✅ Security Gate B6 — ESTABLE")
        return True
    else:
        print("❌ Security Gate B6 — TIENE FALLOS")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)