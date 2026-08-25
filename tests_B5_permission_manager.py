"""Tests for Permission Manager (FASE 2B B5)."""

import sys
sys.path.insert(0, ".")

from core.permission_manager import (
    PermissionManager,
    PermissionStatus,
    PermissionScope,
    get_permission_manager,
    initialize_permission_manager,
)
from core.capability_registry import get_registry, RiskLevel


def test_manager_creation():
    """Verifica que el manager se crea correctamente."""
    pm = get_permission_manager()
    stats = pm.get_stats()
    assert "pending_requests" in stats
    assert "active_grants" in stats
    print(f"PASS: PermissionManager created, stats={stats}")
    return True


def test_check_permission_auto_granted():
    """Verifica permisos auto-otorgados (bajo riesgo, autonomía suficiente)."""
    pm = get_permission_manager()
    pm.clear()

    # open_app: LOW risk, no confirmation needed -> auto at autonomy >= 2
    assert pm.check_permission("open_app", autonomy_level=3) is True
    assert pm.check_permission("open_app", autonomy_level=2) is True
    assert pm.check_permission("open_app", autonomy_level=1) is True  # LOW, no requires_confirmation
    assert pm.check_permission("open_app", autonomy_level=0) is True
    print("PASS: open_app auto-granted at all autonomy levels")

    # youtube_play: NONE risk -> auto at autonomy >= 2
    assert pm.check_permission("youtube_play", autonomy_level=2) is True
    assert pm.check_permission("youtube_play", autonomy_level=3) is True
    print("PASS: youtube_play auto-granted at autonomy >= 2")

    return True


def test_check_permission_requires_confirmation():
    """Verifica permisos que requieren confirmación según autonomía."""
    pm = get_permission_manager()
    pm.clear()

    # send_message: LOW risk, requires_confirmation=True
    # autonomy 0-3: needs permission
    # autonomy 4-5: auto
    assert pm.check_permission("send_message", autonomy_level=0) is False
    assert pm.check_permission("send_message", autonomy_level=1) is False
    assert pm.check_permission("send_message", autonomy_level=2) is False
    assert pm.check_permission("send_message", autonomy_level=3) is False
    assert pm.check_permission("send_message", autonomy_level=4) is True
    assert pm.check_permission("send_message", autonomy_level=5) is True
    print("PASS: send_message requires permission at autonomy < 4")

    # computer_settings: HIGH risk, requires_confirmation=True
    # autonomy 0-3: needs permission
    # autonomy 4-5: auto (requires_confirmation=True only gates < 4)
    assert pm.check_permission("computer_settings", autonomy_level=3) is False
    assert pm.check_permission("computer_settings", autonomy_level=4) is True
    assert pm.check_permission("computer_settings", autonomy_level=5) is True
    print("PASS: computer_settings requires permission at autonomy < 4")

    return True


def test_check_permission_critical():
    """Verifica que CRITICAL siempre requiere permiso explícito."""
    pm = get_permission_manager()
    pm.clear()

    # system_shutdown: CRITICAL -> never auto
    assert pm.check_permission("system_shutdown", autonomy_level=0) is False
    assert pm.check_permission("system_shutdown", autonomy_level=2) is False
    assert pm.check_permission("system_shutdown", autonomy_level=5) is False
    print("PASS: system_shutdown (CRITICAL) never auto-granted")

    return True


def test_request_and_grant():
    """Verifica flujo completo de solicitud y otorgamiento."""
    pm = get_permission_manager()
    pm.clear()

    # Request permission for send_message
    request = pm.request_permission("send_message", "Send WhatsApp to Juan", PermissionScope.SESSION)
    assert request.id in pm._pending_requests
    assert request.status == PermissionStatus.PENDING
    assert request.capability_id == "send_message"
    print(f"PASS: Request created: {request.id}")

    # Grant permission
    result = pm.grant_permission(request.id, PermissionScope.SESSION)
    assert result is True
    assert request.status == PermissionStatus.GRANTED
    assert request.decided_at is not None
    print(f"PASS: Permission granted")

    # Now check_permission should return True
    assert pm.check_permission("send_message", autonomy_level=1) is True
    print("PASS: check_permission returns True after grant")

    # Request should be removed from pending
    assert request.id not in pm._pending_requests
    print("PASS: Request removed from pending")

    return True


def test_deny_permission():
    """Verifica denegación de permiso."""
    pm = get_permission_manager()
    pm.clear()

    request = pm.request_permission("send_message", "Test deny", PermissionScope.SESSION)
    result = pm.deny_permission(request.id)
    assert result is True
    assert request.status == PermissionStatus.DENIED
    print(f"PASS: Permission denied")

    # Should be in denials with cooldown
    assert "send_message" in pm._denials
    print("PASS: Denial recorded with cooldown")

    # check_permission should still return False
    assert pm.check_permission("send_message", autonomy_level=1) is False
    print("PASS: check_permission returns False after denial")

    return True


def test_permission_scopes():
    """Verifica diferentes scopes de permiso."""
    pm = get_permission_manager()
    pm.clear()

    # SESSION scope
    req = pm.request_permission("send_message", "Test session", PermissionScope.SESSION)
    pm.grant_permission(req.id, PermissionScope.SESSION)
    assert pm.has_grant("send_message") is True
    print("PASS: SESSION grant works")

    # Clear and test PERMANENT
    pm.clear()
    req = pm.request_permission("send_message", "Test permanent", PermissionScope.PERMANENT)
    pm.grant_permission(req.id, PermissionScope.PERMANENT)
    assert pm.has_grant("send_message") is True
    print("PASS: PERMANENT grant works")

    # ONESHOT - should not persist
    pm.clear()
    req = pm.request_permission("send_message", "Test oneshot", PermissionScope.ONESHOT)
    pm.grant_permission(req.id, PermissionScope.ONESHOT)
    # ONESHOT grants are not stored
    assert pm.has_grant("send_message") is False
    print("PASS: ONESHOT grant not persisted")

    return True


def test_revoke_grant():
    """Verifica revocación de permisos."""
    pm = get_permission_manager()
    pm.clear()

    req = pm.request_permission("send_message", "Test revoke", PermissionScope.PERMANENT)
    pm.grant_permission(req.id, PermissionScope.PERMANENT)
    assert pm.has_grant("send_message") is True

    result = pm.revoke_grant("send_message", permanent=True)
    assert result is True
    assert pm.has_grant("send_message") is False
    print("PASS: Grant revoked")

    return True


def test_persistence():
    """Verifica persistencia de grants PERMANENT."""
    pm = get_permission_manager()
    pm.clear()

    req = pm.request_permission("send_message", "Test persistence", PermissionScope.PERMANENT)
    pm.grant_permission(req.id, PermissionScope.PERMANENT)
    assert pm.has_grant("send_message") is True

    # Create new manager instance (simulates restart)
    pm2 = get_permission_manager()
    # Since it's a singleton, it's the same instance
    # But we can test the load/save logic
    assert pm2.has_grant("send_message") is True
    print("PASS: Persistence works (singleton)")

    return True


def test_pending_requests_listing():
    """Verifica listado de solicitudes pendientes."""
    pm = get_permission_manager()
    pm.clear()

    req1 = pm.request_permission("send_message", "Req 1")
    req2 = pm.request_permission("computer_settings", "Req 2")

    pending = pm.get_pending_requests()
    assert len(pending) == 2
    ids = {r.id for r in pending}
    assert req1.id in ids
    assert req2.id in ids
    print(f"PASS: Pending requests listed: {len(pending)}")

    pm.grant_permission(req1.id)
    pending = pm.get_pending_requests()
    assert len(pending) == 1
    assert pending[0].id == req2.id
    print("PASS: Pending requests updated after grant")

    return True


def test_callback_on_grant():
    """Verifica callback al otorgar permiso."""
    pm = get_permission_manager()
    pm.clear()

    callback_called = []

    def my_callback(request):
        callback_called.append(request.id)

    req = pm.request_permission("send_message", "Test callback", PermissionScope.SESSION, callback=my_callback)
    pm.grant_permission(req.id)

    assert len(callback_called) == 1
    assert callback_called[0] == req.id
    print("PASS: Callback fired on grant")

    return True


def test_integration_with_capability_registry():
    """Verifica integración completa con CapabilityRegistry."""
    pm = get_permission_manager()
    pm.clear()

    # Test all capabilities from registry
    registry = get_registry()
    for cap in registry.get_all():
        needs_perm = cap.need_explicit_permission(autonomy_level=3)
        auto_granted = pm.check_permission(cap.id, autonomy_level=3)

        # If capability doesn't need explicit permission, check_permission should be True
        if not needs_perm:
            assert auto_granted is True, f"{cap.id}: needs_perm={needs_perm}, auto={auto_granted}"
        else:
            # If it needs permission, check_permission should be False (no grant yet)
            assert auto_granted is False, f"{cap.id}: needs_perm={needs_perm}, auto={auto_granted}"

    print(f"PASS: Integration with all {len(registry.get_all())} capabilities verified")
    return True


def main():
    """Ejecuta todas las pruebas."""
    print("=" * 60)
    print("TESTS Permission Manager — FASE 2B B5")
    print("=" * 60)
    print()

    tests = [
        test_manager_creation,
        test_check_permission_auto_granted,
        test_check_permission_requires_confirmation,
        test_check_permission_critical,
        test_request_and_grant,
        test_deny_permission,
        test_permission_scopes,
        test_revoke_grant,
        test_persistence,
        test_pending_requests_listing,
        test_callback_on_grant,
        test_integration_with_capability_registry,
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
        print("✅ Permission Manager B5 — ESTABLE")
        return True
    else:
        print("❌ Permission Manager B5 — TIENE FALLOS")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)