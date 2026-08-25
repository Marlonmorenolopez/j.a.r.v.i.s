# tests_B1_capability_registry.py
"""Pruebas unitarias para Capability Registry (FASE 2B B1)."""

import sys
sys.path.insert(0, ".")

from core.capability_registry import (
    CapabilityRegistry,
    Capability,
    RiskLevel,
    VerificationCost,
    CapabilityStatus,
    LocalOrRemote,
    get_registry,
    query_capabilities,
)


def test_registry_carga():
    """Verifica que el registry carga correctamente."""
    registry = get_registry()
    caps = registry.get_all()
    assert len(caps) > 0, "El registry no debe estar vacío"
    print(f"PASS: registry cargó {len(caps)} capabilities")
    return True


def test_get_by_id():
    """Verifica que se puede obtener una capability por ID."""
    registry = get_registry()
    cap = registry.get("open_app")
    assert cap is not None, "open_app no debería ser None"
    assert cap.id == "open_app"
    assert cap.name_es == "Abrir aplicaciones"
    assert cap.domain == "system"
    assert cap.tools == ["open_app"]
    assert cap.risk_level == RiskLevel.LOW
    assert cap.requires_confirmation is False
    assert cap.rollback_possible is False
    assert cap.verification_method == "process_exists"
    assert cap.verification_cost == VerificationCost.LOW
    assert cap.local_or_remote == LocalOrRemote.LOCAL
    print(f"PASS: get('open_app') → id={cap.id}, domain={cap.domain}, risk={cap.risk_level.value}")
    return True


def test_query_by_domain():
    """Verifica consulta por dominio."""
    registry = get_registry()
    media_caps = registry.query(domain="media")
    assert len(media_caps) > 0, "Debería haber al menos 1 capability en media"
    for cap in media_caps:
        assert cap.domain == "media", f"{cap.id} no es domain=media"
    print(f"PASS: query(domain='media') → {len(media_caps)} capabilities")
    return True


def test_query_by_risk():
    """Verifica consulta por nivel de riesgo."""
    registry = get_registry()
    critical_caps = registry.query(risk_level=RiskLevel.CRITICAL)
    assert len(critical_caps) >= 1, "Debería haber al menos 1 capability CRITICAL"
    for cap in critical_caps:
        assert cap.risk_level == RiskLevel.CRITICAL
    print(f"PASS: query(risk_level=CRITICAL) → {len(critical_caps)} capabilities")
    return True


def test_actionable():
    """Verifica que is_actionable funciona correctamente."""
    registry = get_registry()

    # open_app (LOW, no requiere confirmación) → actionable en nivel 2
    cap = registry.get("open_app")
    assert cap.is_actionable(autonomy_level=2), "open_app debería ser actionable en nivel 2"
    assert cap.is_actionable(autonomy_level=3), "open_app debería ser actionable en nivel 3"
    assert not cap.is_actionable(autonomy_level=0), "open_app NO debería ser actionable en nivel 0"
    print("PASS: open_app is_actionable(nivel 2+) = True")

    # send_message (LOW, requires_confirmation=True) → NO actionable en nivel 2
    cap = registry.get("send_message")
    assert not cap.is_actionable(autonomy_level=2), "send_message requiere confirmación, no debería ser actionable en nivel 2"
    assert cap.is_actionable(autonomy_level=4), "send_message debería ser actionable en nivel 4 (ADVANCED)"
    print("PASS: send_message is_actionable(nivel 2) = False, (nivel 4) = True")

    # shutdown_jarvis (CRITICAL) → NUNCA actionable automáticamente
    cap = registry.get("system_shutdown")
    assert not cap.is_actionable(autonomy_level=5), "shutdown_jarvis es CRITICAL, nunca actionable automáticamente"
    assert cap.need_explicit_permission(autonomy_level=5), "shutdown_jarvis siempre necesita permiso explícito"
    print("PASS: shutdown_jarvis is_actionable(nivel 5) = False (CRITICAL safety rule)")

    return True


def test_need_explicit_permission():
    """Verifica que need_explicit_permission respeta la regla de seguridad."""
    registry = get_registry()

    # shutdown_jarvis → siempre True
    cap = registry.get("system_shutdown")
    assert cap.need_explicit_permission(autonomy_level=0)
    assert cap.need_explicit_permission(autonomy_level=2)
    assert cap.need_explicit_permission(autonomy_level=5)
    print("PASS: shutdown_jarvis necesita permiso explícito en TODOS los niveles")

    # send_message → necesita en niveles < 4
    cap = registry.get("send_message")
    assert cap.need_explicit_permission(autonomy_level=0)
    assert cap.need_explicit_permission(autonomy_level=2)
    assert cap.need_explicit_permission(autonomy_level=3)
    assert not cap.need_explicit_permission(autonomy_level=4)
    assert not cap.need_explicit_permission(autonomy_level=5)
    print("PASS: send_message necesita permiso en niveles < 4")

    # open_app → nunca necesita
    cap = registry.get("open_app")
    assert not cap.need_explicit_permission(autonomy_level=0)
    assert not cap.need_explicit_permission(autonomy_level=5)
    print("PASS: open_app nunca necesita permiso explícito")

    return True


def test_get_by_tool():
    """Verifica que get_by_tool funciona."""
    registry = get_registry()

    caps = registry.get_by_tool("youtube_play")
    assert len(caps) > 0, "Debería haber capabilities que usen youtube_play"
    for cap in caps:
        assert "youtube_play" in cap.tools
    print(f"PASS: get_by_tool('youtube_play') → {len(caps)} capabilities")

    caps = registry.get_by_tool("non_existent_tool_xyz")
    assert len(caps) == 0
    print("PASS: get_by_tool('non_existent') → 0 capabilities")

    return True


def test_verification_data():
    """Verifica que las capabilities tienen información de verificación."""
    registry = get_registry()

    # open_app: verificación por proceso
    cap = registry.get("open_app")
    assert cap.verification_method == "process_exists"
    assert cap.verification_cost == VerificationCost.LOW
    assert cap.verification_latency in ["0.5-2s", "1-2s"]
    print(f"PASS: open_app verification → method={cap.verification_method}, cost={cap.verification_cost.value}, latency={cap.verification_latency}")

    # shutdown_jarvis: verificación por proceso
    cap = registry.get("system_shutdown")
    assert cap.verification_method == "process_terminated"
    assert cap.verification_cost == VerificationCost.LOW
    print(f"PASS: shutdown_jarvis verification → method={cap.verification_method}, cost={cap.verification_cost.value}")

    # file_operations: verificación por archivo
    cap = registry.get("file_operations")
    assert cap.verification_method == "file_exists_size_hash"
    assert cap.verification_cost == VerificationCost.LOW
    print(f"PASS: file_operations verification → method={cap.verification_method}, cost={cap.verification_cost.value}")

    # project_development: verificación por pruebas (alta)
    cap = registry.get("project_development")
    assert cap.verification_method == "functional_test_or_review"
    assert cap.verification_cost == VerificationCost.HIGH
    print(f"PASS: project_development verification → method={cap.verification_method}, cost={cap.verification_cost.value}")

    return True


def test_metadata_completa():
    """Verifica que cada capability tiene todos los campos requeridos."""
    registry = get_registry()

    required_fields = [
        "id",
        "name_es",
        "domain",
        "tools",
        "risk_level",
        "requires_confirmation",
        "rollback_possible",
        "verification_method",
        "verification_cost",
        "verification_latency",
        "dependencies",
        "latency_hint",
        "local_or_remote",
    ]

    for cap in registry.get_all():
        for field in required_fields:
            assert hasattr(cap, field), f"Capability {cap.id} falta campo: {field}"

    print(f"PASS: todas las {len(registry.get_all())} capabilities tienen metadata completa")
    return True


def test_get_status():
    """Verifica get_status."""
    registry = get_registry()
    status = registry.get_status()
    assert status["total"] == 31
    assert status["by_risk"]["CRITICAL"] >= 1
    assert status["by_risk"]["NONE"] > 0
    assert status["by_risk"]["LOW"] > 0
    assert status["by_risk"]["MEDIUM"] > 0
    assert status["by_risk"]["HIGH"] > 0
    print(f"PASS: get_status() → total={status['total']}, risks={status['by_risk']}")
    return True


def test_fast_path_capabilities():
    """Verifica que hay capabilities adecuadas para Fast Path."""
    registry = get_registry()

    fast_path = registry.query(
        actionable=True,
        autonomy_level=2,
        risk_level=RiskLevel.NONE,
    )

    assert len(fast_path) >= 3, "Debería haber al menos 3 capabilities para Fast Path"
    fast_path_ids = [c.id for c in fast_path]
    print(f"PASS: Fast Path capabilities (NONE, actionable nivel 2): {fast_path_ids}")

    # Verifica que shutdown_jarvis NO está en Fast Path
    assert "system_shutdown" not in fast_path_ids
    print("PASS: shutdown_jarvis EXCLUIDO de Fast Path (CRITICAL)")

    return True


def test_regresion_herramientas():
    """Verifica que todas las 20 herramientas originales están mapeadas."""
    registry = get_registry()

    original_tools = [
        "open_app",
        "web_search",
        "weather_report",
        "send_message",
        "reminder",
        "youtube_play",  # was youtube_video, now split into multiple
        "screen_capture",  # was screen_process
        "computer_settings",
        "browser_navigation",  # was browser_control
        "file_operations",  # was file_controller
        "desktop_management",  # was desktop_control
        "code_assistance",  # was code_helper
        "project_development",  # was dev_agent
        "agent_task",
        "computer_control",
        "game_updates",  # was game_updater
        "flight_search",  # was flight_finder
        "file_operations",  # was file_processor (same as above)
        "system_shutdown",  # was shutdown_jarvis
        "memory_save",  # was save_memory
    ]

    for tool_name in original_tools:
        caps = registry.get_by_tool(tool_name)
        assert len(caps) > 0, f"Tool {tool_name} no está mapeado en ninguna capability"

    print("PASS: las 20 herramientas originales están mapeadas en capabilities")
    return True


def test_domain_coverage():
    """Verifica que los dominios principales están representados."""
    registry = get_registry()

    domains_expected = ["system", "web", "media", "communication", "coding"]

    for domain in domains_expected:
        caps = registry.query(domain=domain)
        assert len(caps) > 0, f"Domain {domain} no tiene capabilities"

    print(f"PASS: dominios principales ({domains_expected}) tienen capabilities")
    return True


def main():
    """Ejecuta todas las pruebas."""
    print("=" * 60)
    print("TESTS Capability Registry — FASE 2B B1")
    print("=" * 60)
    print()

    tests = [
        test_registry_carga,
        test_get_by_id,
        test_query_by_domain,
        test_query_by_risk,
        test_actionable,
        test_need_explicit_permission,
        test_get_by_tool,
        test_verification_data,
        test_metadata_completa,
        test_get_status,
        test_fast_path_capabilities,
        test_regresion_herramientas,
        test_domain_coverage,
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
        print("✅ Capability Registry B1 — ESTABLE")
        return True
    else:
        print("❌ Capability Registry B1 — TIENE FALLOS")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
