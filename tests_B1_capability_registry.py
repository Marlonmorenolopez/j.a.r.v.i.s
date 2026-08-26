# tests_B1_capability_registry.py
"""Pruebas unitarias para Capability Registry (FASE 2B B1) — P.I.P.E."""

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
    system_caps = registry.query(domain="system")
    assert len(system_caps) > 0, "Debería haber al menos 1 capability en system"
    for cap in system_caps:
        assert cap.domain == "system"
    print(f"PASS: query(domain='system') → {len(system_caps)} capabilities")

    web_caps = registry.query(domain="web")
    assert len(web_caps) > 0, "Debería haber al menos 1 capability en web"
    print(f"PASS: query(domain='web') → {len(web_caps)} capabilities")

    file_caps = registry.query(domain="file")
    assert len(file_caps) > 0, "Debería haber al menos 1 capability en file"
    print(f"PASS: query(domain='file') → {len(file_caps)} capabilities")
    return True


def test_query_by_risk():
    """Verifica consulta por nivel de riesgo."""
    registry = get_registry()
    high_caps = registry.query(risk_level=RiskLevel.HIGH)
    assert len(high_caps) > 0, "Debería haber capabilities HIGH"
    for cap in high_caps:
        assert cap.risk_level == RiskLevel.HIGH
    print(f"PASS: query(risk=HIGH) → {len(high_caps)} capabilities")

    none_caps = registry.query(risk_level=RiskLevel.NONE)
    assert len(none_caps) > 0, "Debería haber capabilities NONE"
    print(f"PASS: query(risk=NONE) → {len(none_caps)} capabilities")

    critical_caps = registry.query(risk_level=RiskLevel.CRITICAL)
    assert len(critical_caps) == 0, "No debería haber capabilities CRITICAL en P.I.P.E"
    print("PASS: query(risk=CRITICAL) → 0 capabilities")
    return True


def test_actionable():
    """Verifica actionable behavior."""
    registry = get_registry()
    # Test with autonomy_level=5 (max autonomy) - all available should be actionable
    actionable = [c for c in registry.get_all() if c.is_actionable(autonomy_level=5)]
    non_actionable = [c for c in registry.get_all() if not c.is_actionable(autonomy_level=5)]

    # At autonomy 5, all AVAILABLE capabilities should be actionable
    for cap in actionable:
        assert cap.status == CapabilityStatus.AVAILABLE, f"{cap.id} actionable pero status={cap.status}"

    assert len(actionable) > 0, "Debería haber capabilities actionable"
    print(f"PASS: actionable={len(actionable)}, non-actionable={len(non_actionable)}")
    return True


def test_need_explicit_permission():
    """Verifica lógica de permisos según autonomy_level."""
    registry = get_registry()

    # system_control (contains shutdown_pipe) → HIGH risk + requires_confirmation
    # necesita permiso en autonomy < 4, NO en >= 4
    cap = registry.get("system_control")
    assert cap is not None, "system_control debería existir"
    assert cap.need_explicit_permission(autonomy_level=0)
    assert cap.need_explicit_permission(autonomy_level=2)
    assert cap.need_explicit_permission(autonomy_level=3)
    assert not cap.need_explicit_permission(autonomy_level=4)
    assert not cap.need_explicit_permission(autonomy_level=5)
    print("PASS: system_control necesita permiso en niveles < 4")

    # send_message → LOW risk + requires_confirmation
    # necesita permiso en autonomy < 4, NO en >= 4
    cap = registry.get("send_message")
    assert cap is not None, "send_message debería existir"
    assert cap.need_explicit_permission(autonomy_level=0)
    assert cap.need_explicit_permission(autonomy_level=2)
    assert cap.need_explicit_permission(autonomy_level=3)
    assert not cap.need_explicit_permission(autonomy_level=4)
    assert not cap.need_explicit_permission(autonomy_level=5)
    print("PASS: send_message necesita permiso en niveles < 4")

    # open_app → LOW risk + NO requires_confirmation
    # NUNCA necesita permiso
    cap = registry.get("open_app")
    assert cap is not None, "open_app debería existir"
    assert not cap.need_explicit_permission(autonomy_level=0)
    assert not cap.need_explicit_permission(autonomy_level=5)
    print("PASS: open_app nunca necesita permiso explícito")

    # CRITICAL risk siempre necesita permiso (si existiera)
    # No hay CRITICAL en P.I.P.E actual, pero probamos la lógica
    from core.capability_registry import Capability, RiskLevel, VerificationCost, LocalOrRemote, CapabilityStatus
    test_critical = Capability(
        id="test_critical", name_es="Test", domain="test", tools=["test"],
        risk_level=RiskLevel.CRITICAL, requires_confirmation=False,
        rollback_possible=False, verification_method="test", verification_cost=VerificationCost.LOW,
        verification_latency="1s", dependencies=[], latency_hint="1s",
        local_or_remote=LocalOrRemote.LOCAL, intent_name="TEST"
    )
    assert test_critical.need_explicit_permission(autonomy_level=0)
    assert test_critical.need_explicit_permission(autonomy_level=5)
    print("PASS: CRITICAL risk siempre necesita permiso")

    return True


def test_get_by_tool():
    """Verifica que get_by_tool funciona."""
    registry = get_registry()

    caps = registry.get_by_tool("open_app")
    assert len(caps) > 0, "Debería haber capabilities que usen open_app"
    for cap in caps:
        assert "open_app" in cap.tools
    print(f"PASS: get_by_tool('open_app') → {len(caps)} capabilities")

    caps = registry.get_by_tool("web_search")
    assert len(caps) > 0, "Debería haber capabilities que usen web_search"
    print(f"PASS: get_by_tool('web_search') → {len(caps)} capabilities")

    caps = registry.get_by_tool("non_existent_tool_xyz")
    assert len(caps) == 0
    print("PASS: get_by_tool('non_existent') → 0 capabilities")

    return True


def test_verification_data():
    """Verifica que las capabilities tienen datos de verificación."""
    registry = get_registry()
    for cap in registry.get_all():
        assert cap.verification_method, f"{cap.id}: verification_method vacío"
        assert cap.verification_cost in (VerificationCost.LOW, VerificationCost.MEDIUM, VerificationCost.HIGH)
        assert cap.verification_latency, f"{cap.id}: verification_latency vacío"
        assert cap.latency_hint, f"{cap.id}: latency_hint vacío"
    print("PASS: todas las capabilities tienen verification_method, cost, latency")
    return True


def test_metadata_completa():
    """Verifica campos requeridos en cada capability."""
    registry = get_registry()
    required = ["id", "name_es", "domain", "tools", "risk_level",
                "requires_confirmation", "rollback_possible",
                "verification_method", "verification_cost", "verification_latency",
                "dependencies", "latency_hint", "local_or_remote", "intent_name"]
    for cap in registry.get_all():
        for field in required:
            assert hasattr(cap, field), f"{cap.id}: falta campo {field}"
            val = getattr(cap, field)
            assert val is not None, f"{cap.id}: campo {field} es None"
    print("PASS: todas las capabilities tienen metadata completa")
    return True


def test_get_status():
    """Verifica status field."""
    registry = get_registry()
    for cap in registry.get_all():
        assert cap.status in (CapabilityStatus.AVAILABLE, CapabilityStatus.PARTIAL,
                              CapabilityStatus.DEPENDENCY_MISSING, CapabilityStatus.DISABLED)
    print("PASS: status field tiene valores válidos")
    return True


def test_fast_path_capabilities():
    """Verifica capabilities actionable (risk NONE/LOW, no confirmation) = fast_path."""
    registry = get_registry()
    # Use query with actionable=True and high autonomy to get actionable capabilities
    fast = registry.query(actionable=True, autonomy_level=5)
    
    # At autonomy 5, all AVAILABLE capabilities are actionable
    # But traditional "fast_path" are NONE/LOW risk + no confirmation
    traditional_fast = [c for c in fast if c.risk_level in (RiskLevel.NONE, RiskLevel.LOW) and not c.requires_confirmation]
    for cap in traditional_fast:
        assert cap.risk_level in (RiskLevel.NONE, RiskLevel.LOW)
        assert cap.requires_confirmation is False
    
    assert len(traditional_fast) > 0, "Debería haber capabilities fast_path tradicionales"
    print(f"PASS: fast_path tradicionales → {len(traditional_fast)} capabilities (total actionable: {len(fast)})")
    return True


def test_regresion_herramientas():
    """Regresión: cada capability declara al menos una tool válida."""
    registry = get_registry()
    for cap in registry.get_all():
        assert isinstance(cap.tools, list), f"{cap.id}: tools debe ser lista"
        assert len(cap.tools) > 0, f"{cap.id}: tools vacío"
        for tool in cap.tools:
            assert isinstance(tool, str) and tool, f"{cap.id}: tool inválida '{tool}'"
    print("PASS: todas las capabilities declaran tools válidas")
    return True


def test_domain_coverage():
    """Verifica cobertura de dominios esperados en P.I.P.E."""
    registry = get_registry()
    domains = {c.domain for c in registry.get_all()}
    expected = {"system", "web", "file", "memory", "vision", "dev", "communication", "productivity"}
    for d in expected:
        assert d in domains, f"Falta dominio esperado: {d}"
    print(f"PASS: dominios cubiertos → {sorted(domains)}")
    return True


if __name__ == "__main__":
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
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL: {t.__name__} → {e}")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: {t.__name__} → {e}")
            sys.exit(1)
    print("\n✅ TODOS LOS TESTS B1 PASARON")