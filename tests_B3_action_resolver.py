"""Tests for Action Resolver (FASE 2B B3)."""

import sys
sys.path.insert(0, ".")

from agent.action_resolver import (
    ActionResolver,
    ParameterSpec,
    ParameterSource,
    ActionSpec,
    ResolutionError,
    get_action_resolver,
    initialize_action_resolver,
)
from agent.intent_router import (
    Intent,
    IntentType,
    RoutingPath,
    Entity,
    get_intent_router,
)
from core.capability_registry import get_registry


def test_resolver_carga():
    """Verifica que el resolver carga correctamente."""
    resolver = get_action_resolver()
    specs = resolver._tool_param_specs
    assert len(specs) > 0, "El resolver no debe estar vacío"
    print(f"PASS: resolver cargó {len(specs)} capability param specs")
    return True


def test_resolve_fast_path():
    """Verifica resolución para Fast Path capabilities."""
    resolver = get_action_resolver()
    router = get_intent_router()
    context_engine = __import__('agent.jarvis_context', fromlist=['get_context_engine']).get_context_engine()

    # Test OPEN_APP
    intent = router.classify("abre Chrome", autonomy_level=3)
    action_context = context_engine.get_action_context("open_app")
    action = resolver.resolve(intent, action_context)

    assert action.capability_id == "open_app"
    assert action.tool_name == "open_app"
    assert action.parameters.get("app") == "Chrome"
    assert action.timeout_ms > 0
    assert action.verification is not None
    print(f"PASS: resolve OPEN_APP → tool={action.tool_name}, params={action.parameters}")

    # Test YOUTUBE_PLAY
    intent = router.classify("pon despacito en youtube", autonomy_level=3)
    action_context = context_engine.get_action_context("youtube_play")
    action = resolver.resolve(intent, action_context)

    assert action.capability_id == "youtube_play"
    assert action.tool_name == "youtube_play"
    assert "despacito" in action.parameters.get("query", "")
    print(f"PASS: resolve YOUTUBE_PLAY → tool={action.tool_name}, params={action.parameters}")

    # Test SYSTEM_VOLUME_UP (now separate capability)
    intent = router.classify("sube el volumen", autonomy_level=3)
    action_context = context_engine.get_action_context("system_volume_up")
    action = resolver.resolve(intent, action_context)

    assert action.capability_id == "system_volume_up"
    assert action.tool_name == "computer_settings"
    print(f"PASS: resolve SYSTEM_VOLUME_UP → tool={action.tool_name}, params={action.parameters}")

    # Test SYSTEM_VOLUME_DOWN
    intent = router.classify("baja el volumen", autonomy_level=3)
    action_context = context_engine.get_action_context("system_volume_down")
    action = resolver.resolve(intent, action_context)

    assert action.capability_id == "system_volume_down"
    assert action.tool_name == "computer_settings"
    print(f"PASS: resolve SYSTEM_VOLUME_DOWN → tool={action.tool_name}, params={action.parameters}")

    return True


def test_resolve_agent_path():
    """Verifica resolución para Agent Path capabilities."""
    resolver = get_action_resolver()
    router = get_intent_router()
    context_engine = __import__('agent.jarvis_context', fromlist=['get_context_engine']).get_context_engine()

    # Test CODE_ASSISTANCE
    intent = router.classify("escribe un script en Python", autonomy_level=3)
    action_context = context_engine.get_action_context("code_assistance")
    action = resolver.resolve(intent, action_context)

    assert action.capability_id == "code_assistance"
    assert action.tool_name == "code_assistance"
    assert action.parameters.get("description") is not None
    print(f"PASS: resolve CODE_ASSISTANCE → tool={action.tool_name}, params={action.parameters}")

    # Test PROJECT_DEVELOPMENT
    intent = router.classify("crea una aplicación web", autonomy_level=3)
    action_context = context_engine.get_action_context("project_development")
    action = resolver.resolve(intent, action_context)

    assert action.capability_id == "project_development"
    assert action.tool_name == "project_development"
    print(f"PASS: resolve PROJECT_DEVELOPMENT → tool={action.tool_name}, params={action.parameters}")

    return True


def test_resolve_missing_parameters():
    """Verifica manejo de parámetros faltantes."""
    resolver = get_action_resolver()
    router = get_intent_router()
    context_engine = __import__('agent.jarvis_context', fromlist=['get_context_engine']).get_context_engine()

    # FILE_WRITE requires content (prompt) - should fail without it
    intent = router.classify("escribe archivo test.txt", autonomy_level=3)
    action_context = context_engine.get_action_context("file_write")

    try:
        action = resolver.resolve(intent, action_context)
        # Should have raised ResolutionError
        print("FAIL: should have raised ResolutionError for missing content")
        return False
    except ResolutionError as e:
        assert "content" in str(e.missing_params) or "Missing required parameters" in str(e)
        print(f"PASS: correctly raised ResolutionError for missing params: {e.missing_params}")

    return True


def test_fast_path_eligible():
    """Verifica detección de Fast Path eligibility."""
    resolver = get_action_resolver()
    router = get_intent_router()
    context_engine = __import__('agent.jarvis_context', fromlist=['get_context_engine']).get_context_engine()

    # OPEN_APP - Low risk, Fast Path, no confirmation → eligible
    intent = router.classify("abre Chrome", autonomy_level=3)
    action_context = context_engine.get_action_context("open_app")
    action = resolver.resolve(intent, action_context)

    assert action.metadata.get("fast_path_eligible") is True
    print(f"PASS: OPEN_APP fast_path_eligible = {action.metadata.get('fast_path_eligible')}")

    # SEND_MESSAGE - Medium risk, Agent Path, confirmation needed at low autonomy → not eligible
    # Just check the metadata without full resolution (which requires extra params)
    # We can test the capability directly
    from core.capability_registry import get_registry
    registry = get_registry()
    cap = registry.get("send_message")
    assert cap is not None
    # At autonomy 1, should require confirmation
    assert cap.need_explicit_permission(1) is True
    # At autonomy 4, should not require confirmation
    assert cap.need_explicit_permission(4) is False
    print(f"PASS: SEND_MESSAGE fast_path_eligible logic verified via capability")

    return True


def test_verification_plan():
    """Verifica construcción de plan de verificación."""
    resolver = get_action_resolver()
    router = get_intent_router()
    context_engine = __import__('agent.jarvis_context', fromlist=['get_context_engine']).get_context_engine()

    intent = router.classify("abre Chrome", autonomy_level=3)
    action_context = context_engine.get_action_context("open_app")
    action = resolver.resolve(intent, action_context)

    verification = action.verification
    assert verification is not None
    assert "method" in verification
    assert "cost" in verification
    assert verification["method"] == "process_exists"
    assert verification["cost"] == "low"
    print(f"PASS: verification plan = {verification}")

    return True


def test_timeout_estimation():
    """Verifica estimación de timeout."""
    resolver = get_action_resolver()
    router = get_intent_router()
    context_engine = __import__('agent.jarvis_context', fromlist=['get_context_engine']).get_context_engine()

    # Fast operations should have shorter timeouts
    intent = router.classify("qué hora es", autonomy_level=3)
    action_context = context_engine.get_action_context("get_time")
    action = resolver.resolve(intent, action_context)
    assert action.timeout_ms <= 10000
    print(f"PASS: GET_TIME timeout = {action.timeout_ms}ms")

    # Complex operations should have longer timeouts
    intent = router.classify("crea una aplicación web", autonomy_level=3)
    action_context = context_engine.get_action_context("project_development")
    action = resolver.resolve(intent, action_context)
    assert action.timeout_ms >= 60000
    print(f"PASS: PROJECT_DEVELOPMENT timeout = {action.timeout_ms}ms")

    return True


def test_parameter_sources():
    """Verifica diferentes fuentes de parámetros."""
    resolver = get_action_resolver()

    # Check that specs are registered for key capabilities
    assert "open_app" in resolver._tool_param_specs
    assert "youtube_play" in resolver._tool_param_specs
    assert "file_read" in resolver._tool_param_specs
    assert "web_search" in resolver._tool_param_specs

    # Check parameter spec structure
    open_app_specs = resolver._tool_param_specs["open_app"]
    assert len(open_app_specs) == 1
    assert open_app_specs[0].name == "app"
    assert open_app_specs[0].source == ParameterSource.ENTITY
    assert open_app_specs[0].entity_name == "app_name"
    print(f"PASS: OPEN_APP param spec = {open_app_specs[0].name}, source={open_app_specs[0].source.value}")

    # Check default parameter
    file_read_specs = resolver._tool_param_specs["file_read"]
    offset_spec = next(s for s in file_read_specs if s.name == "offset")
    assert offset_spec.source == ParameterSource.DEFAULT
    assert offset_spec.default == 1
    print(f"PASS: FILE_READ offset default = {offset_spec.default}")

    return True


def test_generation_tracking():
    """Verifica tracking de generación."""
    resolver = get_action_resolver()
    gen1 = resolver.get_generation()
    resolver.clear()
    gen2 = resolver.get_generation()
    assert gen2 > gen1
    print(f"PASS: generation tracking works: {gen1} -> {gen2}")
    return True


def main():
    """Ejecuta todas las pruebas."""
    print("=" * 60)
    print("TESTS Action Resolver — FASE 2B B3")
    print("=" * 60)
    print()

    tests = [
        test_resolver_carga,
        test_resolve_fast_path,
        test_resolve_agent_path,
        test_resolve_missing_parameters,
        test_fast_path_eligible,
        test_verification_plan,
        test_timeout_estimation,
        test_parameter_sources,
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
        print("✅ Action Resolver B3 — ESTABLE")
        return True
    else:
        print("❌ Action Resolver B3 — TIENE FALLOS")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)