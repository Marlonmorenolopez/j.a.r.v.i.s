"""Tests for Intent Router (FASE 2B B2)."""

import sys
sys.path.insert(0, ".")

from agent.intent_router import (
    IntentRouter,
    IntentType,
    RoutingPath,
    Entity,
    Intent,
    get_intent_router,
    initialize_intent_router,
)


def test_router_carga():
    """Verifica que el router carga correctamente."""
    router = get_intent_router()
    intents = router.list_intents()
    assert len(intents) > 0, "El router no debe estar vacío"
    print(f"PASS: router cargó {len(intents)} intents")
    return True


def test_get_intent_pattern():
    """Verifica que se puede obtener un pattern por nombre."""
    router = get_intent_router()
    pattern = router.get_intent_pattern("OPEN_APP")
    assert pattern is not None, "OPEN_APP no debería ser None"
    assert pattern.intent_name == "OPEN_APP"
    assert pattern.intent_type == IntentType.SYSTEM_CONTROL
    assert pattern.path == RoutingPath.FAST_PATH
    print(f"PASS: get_intent_pattern('OPEN_APP') → type={pattern.intent_type.value}, path={pattern.path.value}")
    return True


def test_classify_fast_path():
    """Verifica clasificación para Fast Path."""
    router = get_intent_router()

    # Test OPEN_APP
    intent = router.classify("abre Chrome", autonomy_level=3)
    assert intent.name == "OPEN_APP"
    assert intent.path == RoutingPath.FAST_PATH
    assert intent.type == IntentType.SYSTEM_CONTROL
    assert intent.confidence > 0.5
    assert "app_name" in [e.name for e in intent.entities]
    print(f"PASS: 'abre Chrome' → {intent.name}, path={intent.path.value}, entities={[e.name for e in intent.entities]}")

    # Test YOUTUBE_PLAY
    intent = router.classify("pon música en youtube", autonomy_level=3)
    assert intent.name == "YOUTUBE_PLAY"
    assert intent.path == RoutingPath.FAST_PATH
    assert "query" in [e.name for e in intent.entities]
    print(f"PASS: 'pon música en youtube' → {intent.name}, entities={[e.name for e in intent.entities]}")

    # Test SYSTEM_VOLUME_UP
    intent = router.classify("sube el volumen", autonomy_level=3)
    assert intent.name == "SYSTEM_VOLUME_UP"
    assert intent.path == RoutingPath.FAST_PATH
    assert "direction" in [e.name for e in intent.entities]
    print(f"PASS: 'sube el volumen' → {intent.name}, direction={[e.value for e in intent.entities]}")

    # Test GET_TIME
    intent = router.classify("qué hora es", autonomy_level=3)
    assert intent.name == "GET_TIME"
    assert intent.path == RoutingPath.FAST_PATH
    print(f"PASS: 'qué hora es' → {intent.name}")

    # Test WEB_SEARCH
    intent = router.classify("busca Python en Google", autonomy_level=3)
    assert intent.name == "WEB_SEARCH"
    assert intent.path == RoutingPath.FAST_PATH
    assert "query" in [e.name for e in intent.entities]
    print(f"PASS: 'busca Python en Google' → {intent.name}")

    # Test YOUTUBE_PAUSE
    intent = router.classify("pausa youtube", autonomy_level=3)
    assert intent.name == "YOUTUBE_PAUSE"
    assert intent.path == RoutingPath.FAST_PATH
    print(f"PASS: 'pausa youtube' → {intent.name}")

    return True


def test_classify_agent_path():
    """Verifica clasificación para Agent Path."""
    router = get_intent_router()

    # Test CODE_ASSISTANCE
    intent = router.classify("escribe un script en Python", autonomy_level=3)
    assert intent.name == "CODE_ASSISTANCE"
    assert intent.path == RoutingPath.AGENT_PATH
    print(f"PASS: 'escribe un script en Python' → {intent.name}, path={intent.path.value}")

    # Test PROJECT_DEVELOPMENT
    intent = router.classify("crea una aplicación web", autonomy_level=3)
    assert intent.name == "PROJECT_DEVELOPMENT"
    assert intent.path == RoutingPath.AGENT_PATH
    print(f"PASS: 'crea una aplicación web' → {intent.name}, path={intent.path.value}")

    # Test AGENT_TASK
    intent = router.classify("ejecuta la tarea de backup", autonomy_level=3)
    assert intent.name == "AGENT_TASK"
    assert intent.path == RoutingPath.AGENT_PATH
    print(f"PASS: 'ejecuta la tarea de backup' → {intent.name}, path={intent.path.value}")

    return True


def test_classify_clarify():
    """Verifica clasificación para Clarify (ambiguous)."""
    router = get_intent_router()

    # Test SYSTEM_SHUTDOWN (always clarify)
    intent = router.classify("apaga jarvis", autonomy_level=3)
    assert intent.name == "SYSTEM_SHUTDOWN"
    assert intent.path == RoutingPath.CLARIFY
    print(f"PASS: 'apaga jarvis' → {intent.name}, path={intent.path.value}")

    # Test HELP
    intent = router.classify("ayuda", autonomy_level=3)
    assert intent.name == "HELP"
    assert intent.path == RoutingPath.CLARIFY
    print(f"PASS: 'ayuda' → {intent.name}, path={intent.path.value}")

    # Test UNKNOWN
    intent = router.classify("xyz random text abc", autonomy_level=3)
    assert intent.name == "UNKNOWN"
    assert intent.path == RoutingPath.CLARIFY
    print(f"PASS: 'xyz random text abc' → {intent.name}, path={intent.path.value}")

    return True


def test_autonomy_gating():
    """Verifica que la autonomía afecta requires_confirmation."""
    router = get_intent_router()

    # SEND_MESSAGE requires confirmation at low autonomy
    intent = router.classify("envía un mensaje a Juan", autonomy_level=1)
    assert intent.requires_confirmation is True
    print(f"PASS: 'envía un mensaje' nivel 1 → requires_confirmation={intent.requires_confirmation}")

    intent = router.classify("envía un mensaje a Juan", autonomy_level=4)
    assert intent.requires_confirmation is False
    print(f"PASS: 'envía un mensaje' nivel 4 → requires_confirmation={intent.requires_confirmation}")

    # CRITICAL always requires confirmation
    intent = router.classify("apaga jarvis", autonomy_level=5)
    assert intent.requires_confirmation is True
    print(f"PASS: 'apaga jarvis' nivel 5 → requires_confirmation={intent.requires_confirmation} (CRITICAL rule)")

    return True


def test_entity_extraction():
    """Verifica extracción de entidades."""
    router = get_intent_router()

    # App name extraction
    intent = router.classify("abre notepad", autonomy_level=3)
    app_entities = [e for e in intent.entities if e.name == "app_name"]
    assert len(app_entities) == 1
    assert "notepad" in app_entities[0].value.lower()
    print(f"PASS: app_name extracted: {app_entities[0].value}")

    # YouTube query extraction
    intent = router.classify("reproduce despacito en youtube", autonomy_level=3)
    query_entities = [e for e in intent.entities if e.name == "query"]
    assert len(query_entities) == 1
    assert "despacito" in query_entities[0].value.lower()
    print(f"PASS: query extracted: {query_entities[0].value}")

    # Volume direction extraction
    intent = router.classify("baja el volumen", autonomy_level=3)
    dir_entities = [e for e in intent.entities if e.name == "direction"]
    assert len(dir_entities) == 1
    assert dir_entities[0].value == "down"
    print(f"PASS: direction extracted: {dir_entities[0].value}")

    # URL extraction
    intent = router.classify("abre https://github.com", autonomy_level=3)
    url_entities = [e for e in intent.entities if e.name == "url"]
    assert len(url_entities) == 1
    assert "github.com" in url_entities[0].value
    print(f"PASS: url extracted: {url_entities[0].value}")

    return True


def test_confidence_calculation():
    """Verifica cálculo de confianza."""
    router = get_intent_router()

    # High confidence for exact matches
    intent = router.classify("abre Chrome", autonomy_level=3)
    assert intent.confidence >= 0.8
    print(f"PASS: confidence for exact match: {intent.confidence}")

    # Lower confidence for partial
    intent = router.classify("por favor abre Chrome ahora", autonomy_level=3)
    assert intent.confidence >= 0.7
    print(f"PASS: confidence for partial match: {intent.confidence}")

    return True


def test_intent_serialization():
    """Verifica serialización de Intent."""
    router = get_intent_router()
    intent = router.classify("abre Chrome", autonomy_level=3)
    data = intent.to_dict()

    assert data["name"] == "OPEN_APP"
    assert data["type"] == "system_control"
    assert data["path"] == "fast_path"
    assert "entities" in data
    assert len(data["entities"]) > 0
    print(f"PASS: Intent serialization works")
    return True


def test_generation_tracking():
    """Verifica tracking de generación para cache invalidation."""
    router = get_intent_router()
    gen1 = router.get_generation()
    router.clear()
    gen2 = router.get_generation()
    assert gen2 > gen1
    print(f"PASS: generation tracking works: {gen1} -> {gen2}")
    return True


def main():
    """Ejecuta todas las pruebas."""
    print("=" * 60)
    print("TESTS Intent Router — FASE 2B B2")
    print("=" * 60)
    print()

    tests = [
        test_router_carga,
        test_get_intent_pattern,
        test_classify_fast_path,
        test_classify_agent_path,
        test_classify_clarify,
        test_autonomy_gating,
        test_entity_extraction,
        test_confidence_calculation,
        test_intent_serialization,
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
        print("✅ Intent Router B2 — ESTABLE")
        return True
    else:
        print("❌ Intent Router B2 — TIENE FALLOS")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)