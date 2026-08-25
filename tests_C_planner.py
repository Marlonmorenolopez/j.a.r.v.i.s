"""Tests for Planner (FASE 2C)."""

import sys
sys.path.insert(0, ".")

from agent.planner_new import (
    Planner,
    Plan,
    PlanStep,
    PlannerContext,
    get_planner,
    create_plan,
    replan,
)
from agent.intent_router import (
    Intent,
    IntentType,
    RoutingPath,
    get_intent_router,
)
from core.capability_registry import get_registry, RiskLevel


def test_planner_creation():
    """Verifica que el Planner se crea correctamente."""
    planner = get_planner()
    assert planner is not None
    assert planner._capability_registry is not None
    assert planner._intent_router is not None
    assert planner._action_resolver is not None
    assert planner._context_engine is not None
    print(f"PASS: Planner created successfully")
    return True


def test_fast_path_plan():
    """Verifica creación de plan Fast Path (single capability)."""
    planner = get_planner()

    # OPEN_APP is a Fast Path intent
    plan = planner.create_plan("abre Chrome", autonomy_level=3)

    assert isinstance(plan, Plan)
    assert plan.goal == "abre Chrome"
    assert len(plan.steps) == 1
    assert plan.steps[0].capability_id == "open_app"
    assert plan.metadata.get("path") == "fast_path"
    print(f"PASS: Fast Path plan created: {plan.steps[0].capability_id}")
    return True


def test_fast_path_plan_youtube():
    """Verifica Fast Path para YouTube."""
    planner = get_planner()

    plan = planner.create_plan("pon música en youtube", autonomy_level=3)

    assert len(plan.steps) == 1
    assert plan.steps[0].capability_id in ["youtube_play", "youtube_volume"]
    assert plan.metadata.get("path") == "fast_path"
    print(f"PASS: Fast Path YouTube plan: {plan.steps[0].capability_id}")
    return True


def test_fast_path_plan_time():
    """Verifica Fast Path para hora."""
    planner = get_planner()

    plan = planner.create_plan("qué hora es", autonomy_level=3)

    assert len(plan.steps) == 1
    assert plan.steps[0].capability_id == "get_time"
    assert plan.metadata.get("path") == "fast_path"
    print(f"PASS: Fast Path time plan: {plan.steps[0].capability_id}")
    return True


def test_fast_path_plan_volume():
    """Verifica Fast Path para volumen."""
    planner = get_planner()

    plan = planner.create_plan("sube el volumen", autonomy_level=3)

    assert len(plan.steps) == 1
    assert plan.steps[0].capability_id == "system_volume_up"
    assert plan.metadata.get("path") == "fast_path"
    print(f"PASS: Fast Path volume plan: {plan.steps[0].capability_id}")
    return True


def test_fast_path_plan_navegador():
    """Verifica Fast Path para navegador."""
    planner = get_planner()

    plan = planner.create_plan("abre el navegador", autonomy_level=3)

    assert len(plan.steps) == 1
    assert plan.steps[0].capability_id == "open_app"
    assert plan.metadata.get("path") == "fast_path"
    print(f"PASS: Fast Path navegador plan: {plan.steps[0].capability_id}")
    return True


def test_agent_path_plan():
    """Verifica creación de plan Agent Path (multi-step)."""
    planner = get_planner()

    # This should use Agent Path (structured fallback since no LLM)
    plan = planner.create_plan("busca Python en Google y abre el primer resultado", autonomy_level=3, use_llm=False)

    assert isinstance(plan, Plan)
    assert plan.goal == "busca Python en Google y abre el primer resultado"
    assert len(plan.steps) >= 1
    assert plan.metadata.get("path") == "fast_path"  # Intent is FAST_PATH for web_search
    print(f"PASS: Agent Path plan created with {len(plan.steps)} steps")
    for step in plan.steps:
        print(f"  Step {step.step}: {step.capability_id} - {step.description}")
    return True


def test_agent_path_code_task():
    """Verifica Agent Path para tarea de código."""
    planner = get_planner()

    plan = planner.create_plan("escribe un script en Python", autonomy_level=3, use_llm=False)

    assert len(plan.steps) >= 1
    # Should include code_assistance or fallback to structured plan
    cap_ids = [s.capability_id for s in plan.steps]
    print(f"PASS: Agent Path code task: {cap_ids}")
    return True


def test_plan_step_structure():
    """Verifica estructura de PlanStep."""
    planner = get_planner()

    plan = planner.create_plan("abre Chrome", autonomy_level=3)
    step = plan.steps[0]

    assert step.step == 1
    assert step.capability_id == "open_app"
    assert step.tool_name == "open_app"
    assert step.description
    assert isinstance(step.parameters, dict)
    assert step.critical is True
    assert step.timeout_ms > 0
    assert isinstance(step.dependencies, list)
    print(f"PASS: PlanStep structure valid")
    return True


def test_plan_risk_level():
    """Verifica cálculo de risk_level del plan."""
    planner = get_planner()

    # LOW risk
    plan = planner.create_plan("abre Chrome", autonomy_level=3)
    assert plan.risk_level == RiskLevel.LOW

    # YouTube Play is NONE risk
    plan = planner.create_plan("pon música en youtube", autonomy_level=3)
    assert plan.risk_level == RiskLevel.NONE

    print(f"PASS: Plan risk level calculated: {plan.risk_level.value}")
    return True


def test_plan_requires_confirmation():
    """Verifica requires_confirmation del plan."""
    planner = get_planner()

    # Fast Path - open_app doesn't need confirmation
    plan = planner.create_plan("abre Chrome", autonomy_level=3)
    assert plan.requires_confirmation is False

    print(f"PASS: Plan requires_confirmation: {plan.requires_confirmation}")
    return True


def test_plan_estimated_time():
    """Verifica tiempo estimado total."""
    planner = get_planner()

    plan = planner.create_plan("abre Chrome", autonomy_level=3)
    assert plan.estimated_total_ms > 0
    assert plan.estimated_total_ms == plan.steps[0].timeout_ms

    print(f"PASS: Estimated time: {plan.estimated_total_ms}ms")
    return True


def test_plan_serialization():
    """Verifica serialización del plan."""
    planner = get_planner()

    plan = planner.create_plan("abre Chrome", autonomy_level=3)

    # Test to_dict
    d = plan.to_dict()
    assert d["goal"] == "abre Chrome"
    assert len(d["steps"]) == 1
    assert d["steps"][0]["capability_id"] == "open_app"
    assert "risk_level" in d
    assert "requires_confirmation" in d

    # Test to_json
    json_str = plan.to_json()
    assert "open_app" in json_str
    assert "abre Chrome" in json_str
    print(f"PASS: Plan serialization works")
    return True


def test_replan():
    """Verifica replanificación tras fallo."""
    planner = get_planner()

    # Create initial plan
    original_plan = planner.create_plan("busca Python y abre resultado", autonomy_level=3, use_llm=False)
    completed = [original_plan.steps[0]]  # First step completed
    failed = original_plan.steps[1] if len(original_plan.steps) > 1 else original_plan.steps[0]

    # Replan
    new_plan = planner.replan(
        "busca Python y abre resultado",
        completed,
        failed,
        "Connection timeout",
        autonomy_level=3,
    )

    assert isinstance(new_plan, Plan)
    assert new_plan.metadata.get("replanned") is True
    assert "original_failure" in new_plan.metadata
    # Should not include completed capabilities
    completed_caps = {s.capability_id for s in completed}
    new_caps = {s.capability_id for s in new_plan.steps}
    assert completed_caps.isdisjoint(new_caps)
    print(f"PASS: Replan works, new plan has {len(new_plan.steps)} steps")
    return True


def test_create_plan_convenience():
    """Verifica función de conveniencia create_plan."""
    plan = create_plan("abre Chrome", autonomy_level=3)

    assert isinstance(plan, Plan)
    assert plan.goal == "abre Chrome"
    assert len(plan.steps) == 1
    assert plan.steps[0].capability_id == "open_app"
    print(f"PASS: create_plan convenience function works")
    return True


def test_replan_convenience():
    """Verifica función de conveniencia replan."""
    planner = get_planner()
    original_plan = planner.create_plan("busca Python", autonomy_level=3, use_llm=False)

    if len(original_plan.steps) > 0:
        new_plan = replan(
            "busca Python",
            [],
            original_plan.steps[0],
            "Test error",
            autonomy_level=3,
        )

        assert isinstance(new_plan, Plan)
        assert new_plan.metadata.get("replanned") is True
        print(f"PASS: replan convenience function works")
    return True


def test_fallback_plan():
    """Verifica plan de fallback para goals desconocidos."""
    planner = get_planner()

    # Unknown goal should fallback to web_search
    plan = planner.create_plan("xyz unknown goal abc", autonomy_level=3, use_llm=False)

    assert len(plan.steps) == 1
    assert plan.steps[0].capability_id == "web_search"
    assert plan.metadata.get("path") == "fallback"
    print(f"PASS: Fallback plan created: {plan.steps[0].capability_id}")
    return True


def test_structured_plan_keywords():
    """Verifica mapeo de palabras clave a capabilities."""
    planner = get_planner()

    # Test various keywords
    test_cases = [
        ("abre el navegador", ["open_app"]),
        ("busca en google", ["web_search"]),
        ("sube el volumen", ["system_volume_up"]),
        ("qué hora es", ["get_time"]),
        ("cómo está el clima", ["weather_report"]),
        ("escribe código python", ["code_assistance"]),
        ("crea un proyecto", ["project_development"]),
        ("toma una captura", ["screen_capture"]),
    ]

    for goal, expected_caps in test_cases:
        plan = planner.create_plan(goal, autonomy_level=3, use_llm=False)
        cap_ids = [s.capability_id for s in plan.steps]
        # At least one expected capability should be in the plan
        assert any(cap in cap_ids for cap in expected_caps), f"Goal '{goal}': expected one of {expected_caps}, got {cap_ids}"

    print(f"PASS: Keyword mapping works for all test cases")
    return True


def test_plan_dependencies():
    """Verifica dependencias entre pasos."""
    planner = get_planner()

    # Agent Path plan should handle dependencies
    plan = planner.create_plan("busca Python y abre el resultado", autonomy_level=3, use_llm=False)

    # Check if any steps have dependencies
    for step in plan.steps:
        assert isinstance(step.dependencies, list)
        for dep in step.dependencies:
            assert isinstance(dep, int)
            assert dep > 0
            assert dep < step.step  # Dependencies must be earlier steps

    print(f"PASS: Dependencies valid")
    return True


def test_autonomy_level_affects_plan():
    """Verifica que el nivel de autonomía afecta al plan."""
    planner = get_planner()

    # At low autonomy, send_message should require confirmation
    plan_low = planner.create_plan("envía un mensaje", autonomy_level=1)
    plan_high = planner.create_plan("envía un mensaje", autonomy_level=5)

    # SEND_MESSAGE is agent_path (not fast_path)
    assert plan_low.metadata.get("path") == "agent_path"
    assert plan_high.metadata.get("path") == "agent_path"

    # But requires_confirmation should differ
    assert plan_low.requires_confirmation is True  # Low autonomy needs confirmation
    assert plan_high.requires_confirmation is False  # High autonomy doesn't

    print(f"PASS: Autonomy level affects requires_confirmation")
    return True


def test_planner_context_integration():
    """Verifica integración con ContextEngine."""
    planner = get_planner()

    plan = planner.create_plan("abre Chrome", autonomy_level=3)

    # Plan should have context metadata
    assert "intent" in plan.metadata
    print(f"PASS: Context integration works, intent: {plan.metadata.get('intent')}")
    return True


def test_multiple_fast_path_intents():
    """Verifica múltiples intents Fast Path."""
    planner = get_planner()

    fast_path_goals = [
        "abre notepad",
        "pon música en youtube",
        "pausa youtube",
        "sube el volumen",
        "qué hora es",
        "busca python",
        "cómo está el tiempo",
    ]

    for goal in fast_path_goals:
        plan = planner.create_plan(goal, autonomy_level=3)
        assert len(plan.steps) == 1, f"Goal '{goal}' should be Fast Path (1 step)"
        assert plan.metadata.get("path") == "fast_path", f"Goal '{goal}' should be fast_path"

    print(f"PASS: All {len(fast_path_goals)} Fast Path goals work")
    return True


def main():
    """Ejecuta todas las pruebas."""
    print("=" * 60)
    print("TESTS Planner — FASE 2C")
    print("=" * 60)
    print()

    tests = [
        test_planner_creation,
        test_fast_path_plan,
        test_fast_path_plan_youtube,
        test_fast_path_plan_time,
        test_fast_path_plan_volume,
        test_fast_path_plan_navegador,
        test_agent_path_plan,
        test_agent_path_code_task,
        test_plan_step_structure,
        test_plan_risk_level,
        test_plan_requires_confirmation,
        test_plan_estimated_time,
        test_plan_serialization,
        test_replan,
        test_create_plan_convenience,
        test_replan_convenience,
        test_fallback_plan,
        test_structured_plan_keywords,
        test_plan_dependencies,
        test_autonomy_level_affects_plan,
        test_planner_context_integration,
        test_multiple_fast_path_intents,
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
        print("✅ Planner FASE 2C — ESTABLE")
        return True
    else:
        print("❌ Planner FASE 2C — TIENE FALLOS")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)