"""Planner — Task planning module for P.I.P.E (FASE 2C).

This module creates execution plans for complex goals using the Agent Path.
It integrates with:
- Intent Router: to understand the goal
- Capability Registry: to discover available capabilities
- Context Engine: to get conversation context and history
- Action Resolver: to understand tool parameters

The Planner replaces the old LLM-based planner with a structured approach
that uses the new core components for better reliability and verification.
"""

from __future__ import annotations

import json
import logging
import sys
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from core.capability_registry import (
    Capability,
    RiskLevel,
    get_registry,
)
from agent.intent_router import (
    Intent,
    IntentType,
    RoutingPath,
    get_intent_router,
)
from agent.action_resolver import (
    ActionSpec,
    get_action_resolver,
    ResolutionError,
)
from agent.pipe_context import (
    ContextEngine,
    get_context_engine,
)

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass(frozen=True)
class PlanStep:
    """A single step in an execution plan."""
    step: int
    capability_id: str
    tool_name: str
    description: str
    parameters: Dict[str, Any]
    critical: bool = True
    timeout_ms: int = 30000
    verification: Optional[Dict[str, Any]] = None
    dependencies: List[int] = field(default_factory=list)  # Step numbers this depends on
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Plan:
    """Complete execution plan."""
    goal: str
    steps: List[PlanStep]
    estimated_total_ms: int = 0
    risk_level: RiskLevel = RiskLevel.NONE
    requires_confirmation: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging/debugging."""
        return {
            "goal": self.goal,
            "steps": [
                {
                    "step": s.step,
                    "capability_id": s.capability_id,
                    "tool_name": s.tool_name,
                    "description": s.description,
                    "parameters": s.parameters,
                    "critical": s.critical,
                    "timeout_ms": s.timeout_ms,
                    "verification": s.verification,
                    "dependencies": s.dependencies,
                }
                for s in self.steps
            ],
            "estimated_total_ms": self.estimated_total_ms,
            "risk_level": self.risk_level.value,
            "requires_confirmation": self.requires_confirmation,
        }

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


@dataclass
class PlannerContext:
    """Context package passed to planner from ContextEngine."""
    session: Optional[Dict[str, Any]]
    recent_messages: List[Dict[str, Any]]
    session_context: Dict[str, Any]
    user_context: Dict[str, Any]
    system_context: Dict[str, Any]
    temporary_context: Dict[str, Any]
    capabilities_summary: List[Dict[str, Any]]
    timestamp: float


# ============================================================================
# PLANNER CLASS
# ============================================================================

class Planner:
    """Task planner for P.I.P.E Agent Path.

    Creates structured plans from user goals by:
    1. Analyzing the goal with Intent Router
    2. Discovering relevant capabilities from CapabilityRegistry
    3. Building a step-by-step plan using ActionResolver for parameter specs
    4. Incorporating context from ContextEngine
    """

    def __init__(self):
        self._capability_registry = get_registry()
        self._intent_router = get_intent_router()
        self._action_resolver = get_action_resolver()
        self._context_engine = get_context_engine()

    def create_plan(
        self,
        goal: str,
        autonomy_level: int = 3,
        use_llm: bool = True,
    ) -> Plan:
        """Create an execution plan for a user goal.

        Args:
            goal: User's natural language goal
            autonomy_level: Current autonomy level (0-5)
            use_llm: Whether to use LLM for complex planning (Agent Path)

        Returns:
            Plan with ordered steps
        """
        logger.info("Creating plan for goal: %s", goal[:80])

        # 1. Classify intent to understand the goal
        intent = self._intent_router.classify(goal, autonomy_level=autonomy_level)

        # 2. Get context from ContextEngine
        context = self._context_engine.get_planner_context()

        # 3. Determine if this is Fast Path or Agent Path
        if intent.path == RoutingPath.FAST_PATH and intent.capability_id:
            # Fast Path: single capability, direct execution
            return self._create_fast_path_plan(intent, goal, autonomy_level)

        # 4. Agent Path: complex multi-step planning
        return self._create_agent_path_plan(goal, intent, context, autonomy_level, use_llm)

    def _create_fast_path_plan(
        self,
        intent: Intent,
        goal: str,
        autonomy_level: int,
    ) -> Plan:
        """Create a single-step plan for Fast Path execution."""
        capability_id = intent.capability_id
        capability = self._capability_registry.get(capability_id)

        if not capability:
            # Fallback: create a minimal plan
            return self._fallback_plan(goal)

        # Resolve action to get tool and parameters
        try:
            context = self._context_engine.get_action_context(capability_id)
            action_spec = self._action_resolver.resolve(intent, context)
        except ResolutionError as e:
            logger.warning("Action resolution failed for Fast Path: %s", e)
            return self._fallback_plan(goal)

        step = PlanStep(
            step=1,
            capability_id=capability_id,
            tool_name=action_spec.tool_name,
            description=f"Execute {capability.name_es}",
            parameters=action_spec.parameters,
            critical=True,
            timeout_ms=action_spec.timeout_ms,
            verification=action_spec.verification,
            metadata=action_spec.metadata,
        )

        return Plan(
            goal=goal,
            steps=[step],
            estimated_total_ms=action_spec.timeout_ms,
            risk_level=capability.risk_level,
            requires_confirmation=intent.requires_confirmation,
            metadata={"path": "fast_path", "intent": intent.name},
        )

    def _create_agent_path_plan(
        self,
        goal: str,
        initial_intent: Intent,
        context: Dict[str, Any],
        autonomy_level: int,
        use_llm: bool,
    ) -> Plan:
        """Create a multi-step plan for Agent Path execution."""
        # Try LLM-based planning first
        if use_llm:
            try:
                return self._llm_plan(goal, context, autonomy_level)
            except Exception as e:
                logger.warning("LLM planning failed, falling back to structured: %s", e)

        # Fallback: structured rule-based planning
        return self._structured_plan(goal, context, autonomy_level, initial_intent)

    def _llm_plan(
        self,
        goal: str,
        context: PlannerContext,
        autonomy_level: int,
    ) -> Plan:
        """Create plan using LLM (Gemini)."""
        import google.generativeai as genai
        from core.config_loader import get_gemini_api_key

        genai.configure(api_key=get_gemini_api_key())

        # Build capability descriptions for the prompt
        capabilities = context.capabilities_summary
        cap_descriptions = []
        for cap in capabilities:
            cap_descriptions.append(
                f"- {cap['id']}: {cap['name_es']} (domain: {cap['domain']}, "
                f"risk: {cap['risk_level']}, tools: {cap['tools']})"
            )

        # Build prompt
        prompt = self._build_planner_prompt(goal, context, cap_descriptions, autonomy_level)

        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=prompt,
        )

        response = model.generate_content(f"Goal: {goal}")
        text = response.text.strip()
        text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()

        plan_data = json.loads(text)
        return self._parse_plan(goal, plan_data, autonomy_level)

    def _build_planner_prompt(
        self,
        goal: str,
        context: PlannerContext,
        cap_descriptions: List[str],
        autonomy_level: int,
    ) -> str:
        """Build the planner system prompt."""
        recent_msgs = context.recent_messages[-5:] if context.recent_messages else []
        msg_context = "\n".join(f"  {m['role']}: {m['content'][:100]}" for m in recent_msgs)

        caps_text = "\n".join(cap_descriptions)

        return f"""You are the Planner for P.I.P.E, a universal desktop agent.
Your job: break any user goal into a sequence of steps using ONLY the capabilities listed below.

ABSOLUTE RULES:
- NEVER use capabilities not listed below
- Each step must be a single capability execution
- Max 8 steps. Use minimum steps needed.
- Steps can depend on previous steps (use dependencies field)
- Consider risk levels: CRITICAL capabilities always need explicit user confirmation
- Autonomy level: {autonomy_level} (0=CHAT, 1=SUGGEST, 2=CONFIRM, 3=SUPERVISED, 4=ADVANCED, 5=FULL)

AVAILABLE CAPABILITIES:
{caps_text}

RECENT CONVERSATION CONTEXT:
{msg_context or "  (none)"}

USER CONTEXT (preferences, memory):
{json.dumps(context.user_context, ensure_ascii=False)[:500] if context.user_context else "  (none)"}

OUTPUT — return ONLY valid JSON, no markdown, no explanation, no code blocks:
{{
  "goal": "...",
  "steps": [
    {{
      "step": 1,
      "capability_id": "capability_id_from_list",
      "description": "what this step does",
      "parameters": {{}},
      "critical": true,
      "dependencies": []
    }}
  ]
}}

IMPORTANT: The 'parameters' field should contain the parameters needed for the capability's primary tool.
The ActionResolver will fill in details from context/entities.
"""

    def _structured_plan(
        self,
        goal: str,
        context: Dict[str, Any],
        autonomy_level: int,
        initial_intent: Intent = None,
    ) -> Plan:
        """Create plan using structured rules (no LLM)."""
        # Analyze goal for keywords and map to capabilities
        goal_lower = goal.lower()

        # If we have an initial intent with a capability_id, use that as primary
        if initial_intent and initial_intent.capability_id:
            # Check if we can find the capability
            cap = next((c for c in context.get("capabilities_summary", []) if c["id"] == initial_intent.capability_id), None)
            if cap:
                candidate_caps = [cap]
            else:
                candidate_caps = self._find_relevant_capabilities(goal_lower, context.get("capabilities_summary", []))
        else:
            # Determine relevant capabilities based on keywords
            candidate_caps = self._find_relevant_capabilities(goal_lower, context.get("capabilities_summary", []))

        if not candidate_caps:
            return self._fallback_plan(goal)

        # Build steps from candidate capabilities
        steps = []
        step_num = 1

        for cap in candidate_caps[:5]:  # Max 5 steps
            try:
                intent = self._intent_router.classify(goal, autonomy_level=autonomy_level)
                action_context = self._context_engine.get_action_context(cap["id"])
                action_spec = self._action_resolver.resolve(intent, action_context)

                step = PlanStep(
                    step=step_num,
                    capability_id=cap["id"],
                    tool_name=action_spec.tool_name,
                    description=f"Execute {cap['name_es']}",
                    parameters=action_spec.parameters,
                    critical=cap["risk_level"] in ("HIGH", "CRITICAL"),
                    timeout_ms=action_spec.timeout_ms,
                    verification=action_spec.verification,
                    metadata=action_spec.metadata,
                )
                steps.append(step)
                step_num += 1
            except Exception as e:
                logger.debug("Could not create step for %s: %s", cap["id"], e)

        if not steps:
            return self._fallback_plan(goal)

        # Calculate total estimated time and risk
        total_ms = sum(s.timeout_ms for s in steps)
        risk_levels = [RiskLevel(cap["risk_level"]) for cap in candidate_caps if cap["id"] in [s.capability_id for s in steps]]
        max_risk = max(risk_levels, key=lambda r: r.value) if risk_levels else RiskLevel.NONE

        return Plan(
            goal=goal,
            steps=steps,
            estimated_total_ms=total_ms,
            risk_level=max_risk,
            requires_confirmation=any(s.critical for s in steps) or (initial_intent and initial_intent.requires_confirmation),
            metadata={"path": "agent_path", "structured": True},
        )

    def _find_relevant_capabilities(
        self,
        goal_lower: str,
        capabilities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Find capabilities relevant to the goal based on keywords."""
        # Keyword to capability mapping
        keyword_map = {
            "chrome": ["open_app", "browser_navigation"],
            "navegador": ["open_app", "browser_navigation"],
            "busca": ["web_search"],
            "search": ["web_search"],
            "google": ["web_search"],
            "youtube": ["youtube_play", "youtube_pause", "youtube_resume", "youtube_volume", "youtube_next", "youtube_previous"],
            "música": ["youtube_play", "youtube_volume"],
            "musica": ["youtube_play", "youtube_volume"],
            "volumen": ["system_volume_up", "system_volume_down", "youtube_volume"],
            "abre": ["open_app", "file_read", "browser_navigation"],
            "open": ["open_app", "file_read", "browser_navigation"],
            "archivo": ["file_read", "file_write", "file_list", "file_operations"],
            "file": ["file_read", "file_write", "file_list", "file_operations"],
            "captura": ["screen_capture"],
            "screenshot": ["screen_capture"],
            "hora": ["get_time"],
            "time": ["get_time"],
            "tiempo": ["weather_report"],
            "clima": ["weather_report"],
            "weather": ["weather_report"],
            "código": ["code_assistance"],
            "code": ["code_assistance"],
            "proyecto": ["project_development"],
            "project": ["project_development"],
            "aplicación": ["project_development"],
            "app": ["project_development"],
            "tarea": ["agent_task"],
            "task": ["agent_task"],
            "mensaje": ["send_message"],
            "message": ["send_message"],
            "recordatorio": ["reminder"],
            "reminder": ["reminder"],
            "vuelo": ["flight_search"],
            "flight": ["flight_search"],
            "juego": ["game_updates"],
            "game": ["game_updates"],
            "escritorio": ["desktop_management"],
            "desktop": ["desktop_management"],
            "configuración": ["computer_settings"],
            "settings": ["computer_settings"],
            "controla": ["computer_control"],
            "controlar": ["computer_control"],
            "apaga": ["system_shutdown"],
            "shutdown": ["system_shutdown"],
            "guarda": ["memory_save"],
            "save": ["memory_save"],
            "recuerda": ["memory_save"],
        }

        matched = set()
        for keyword, cap_ids in keyword_map.items():
            if keyword in goal_lower:
                matched.update(cap_ids)

        # Filter capabilities to matched ones
        relevant = [c for c in capabilities if c["id"] in matched]

        # If no matches, return top capabilities by domain relevance
        if not relevant:
            # Prioritize by common domains
            priority_domains = ["system", "web", "media", "coding", "automation"]
            relevant = []
            for domain in priority_domains:
                domain_caps = [c for c in capabilities if c["domain"] == domain]
                relevant.extend(domain_caps[:2])
                if len(relevant) >= 5:
                    break

        return relevant[:5]

    def _fallback_plan(self, goal: str) -> Plan:
        """Create a minimal fallback plan (single web_search step)."""
        try:
            intent = self._intent_router.classify(goal, autonomy_level=3)
            action_context = self._context_engine.get_action_context("web_search")
            action_spec = self._action_resolver.resolve(intent, action_context)

            step = PlanStep(
                step=1,
                capability_id="web_search",
                tool_name=action_spec.tool_name,
                description=f"Search for: {goal}",
                parameters=action_spec.parameters,
                critical=True,
                timeout_ms=action_spec.timeout_ms,
                verification=action_spec.verification,
            )
        except Exception:
            step = PlanStep(
                step=1,
                capability_id="web_search",
                tool_name="web_search",
                description=f"Search for: {goal}",
                parameters={"query": goal},
                critical=True,
                timeout_ms=30000,
                verification={"method": "result_non_empty", "cost": "low"},
            )

        return Plan(
            goal=goal,
            steps=[step],
            estimated_total_ms=30000,
            risk_level=RiskLevel.NONE,
            requires_confirmation=False,
            metadata={"path": "fallback"},
        )

    def _parse_plan(
        self,
        goal: str,
        plan_data: Dict[str, Any],
        autonomy_level: int,
    ) -> Plan:
        """Parse LLM plan output into Plan object."""
        steps = []
        total_ms = 0
        max_risk = RiskLevel.NONE
        requires_confirmation = False

        for i, step_data in enumerate(plan_data.get("steps", []), 1):
            capability_id = step_data.get("capability_id", "")
            capability = self._capability_registry.get(capability_id)

            if not capability:
                logger.warning("Unknown capability in plan: %s", capability_id)
                continue

            # Resolve action to get proper parameters
            try:
                intent = Intent(
                    name="PLAN_STEP",
                    type=IntentType.CODE_TASK,
                    path=RoutingPath.AGENT_PATH,
                    confidence=1.0,
                    capability_id=capability_id,
                )
                action_context = self._context_engine.get_action_context(capability_id)
                action_spec = self._action_resolver.resolve(intent, action_context)

                # Merge LLM parameters with resolved ones
                params = {**action_spec.parameters, **step_data.get("parameters", {})}
            except Exception:
                params = step_data.get("parameters", {})

            step = PlanStep(
                step=i,
                capability_id=capability_id,
                tool_name=step_data.get("tool_name", capability.tools[0] if capability.tools else ""),
                description=step_data.get("description", f"Execute {capability.name_es}"),
                parameters=params,
                critical=step_data.get("critical", True),
                timeout_ms=step_data.get("timeout_ms", 30000),
                verification=step_data.get("verification"),
                dependencies=step_data.get("dependencies", []),
            )
            steps.append(step)
            total_ms += step.timeout_ms

            if capability.risk_level.value > max_risk.value:
                max_risk = capability.risk_level
            if step.critical or capability.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
                requires_confirmation = True

        return Plan(
            goal=goal,
            steps=steps,
            estimated_total_ms=total_ms,
            risk_level=max_risk,
            requires_confirmation=requires_confirmation,
            metadata={"path": "agent_path", "llm_generated": True},
        )

    def replan(
        self,
        goal: str,
        completed_steps: List[PlanStep],
        failed_step: PlanStep,
        error: str,
        autonomy_level: int = 3,
    ) -> Plan:
        """Create a revised plan after a step failure."""
        logger.info("Replanning after step %d failure: %s", failed_step.step, error[:100])

        completed_summary = "\n".join(
            f"  - Step {s.step} ({s.capability_id}): DONE" for s in completed_steps
        )

        # Build context for replanning
        context = self._context_engine.get_planner_context()
        context["recent_messages"].append({
            "role": "system",
            "content": f"Previous attempt failed at step {failed_step.step} ({failed_step.capability_id}): {error}",
        })

        if completed_steps:
            context["recent_messages"].append({
                "role": "system",
                "content": f"Completed steps:\n{completed_summary}",
            })

        # Create new plan for remaining work
        new_goal = f"{goal} (retry after failure at step {failed_step.step})"
        plan = self._structured_plan(new_goal, context, autonomy_level)

        # Remove steps that were already completed
        completed_capabilities = {s.capability_id for s in completed_steps}
        filtered_steps = [s for s in plan.steps if s.capability_id not in completed_capabilities]

        # Renumber steps
        renumbered_steps = []
        for i, step in enumerate(filtered_steps, 1):
            new_step = PlanStep(
                step=i,
                capability_id=step.capability_id,
                tool_name=step.tool_name,
                description=step.description,
                parameters=step.parameters,
                critical=step.critical,
                timeout_ms=step.timeout_ms,
                verification=step.verification,
                dependencies=step.dependencies,
                metadata=step.metadata,
            )
            renumbered_steps.append(new_step)

        # Create new Plan with filtered and renumbered steps
        plan = Plan(
            goal=plan.goal,
            steps=renumbered_steps,
            estimated_total_ms=sum(s.timeout_ms for s in renumbered_steps),
            risk_level=plan.risk_level,
            requires_confirmation=plan.requires_confirmation,
            metadata={**plan.metadata, "replanned": True, "original_failure": f"Step {failed_step.step}: {error}"},
        )

        return plan


# Global instance
_planner: Optional[Planner] = None


def get_planner() -> Planner:
    """Get the global Planner instance."""
    global _planner
    if _planner is None:
        _planner = Planner()
    return _planner


def create_plan(
    goal: str,
    autonomy_level: int = 3,
    use_llm: bool = True,
) -> Plan:
    """Convenience function to create a plan."""
    planner = get_planner()
    return planner.create_plan(goal, autonomy_level, use_llm)


def replan(
    goal: str,
    completed_steps: List[PlanStep],
    failed_step: PlanStep,
    error: str,
    autonomy_level: int = 3,
) -> Plan:
    """Convenience function to replan."""
    planner = get_planner()
    return planner.replan(goal, completed_steps, failed_step, error, autonomy_level)