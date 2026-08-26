# core/capability_registry.py
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

# ============================================================
# Capability Registry — fase 2B B1
# ============================================================
# Responsable: cargar y consultar capabilities del P.I.P.E Core.
# NO reemplaza al Tool Registry (core/tool_registry.py) — ese
# es para Gemini Live function_calling. Este es para el Core:
# routing, permissions, verification, planning.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
CAPABILITIES_PATH = BASE_DIR / "config" / "capabilities.json"


# ============================================================
# Enums
# ============================================================

class RiskLevel(Enum):
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class VerificationCost(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class LocalOrRemote(Enum):
    LOCAL = "local"
    REMOTE = "remote"


class CapabilityStatus(Enum):
    AVAILABLE = "AVAILABLE"
    PARTIAL = "PARTIAL"
    DEPENDENCY_MISSING = "DEPENDENCY_MISSING"
    DISABLED = "DISABLED"


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class Capability:
    """Representa una capability del sistema P.I.P.E."""
    id: str
    name_es: str
    domain: str
    tools: list
    risk_level: RiskLevel
    requires_confirmation: bool
    rollback_possible: bool
    verification_method: str
    verification_cost: VerificationCost
    verification_latency: str
    dependencies: list
    latency_hint: str
    local_or_remote: LocalOrRemote
    status: CapabilityStatus = CapabilityStatus.AVAILABLE
    intent_name: str = ""  # Intent name for routing

    def is_actionable(self, autonomy_level: int) -> bool:
        """Determina si esta capability puede ejecutarse en el nivel de autonomía dado."""
        if self.status != CapabilityStatus.AVAILABLE:
            return False
        # Regla de seguridad: CRITICAL siempre requiere confirmación explícita
        if self.risk_level == RiskLevel.CRITICAL:
            return False
        if autonomy_level >= 3:
            return True
        if autonomy_level >= 2 and not self.requires_confirmation:
            return True
        return False

    def need_explicit_permission(self, autonomy_level: int) -> bool:
        """Indica si esta capability necesita confirmación explícita del usuario."""
        if self.risk_level == RiskLevel.CRITICAL:
            return True
        if self.requires_confirmation and autonomy_level < 4:
            return True
        return False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name_es": self.name_es,
            "domain": self.domain,
            "tools": self.tools,
            "risk_level": self.risk_level.value,
            "requires_confirmation": self.requires_confirmation,
            "rollback_possible": self.rollback_possible,
            "verification_method": self.verification_method,
            "verification_cost": self.verification_cost.value,
            "verification_latency": self.verification_latency,
            "dependencies": self.dependencies,
            "latency_hint": self.latency_hint,
            "local_or_remote": self.local_or_remote.value,
            "status": self.status.value,
        }


# ============================================================
# Cargador singleton
# ============================================================

class CapabilityRegistry:
    """Singleton que carga y consulta capabilities."""

    _instance: Optional["CapabilityRegistry"] = None
    _capabilities: Dict[str, Capability] = {}

    def __new__(cls) -> "CapabilityRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self) -> None:
        """Carga capabilities desde config/capabilities.json."""
        if not CAPABILITIES_PATH.exists():
            raise FileNotFoundError(f"capabilities.json no encontrado en {CAPABILITIES_PATH}")

        with open(CAPABILITIES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._intent_to_capability = {}

        for cap_data in data.get("capabilities", []):
            cap = Capability(
                id=cap_data["id"],
                name_es=cap_data["name_es"],
                domain=cap_data["domain"],
                tools=cap_data.get("tools", []),
                risk_level=RiskLevel(cap_data.get("risk_level", "NONE")),
                requires_confirmation=cap_data.get("requires_confirmation", False),
                rollback_possible=cap_data.get("rollback_possible", False),
                verification_method=cap_data.get("verification_method", "none"),
                verification_cost=VerificationCost(cap_data.get("verification_cost", "low")),
                verification_latency=cap_data.get("verification_latency", "unknown"),
                dependencies=cap_data.get("dependencies", []),
                latency_hint=cap_data.get("latency_hint", "unknown"),
                local_or_remote=LocalOrRemote(cap_data.get("local_or_remote", "local")),
                status=CapabilityStatus(cap_data.get("status", "AVAILABLE")),
                intent_name=cap_data.get("intent_name", ""),
            )
            self._capabilities[cap.id] = cap

            # Map intent names to capability IDs (from config or derived from id)
            intent_name = cap_data.get("intent_name", cap.id.upper())
            self._intent_to_capability[intent_name.lower()] = cap.id

            # Also map common variations
            self._intent_to_capability[cap.id.lower()] = cap.id

    def get(self, capability_id: str) -> Optional[Capability]:
        """Obtiene una capability por ID."""
        return self._capabilities.get(capability_id)

    def get_all(self) -> List[Capability]:
        """Obtiene todas las capabilities."""
        return list(self._capabilities.values())

    def query(
        self,
        domain: Optional[str] = None,
        risk_level: Optional[RiskLevel] = None,
        actionable: bool = False,
        autonomy_level: int = 2,
    ) -> List[Capability]:
        """Consulta capabilities con filtros opcionales."""
        results = list(self._capabilities.values())

        if domain:
            results = [c for c in results if c.domain == domain]

        if risk_level:
            results = [c for c in results if c.risk_level == risk_level]

        if actionable:
            results = [c for c in results if c.is_actionable(autonomy_level)]

        return results

    def get_by_tool(self, tool_name: str) -> List[Capability]:
        """Obtiene todas las capabilities que usan una herramienta específica."""
        return [c for c in self._capabilities.values() if tool_name in c.tools]

    def has_tool(self, tool_name: str) -> bool:
        """Indica si alguna capability usa esta herramienta."""
        return any(tool_name in c.tools for c in self._capabilities.values())

    def get_status(self) -> dict:
        """Obtiene el estado del registry."""
        total = len(self._capabilities)
        by_status = {}
        by_domain = {}
        by_risk = {}

        for cap in self._capabilities.values():
            by_status[cap.status.value] = by_status.get(cap.status.value, 0) + 1
            by_domain[cap.domain] = by_domain.get(cap.domain, 0) + 1
            by_risk[cap.risk_level.value] = by_risk.get(cap.risk_level.value, 0) + 1

        return {
            "total": total,
            "by_status": by_status,
            "by_domain": by_domain,
            "by_risk": by_risk,
        }

    def reset(self) -> None:
        """Resetea el registry (para recarga en caliente si es necesario)."""
        self._capabilities = {}
        self._load()


# ============================================================
# Funciones de conveniencia
# ============================================================

def get_registry() -> CapabilityRegistry:
    """Obtiene la instancia del Capability Registry (singleton)."""
    return CapabilityRegistry()


def get_capability(capability_id: str) -> Optional[Capability]:
    return get_registry().get(capability_id)


def list_capabilities() -> List[Capability]:
    return get_registry().get_all()


def query_capabilities(
    domain: Optional[str] = None,
    risk_level: Optional[RiskLevel] = None,
    actionable: bool = False,
    autonomy_level: int = 2,
) -> List[Capability]:
    return get_registry().query(domain, risk_level, actionable, autonomy_level)
