from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class Action:
    action_id: str
    action: str
    args: dict[str, Any]
    label: str
    source: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AuditRef:
    path: str
    sha256: str
    format: str
    step_id: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ObservationEnvelope:
    schema_version: str
    dataset_version: str
    run_id: str
    step_id: int
    game_fingerprint: dict[str, Any]
    phase: str
    context: dict[str, Any]
    agent_observation: dict[str, Any]
    legal_actions: list[dict[str, Any]]
    audit_ref: Optional[dict[str, Any]]
    state_hash: str
    terminal: bool
    terminal_reason: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

