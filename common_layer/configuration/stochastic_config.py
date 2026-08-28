"""
stochastic_config.py — typed settings for two-stage stochastic optimization.

Loads config/stochastic.yaml. Strictly opt-in: `enabled: false` (the
shipped default) means every gate runs the existing deterministic
build_core_model path exactly as before — zero behavior change unless a
user explicitly flips this on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class StochasticConfig:
    enabled: bool
    n_scenarios: int
    gates: List[str]
    risk_measure: str          # "expected_value" | "cvar"
    cvar_alpha: float

    def enabled_for(self, gate: str) -> bool:
        """True only if stochastic mode is globally on AND this gate opted in."""
        return self.enabled and gate in self.gates

    @staticmethod
    def from_dict(d: dict) -> "StochasticConfig":
        s = d.get("stochastic", {})
        return StochasticConfig(
            enabled=bool(s.get("enabled", False)),
            n_scenarios=int(s.get("n_scenarios", 5)),
            gates=list(s.get("gates", ["DA"])),
            risk_measure=str(s.get("risk_measure", "expected_value")),
            cvar_alpha=float(s.get("cvar_alpha", 0.95)),
        )
