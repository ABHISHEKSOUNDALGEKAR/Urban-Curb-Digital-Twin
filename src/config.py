"""Configuration loading, validation and overriding.

Every run of the model is fully described by a :class:`RunConfig`: the network,
the behavioural parameters, the scenario diff and the seed. Serialising that
object is what makes an experiment reproducible, so it is deliberately a plain
nested-dict container with a stable hash rather than a web of live objects.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"

VEHICLE_CLASSES: tuple[str, ...] = ("passenger", "delivery", "ridehail")


class ConfigError(ValueError):
    """Raised when a configuration file is structurally invalid."""


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ConfigError(f"expected a mapping at the top level of {path}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``.

    Mappings are merged key-by-key; every other type is replaced wholesale.
    ``None`` in the override is treated as "no opinion" and leaves the base
    value untouched, which is what lets a scenario declare ``allocation: null``.
    """
    out = copy.deepcopy(base)
    for key, value in override.items():
        if value is None:
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def validate_allocation(alloc: dict[str, float], *, tol: float = 1e-6) -> dict[str, float]:
    """Check that a curb allocation is a valid distribution over vehicle classes."""
    missing = set(VEHICLE_CLASSES) - set(alloc)
    if missing:
        raise ConfigError(f"allocation is missing vehicle classes: {sorted(missing)}")
    unknown = set(alloc) - set(VEHICLE_CLASSES)
    if unknown:
        raise ConfigError(f"allocation has unknown vehicle classes: {sorted(unknown)}")
    if any(v < 0 for v in alloc.values()):
        raise ConfigError(f"allocation shares must be non-negative: {alloc}")
    total = sum(alloc.values())
    if abs(total - 1.0) > tol:
        raise ConfigError(f"allocation shares must sum to 1.0, got {total:.6f}: {alloc}")
    return {k: float(alloc[k]) for k in VEHICLE_CLASSES}


@dataclass(frozen=True)
class RunConfig:
    """Everything needed to reproduce a single simulation replication."""

    scenario: str
    seed: int
    network: dict[str, Any]
    agents: dict[str, Any]
    scenario_spec: dict[str, Any]
    overrides: dict[str, Any] = field(default_factory=dict)

    # -- convenience accessors -------------------------------------------------
    @property
    def horizon_min(self) -> float:
        return float(self.scenario_spec["horizon_min"])

    @property
    def warmup_min(self) -> float:
        return float(self.scenario_spec["warmup_min"])

    @property
    def demand(self) -> dict[str, float]:
        base = dict(self.scenario_spec["demand"])
        scale = self.scenario_spec.get("demand_scale") or {}
        return {k: float(base[k]) * float(scale.get(k, 1.0)) for k in VEHICLE_CLASSES}

    @property
    def allocation(self) -> dict[str, float]:
        alloc = self.scenario_spec.get("allocation")
        if alloc is None:
            alloc = self.network["curb"]["baseline_allocation"]
        return validate_allocation(alloc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "seed": self.seed,
            "network": self.network,
            "agents": self.agents,
            "scenario_spec": self.scenario_spec,
            "overrides": self.overrides,
        }

    def fingerprint(self) -> str:
        """Stable short hash of the configuration, excluding the seed.

        Two runs with the same fingerprint differ only in their random draw,
        which is exactly the grouping the experiment aggregator needs.
        """
        payload = self.to_dict()
        payload.pop("seed")
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:12]

    def with_seed(self, seed: int) -> RunConfig:
        return RunConfig(
            scenario=self.scenario,
            seed=int(seed),
            network=self.network,
            agents=self.agents,
            scenario_spec=self.scenario_spec,
            overrides=self.overrides,
        )

    def with_overrides(self, overrides: dict[str, Any]) -> RunConfig:
        """Return a copy with ``overrides`` applied.

        Supported keys:
          ``agents``        merged into the behavioural parameters (calibration)
          ``network``       merged into the network parameters
          ``allocation``    replaces the curb allocation (optimization)
          ``demand_scale``  multiplies arrival rates per class (elasticities)
          plus any other scenario-level key (``horizon_min``, ``pricing_policy`` ...)
        """
        overrides = copy.deepcopy(overrides)
        agents = deep_merge(self.agents, overrides.pop("agents", {}) or {})
        network = deep_merge(self.network, overrides.pop("network", {}) or {})
        scenario_spec = deep_merge(self.scenario_spec, overrides)
        return RunConfig(
            scenario=self.scenario,
            seed=self.seed,
            network=network,
            agents=agents,
            scenario_spec=scenario_spec,
            overrides=deep_merge(self.overrides, overrides),
        )


def load_scenario(
    scenario: str = "baseline",
    seed: int = 0,
    config_dir: Path | str | None = None,
    overrides: dict[str, Any] | None = None,
) -> RunConfig:
    """Build a :class:`RunConfig` for ``scenario`` from the YAML config tree."""
    cdir = Path(config_dir) if config_dir is not None else CONFIG_DIR
    network = _read_yaml(cdir / "network.yaml")
    agents = _read_yaml(cdir / "agents.yaml")
    scenarios = _read_yaml(cdir / "scenarios.yaml")

    defaults = scenarios.get("defaults", {})
    table = scenarios.get("scenarios", {})
    if scenario not in table:
        raise ConfigError(f"unknown scenario {scenario!r}; available: {sorted(table)}")
    spec = deep_merge(defaults, table[scenario])
    spec.setdefault("label", scenario)

    cfg = RunConfig(
        scenario=scenario,
        seed=int(seed),
        network=network,
        agents=agents,
        scenario_spec=spec,
    )
    if overrides:
        cfg = cfg.with_overrides(overrides)
    # Fail fast on an invalid allocation rather than deep inside the engine.
    _ = cfg.allocation
    return cfg


def list_scenarios(config_dir: Path | str | None = None) -> dict[str, str]:
    """Map scenario name -> human-readable label."""
    cdir = Path(config_dir) if config_dir is not None else CONFIG_DIR
    scenarios = _read_yaml(cdir / "scenarios.yaml").get("scenarios", {})
    return {name: spec.get("label", name) for name, spec in scenarios.items()}


def load_optimization_config(config_dir: Path | str | None = None) -> dict[str, Any]:
    cdir = Path(config_dir) if config_dir is not None else CONFIG_DIR
    return _read_yaml(cdir / "optimization.yaml")


def load_demand_levels(config_dir: Path | str | None = None) -> dict[str, dict[str, float]]:
    cdir = Path(config_dir) if config_dir is not None else CONFIG_DIR
    return _read_yaml(cdir / "scenarios.yaml").get("demand_levels", {})
