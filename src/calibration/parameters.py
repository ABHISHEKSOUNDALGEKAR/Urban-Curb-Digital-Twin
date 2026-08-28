"""The free parameters of the behavioural model, and how they map into config.

Calibration searches over a small, interpretable vector rather than "everything
that is a number". Each entry names a parameter, its bounds, and the config path
it writes to, so the search space is auditable and a reviewer can see exactly
what was allowed to move.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Parameter:
    name: str
    path: tuple[str, ...]  # path into the agents config, e.g. ("passenger", "alpha_walk_per_m")
    low: float
    high: float
    default: float
    description: str

    def clip(self, value: float) -> float:
        return float(np.clip(value, self.low, self.high))


#: The calibrated parameter set. Deliberately small (7 parameters): the
#: identifiability of a simulation model degrades fast as the parameter vector
#: grows relative to the information content of the observations.
PARAMETERS: tuple[Parameter, ...] = (
    Parameter(
        "passenger_alpha_walk",
        ("passenger", "alpha_walk_per_m"),
        0.002,
        0.020,
        0.0075,
        "Passenger dollar cost per metre of walking; sets how far drivers will park from the door.",
    ),
    Parameter(
        "passenger_gamma_search",
        ("passenger", "gamma_search_per_min"),
        0.05,
        1.20,
        0.30,
        "Passenger cost per expected minute of search; trades search against walking.",
    ),
    Parameter(
        "passenger_delta_occupancy",
        ("passenger", "delta_occupancy"),
        0.10,
        3.00,
        1.10,
        "How strongly drivers avoid blocks that look full; drives spatial spreading of demand.",
    ),
    Parameter(
        "passenger_search_radius",
        ("passenger", "search_radius_m"),
        120.0,
        420.0,
        260.0,
        "Radius of the considered choice set around the destination.",
    ),
    Parameter(
        "passenger_compliance",
        ("passenger", "compliance_probability"),
        0.50,
        0.99,
        0.86,
        "Probability a driver keeps searching rather than parking illegally.",
    ),
    Parameter(
        "delivery_illegal_prob",
        ("delivery", "illegal_parking_probability"),
        0.10,
        0.90,
        0.55,
        "Propensity of a courier to double-park once a loading zone search fails.",
    ),
    Parameter(
        "ridehail_gamma_search",
        ("ridehail", "gamma_search_per_min"),
        0.20,
        3.00,
        1.30,
        "TNC cost per minute of curb search; governs circling vs. double-parking.",
    ),
)

PARAM_NAMES: tuple[str, ...] = tuple(p.name for p in PARAMETERS)


def bounds() -> list[tuple[float, float]]:
    return [(p.low, p.high) for p in PARAMETERS]


def defaults() -> np.ndarray:
    return np.array([p.default for p in PARAMETERS], dtype=float)


def to_overrides(theta: Sequence[float]) -> dict[str, Any]:
    """Turn a parameter vector into a config override dict."""
    if len(theta) != len(PARAMETERS):
        raise ValueError(f"expected {len(PARAMETERS)} parameters, got {len(theta)}")
    agents: dict[str, Any] = {}
    for p, value in zip(PARAMETERS, theta, strict=True):
        node = agents
        for key in p.path[:-1]:
            node = node.setdefault(key, {})
        node[p.path[-1]] = p.clip(float(value))
    return {"agents": agents}


def to_dict(theta: Sequence[float]) -> dict[str, float]:
    return {p.name: float(v) for p, v in zip(PARAMETERS, theta, strict=True)}


def from_dict(d: dict[str, float]) -> np.ndarray:
    return np.array([float(d[p.name]) for p in PARAMETERS], dtype=float)


def normalise(theta: Sequence[float]) -> np.ndarray:
    """Map a parameter vector onto the unit box (for distance / diagnostics)."""
    return np.array(
        [(float(v) - p.low) / (p.high - p.low) for p, v in zip(PARAMETERS, theta, strict=True)]
    )
