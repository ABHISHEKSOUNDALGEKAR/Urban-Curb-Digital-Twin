"""Reference allocations to compare an optimized allocation against.

An optimizer that beats nothing has proved nothing. These are the allocations a
city might plausibly adopt without running a simulation study at all, and they
are what the search results are reported against:

``current``       today's posted allocation, from ``config/network.yaml``
``equal``         an even three-way split - the naive fairness answer
``demand_share``  stalls proportional to each class's *curb-time* demand
                  (arrival rate x mean dwell), the standard back-of-envelope rule
``passenger_max`` the corner solution that maximises general parking, i.e. what
                  happens when curb policy optimises for drivers alone
"""

from __future__ import annotations

import numpy as np

from src.config import VEHICLE_CLASSES, load_scenario
from src.optimization.objective import CurbObjective, Evaluation, project_to_simplex


def current_allocation(scenario: str = "baseline") -> dict[str, float]:
    return load_scenario(scenario, seed=0).allocation


def equal_allocation() -> dict[str, float]:
    return {c: 1.0 / len(VEHICLE_CLASSES) for c in VEHICLE_CLASSES}


def demand_share_allocation(scenario: str = "baseline") -> dict[str, float]:
    """Allocate stalls in proportion to curb-time demand (arrivals x dwell).

    This is the intuitive rule a planner reaches for first. It ignores that the
    three classes have different values of time, different tolerance for
    walking, and different fallback behaviour when they fail - which is exactly
    why the simulation-based answer differs from it.
    """
    cfg = load_scenario(scenario, seed=0)
    demand = cfg.demand
    a = cfg.agents
    dwell_min = {
        "passenger": float(a["passenger"]["mean_dwell_min"]),
        "delivery": float(a["delivery"]["mean_service_min"]),
        "ridehail": float(a["ridehail"]["mean_dwell_s"]) / 60.0
        + float(a["ridehail"]["dropoff_dwell_s"]) / 60.0,
    }
    curb_time = {c: demand[c] * dwell_min[c] for c in VEHICLE_CLASSES}
    total = sum(curb_time.values())
    return {c: curb_time[c] / total for c in VEHICLE_CLASSES}


def passenger_max_allocation(objective: CurbObjective) -> dict[str, float]:
    b = objective.bounds
    x = np.array([b["passenger"][1], b["delivery"][0], b["ridehail"][0]], dtype=float)
    v = project_to_simplex(x, b)
    return {c: float(v[i]) for i, c in enumerate(VEHICLE_CLASSES)}


def all_baselines(
    objective: CurbObjective, scenario: str = "baseline"
) -> dict[str, dict[str, float]]:
    return {
        "current": current_allocation(scenario),
        "equal": equal_allocation(),
        "demand_share": demand_share_allocation(scenario),
        "passenger_max": passenger_max_allocation(objective),
    }


def evaluate_baselines(
    objective: CurbObjective, scenario: str = "baseline", verbose: bool = True
) -> dict[str, Evaluation]:
    out: dict[str, Evaluation] = {}
    for name, alloc in all_baselines(objective, scenario).items():
        x = [alloc[c] for c in VEHICLE_CLASSES]
        ev = objective.evaluate(x)
        out[name] = ev
        if verbose:
            shares = " ".join(f"{c[:3]}={ev.allocation[c]:.3f}" for c in VEHICLE_CLASSES)
            print(f"  {name:14s} {shares}  J = {ev.value:10.1f} +/- {ev.stderr:.1f}", flush=True)
    return out
