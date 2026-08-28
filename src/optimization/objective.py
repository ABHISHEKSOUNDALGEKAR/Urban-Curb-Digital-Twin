"""The curb allocation objective: what "better" means, in dollars.

Decision variable
-----------------
``x = (x_passenger, x_delivery, x_ridehail)``, the district-wide share of curb
stalls assigned to each regulation type, on the simplex ``sum(x) = 1``.

Objective
---------
Weighted district-wide social cost per hour:

    J(x) = w1 * passenger search cost
         + w2 * delivery delay cost
         + w3 * ridehail wait cost
         + w4 * illegal parking externality
         + w5 * external cost of cruising VMT

Time is monetised at class-specific values of time; the illegal-parking and VMT
terms are externality prices from ``config/optimization.yaml``. Meter revenue is
a *transfer*, not a resource cost, so it is excluded from ``J`` and reported
separately - a curb policy that raises revenue by making everyone worse off
should not score well.

The objective is **stochastic**: ``J(x)`` can only be estimated by simulation, so
every evaluation is a mean over several seeds with a standard error attached.
Everything downstream is built to respect that.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.config import VEHICLE_CLASSES, load_optimization_config, load_scenario
from src.experiments.runner import run_replications

#: Components of the objective, in the order they are reported.
COMPONENTS = (
    "passenger_search_cost",
    "delivery_delay_cost",
    "ridehail_wait_cost",
    "illegal_parking_cost",
    "vmt_cost",
)


def project_to_simplex(
    x: Sequence[float], bounds: dict[str, tuple[float, float]] | None = None
) -> np.ndarray:
    """Map an arbitrary non-negative vector onto the allocation simplex.

    The optimizer searches an unconstrained box and this function makes every
    point it proposes feasible, so the equality constraint ``sum(x) = 1`` can
    never be violated by a candidate. Box bounds are then applied and the
    residual redistributed, iterating a few times; this converges quickly for
    the small, feasible boxes used here.
    """
    v = np.clip(np.asarray(x, dtype=float), 1e-9, None)
    v = v / v.sum()
    if not bounds:
        return v
    lo = np.array([bounds[c][0] for c in VEHICLE_CLASSES])
    hi = np.array([bounds[c][1] for c in VEHICLE_CLASSES])
    if lo.sum() > 1.0 + 1e-9 or hi.sum() < 1.0 - 1e-9:
        raise ValueError(
            f"infeasible allocation bounds: lower sum {lo.sum()}, upper sum {hi.sum()}"
        )
    for _ in range(64):
        v = np.clip(v, lo, hi)
        gap = 1.0 - v.sum()
        if abs(gap) < 1e-9:
            break
        # Redistribute the residual only among components that can absorb it.
        room = (hi - v) if gap > 0 else (v - lo)
        total_room = room.sum()
        if total_room <= 1e-12:
            break
        v = v + gap * room / total_room
    return np.clip(v, lo, hi)


def allocation_dict(x: Sequence[float]) -> dict[str, float]:
    v = np.asarray(x, dtype=float)
    v = v / v.sum()
    return {c: float(v[i]) for i, c in enumerate(VEHICLE_CLASSES)}


@dataclass
class Evaluation:
    """One estimate of the objective at one allocation."""

    allocation: dict[str, float]
    value: float
    stderr: float
    components: dict[str, float]
    metrics: dict[str, float]
    n_seeds: int
    seeds: list[int] = field(default_factory=list)

    def as_row(self) -> dict[str, Any]:
        row = {f"x_{k}": v for k, v in self.allocation.items()}
        row.update(
            {
                "objective": self.value,
                "stderr": self.stderr,
                "n_seeds": self.n_seeds,
                **{f"c_{k}": v for k, v in self.components.items()},
                **{
                    k: self.metrics.get(k, float("nan"))
                    for k in (
                        "passenger_search_time_min",
                        "delivery_delay_min",
                        "ridehail_wait_min",
                        "illegal_parking_rate",
                        "vmt_miles",
                        "revenue_usd",
                        "curb_occupancy_passenger",
                        "passenger_abandonment_rate",
                        "system_social_cost_usd",
                    )
                },
            }
        )
        return row


class CurbObjective:
    """Simulation-based objective over curb allocations.

    Two properties matter for how it is used:

    *Noisy.* Each evaluation averages ``seeds_per_evaluation`` replications and
    exposes the standard error, so an "improvement" smaller than the noise can
    be recognised as such rather than reported.

    *Expensive.* One evaluation is several full simulations, which is why the
    evaluation count - not the wall-clock time - is the currency in which the
    search algorithms are compared. Results are memoised on the rounded
    allocation so that repeated visits to the same point are free.
    """

    def __init__(
        self,
        scenario: str = "baseline",
        opt_cfg: dict | None = None,
        seeds: Sequence[int] | None = None,
        workers: int = 1,
        round_to: int = 4,
    ) -> None:
        self.scenario = scenario
        self.opt_cfg = opt_cfg or load_optimization_config()
        self.weights = self.opt_cfg["objective"]["weights"]
        self.social_params = self.opt_cfg["objective"]
        search_cfg = self.opt_cfg.get("search", {})
        n_seeds = int(search_cfg.get("seeds_per_evaluation", 6))
        self.common_random_numbers = bool(search_cfg.get("common_random_numbers", True))
        self.seeds = list(seeds) if seeds is not None else list(range(1001, 1001 + n_seeds))
        self.bounds = {c: tuple(self.opt_cfg["decision"]["bounds"][c]) for c in VEHICLE_CLASSES}
        self.workers = workers
        self.round_to = round_to
        self.cache: dict[tuple, Evaluation] = {}
        self.history: list[Evaluation] = []
        # Values of time are fixed for the whole study; resolve them once.
        self._vot = load_scenario(scenario, seed=0).agents["common"]["value_of_time_per_hour"]

    # -- evaluation -------------------------------------------------------------
    def evaluate(self, x: Sequence[float], seeds: Sequence[int] | None = None) -> Evaluation:
        alloc = allocation_dict(project_to_simplex(x, self.bounds))
        key = tuple(round(alloc[c], self.round_to) for c in VEHICLE_CLASSES)
        if seeds is None and key in self.cache:
            return self.cache[key]

        # Common random numbers: the same seed set at every candidate point, so
        # differences between allocations reflect the allocation and not the
        # draw. This is the single cheapest variance reduction available.
        use_seeds = list(seeds) if seeds is not None else self.seeds
        cfg = load_scenario(self.scenario, seed=use_seeds[0], overrides={"allocation": alloc})
        rows = run_replications(
            cfg, use_seeds, workers=self.workers, social_cost_params=self.social_params
        )
        return self._reduce(alloc, rows, use_seeds, cache_key=key if seeds is None else None)

    def _reduce(
        self,
        alloc: dict[str, float],
        rows: list[dict[str, Any]],
        seeds: Sequence[int],
        cache_key: tuple | None = None,
    ) -> Evaluation:
        per_seed_values = [self._objective_value(r) for r in rows]
        value = float(np.mean(per_seed_values))
        stderr = (
            float(np.std(per_seed_values, ddof=1) / math.sqrt(len(per_seed_values)))
            if len(per_seed_values) > 1
            else 0.0
        )
        components = {k: float(np.mean([self._components(r)[k] for r in rows])) for k in COMPONENTS}
        numeric_keys = [k for k, v in rows[0].items() if isinstance(v, (int, float))]
        metrics = {k: float(np.mean([r.get(k, np.nan) for r in rows])) for k in numeric_keys}
        ev = Evaluation(
            allocation=alloc,
            value=value,
            stderr=stderr,
            components=components,
            metrics=metrics,
            n_seeds=len(rows),
            seeds=list(seeds),
        )
        self.history.append(ev)
        if cache_key is not None:
            self.cache[cache_key] = ev
        return ev

    def _components(self, row: dict[str, Any]) -> dict[str, float]:
        """Monetised cost components for one replication, in dollars."""
        vot = self._vot
        n_pas = row.get("passenger_trips", 0) or 0
        n_dlv = row.get("delivery_trips", 0) or 0
        n_rh = row.get("ridehail_trips", 0) or 0
        return {
            "passenger_search_cost": row.get("passenger_search_time_min", 0.0)
            * n_pas
            * vot["passenger"]
            / 60.0,
            "delivery_delay_cost": row.get("delivery_delay_min", 0.0)
            * n_dlv
            * vot["delivery"]
            / 60.0,
            "ridehail_wait_cost": row.get("ridehail_wait_min", 0.0) * n_rh * vot["ridehail"] / 60.0,
            "illegal_parking_cost": row.get("illegal_parking_events", 0.0)
            * float(self.social_params.get("illegal_event_social_cost", 14.0)),
            "vmt_cost": row.get("vmt_miles", 0.0)
            * float(self.social_params.get("vmt_external_cost_per_mile", 0.62)),
        }

    def _objective_value(self, row: dict[str, Any]) -> float:
        comp = self._components(row)
        return float(sum(float(self.weights.get(k, 1.0)) * comp[k] for k in COMPONENTS))

    # -- convenience ------------------------------------------------------------
    def __call__(self, x: Sequence[float]) -> float:
        return self.evaluate(x).value

    @property
    def n_evaluations(self) -> int:
        return len(self.history)

    @property
    def n_simulations(self) -> int:
        return sum(e.n_seeds for e in self.history)

    def best(self) -> Evaluation | None:
        return min(self.history, key=lambda e: e.value) if self.history else None
