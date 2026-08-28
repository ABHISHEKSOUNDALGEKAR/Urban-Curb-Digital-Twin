"""Turning raw simulation output into the metrics a curb study reports.

Two levels:

``summarise(result)``   one replication -> a flat dict of scalar metrics
``aggregate(summaries)`` many replications -> mean, sd and a 95% CI per metric

Keeping these separate from the engine means a stored raw run can be
re-summarised with new metrics without re-simulating, and means the metric
definitions are unit-testable in isolation.

Conventions
-----------
* Only trips that *arrived after* the warm-up are counted.
* Times are minutes, distances metres unless the name says otherwise, money USD.
* Rates named ``*_rate`` are fractions of the relevant trip population.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.config import VEHICLE_CLASSES
from src.simulation.engine import SimulationResult
from src.simulation.routing import METRES_PER_MILE

# Metrics for which "lower is better"; used by report formatting.
LOWER_IS_BETTER = {
    "passenger_search_time_min",
    "passenger_walk_distance_m",
    "passenger_parking_cost_usd",
    "passenger_failed_attempts",
    "passenger_abandonment_rate",
    "passenger_illegal_rate",
    "delivery_delay_min",
    "delivery_illegal_rate",
    "delivery_search_time_min",
    "ridehail_wait_min",
    "ridehail_pickup_distance_m",
    "ridehail_circling_time_min",
    "ridehail_illegal_rate",
    "illegal_parking_rate",
    "vmt_miles",
    "congestion_index",
    "system_social_cost_usd",
}

HEADLINE_METRICS = [
    "passenger_search_time_min",
    "delivery_delay_min",
    "ridehail_wait_min",
    "illegal_parking_rate",
    "vmt_miles",
    "system_social_cost_usd",
]


def _post_warmup(trips: Sequence[dict]) -> pd.DataFrame:
    df = pd.DataFrame(trips)
    if df.empty:
        return df
    return df.loc[~df["warmup"].astype(bool)].copy()


def _mean(series, default: float = 0.0) -> float:
    if series is None or len(series) == 0:
        return default
    v = float(np.mean(series))
    return v if math.isfinite(v) else default


def _quantile(series, q: float, default: float = 0.0) -> float:
    if series is None or len(series) == 0:
        return default
    return float(np.quantile(series, q))


def summarise(result: SimulationResult, social_cost_params: dict | None = None) -> dict[str, Any]:
    """Reduce one replication to a flat dict of scalar metrics."""
    p = social_cost_params or {}
    vmt_external = float(p.get("vmt_external_cost_per_mile", 0.62))
    illegal_social = float(p.get("illegal_event_social_cost", 14.0))
    vot = result.cfg.agents["common"]["value_of_time_per_hour"]

    df = _post_warmup(result.trips)
    out: dict[str, Any] = {
        "scenario": result.scenario,
        "seed": result.seed,
        "fingerprint": result.cfg.fingerprint(),
        "wall_time_s": result.wall_time_s,
        "total_stalls": result.total_stalls,
        "fleet_size": result.fleet_size,
        "n_trips": int(len(df)),
    }
    for cls in VEHICLE_CLASSES:
        out[f"capacity_{cls}"] = int(result.capacity_by_class.get(cls, 0))
        out[f"capacity_share_{cls}"] = (
            result.capacity_by_class.get(cls, 0) / result.total_stalls
            if result.total_stalls
            else 0.0
        )

    if df.empty:
        return out

    by_class = {c: df.loc[df["vehicle_class"] == c] for c in VEHICLE_CLASSES}

    # -- passenger --------------------------------------------------------------
    pas = by_class["passenger"]
    out["passenger_trips"] = int(len(pas))
    out["passenger_search_time_min"] = _mean(pas["search_time_min"])
    out["passenger_search_time_p90_min"] = _quantile(pas["search_time_min"], 0.90)
    out["passenger_search_distance_m"] = _mean(pas["search_distance_m"])
    out["passenger_walk_distance_m"] = _mean(pas["walk_distance_m"])
    out["passenger_parking_cost_usd"] = _mean(pas["parking_cost_usd"])
    out["passenger_failed_attempts"] = _mean(pas["failed_attempts"])
    out["passenger_illegal_rate"] = _rate(pas, "illegal")
    out["passenger_diverted_rate"] = _rate(pas, "diverted")
    out["passenger_abandonment_rate"] = _rate(pas, "abandoned")
    out["passenger_parked_rate"] = _rate(pas, "parked")

    # -- delivery ---------------------------------------------------------------
    dlv = by_class["delivery"]
    out["delivery_trips"] = int(len(dlv))
    out["delivery_search_time_min"] = _mean(dlv["search_time_min"])
    out["delivery_delay_min"] = _mean(dlv["service_delay_min"])
    out["delivery_delay_p90_min"] = _quantile(dlv["service_delay_min"], 0.90)
    out["delivery_illegal_rate"] = _rate(dlv, "illegal")
    out["delivery_walk_distance_m"] = _mean(dlv["walk_distance_m"])

    # -- ridehail ---------------------------------------------------------------
    rh = by_class["ridehail"]
    out["ridehail_trips"] = int(len(rh))
    out["ridehail_wait_min"] = _mean(rh["wait_time_min"])
    out["ridehail_wait_p90_min"] = _quantile(rh["wait_time_min"], 0.90)
    out["ridehail_pickup_distance_m"] = _mean(rh["pickup_distance_m"])
    out["ridehail_circling_time_min"] = _mean(rh["circling_time_min"])
    out["ridehail_curb_dwell_min"] = _mean(rh["dwell_min"])
    out["ridehail_illegal_rate"] = _rate(rh, "illegal")
    out["ridehail_dropoff_illegal_events"] = int(result.events.get("illegal_ridehail_dropoff", 0))
    out["ridehail_circling_loops"] = int(result.events.get("ridehail_circling_loop", 0))

    # -- curb -------------------------------------------------------------------
    occ = result.occupancy_samples
    if occ:
        out["curb_occupancy"] = _mean([s["overall"] for s in occ])
        for cls in VEHICLE_CLASSES:
            out[f"curb_occupancy_{cls}"] = _mean([s[f"occ_{cls}"] for s in occ])
        out["mean_price_usd_per_hour"] = _mean([s["mean_price"] for s in occ])
        # Peak-load exposure: share of samples in which the *metered* pool is
        # effectively full (>=90%), the regime in which cruising explodes. The
        # district-wide figure hides this, because slack in loading and TNC
        # zones averages it away.
        out["curb_saturated_share"] = float(
            np.mean([1.0 if s["occ_passenger"] >= 0.90 else 0.0 for s in occ])
        )
    horizon_h = max(1e-9, (result.cfg.horizon_min - result.cfg.warmup_min) / 60.0)
    parked = df.loc[df["outcome"].isin(["parked", "diverted"])]
    out["curb_turnover_per_stall_per_hour"] = (
        len(parked) / result.total_stalls / horizon_h if result.total_stalls else 0.0
    )
    out["loading_zone_utilisation"] = out.get("curb_occupancy_delivery", 0.0)

    # -- network ----------------------------------------------------------------
    out["vmt_miles"] = sum(result.vmt_m.values()) / METRES_PER_MILE
    for cls in VEHICLE_CLASSES:
        out[f"vmt_miles_{cls}"] = result.vmt_m.get(cls, 0.0) / METRES_PER_MILE
    out["cruising_vmt_miles"] = float(df["search_distance_m"].sum()) / METRES_PER_MILE
    out["cruising_vmt_share"] = (
        out["cruising_vmt_miles"] / out["vmt_miles"] if out["vmt_miles"] > 0 else 0.0
    )
    if result.network_samples:
        out["mean_speed_kph"] = _mean([s["mean_speed_kph"] for s in result.network_samples])
        out["congestion_index"] = _mean([s["congestion_index"] for s in result.network_samples])

    # -- system -----------------------------------------------------------------
    illegal_events = int((df["outcome"] == "illegal").sum()) + out.get(
        "ridehail_dropoff_illegal_events", 0
    )
    out["illegal_parking_events"] = illegal_events
    out["illegal_parking_rate"] = illegal_events / len(df) if len(df) else 0.0
    out["citations"] = int(result.events.get("citations", 0))
    out["meter_revenue_usd"] = float(result.revenue_usd)
    out["fine_revenue_usd"] = float(result.fines_usd)
    out["revenue_usd"] = float(result.revenue_usd + result.fines_usd)

    # Social cost. Meter payments and fines are transfers between users and the
    # city, not resource costs, so they are excluded here and reported
    # separately. What is counted: time lost searching/waiting/delayed, the
    # external cost of cruising VMT, and the externality of illegal stops.
    time_cost = 0.0
    for cls, col in (
        ("passenger", "search_time_min"),
        ("delivery", "service_delay_min"),
        ("ridehail", "wait_time_min"),
    ):
        sub = by_class[cls]
        if len(sub):
            time_cost += float(sub[col].sum()) * float(vot[cls]) / 60.0
    out["time_cost_usd"] = time_cost
    out["vmt_cost_usd"] = out["vmt_miles"] * vmt_external
    out["illegal_cost_usd"] = illegal_events * illegal_social
    out["system_social_cost_usd"] = time_cost + out["vmt_cost_usd"] + out["illegal_cost_usd"]
    out["social_cost_per_trip_usd"] = out["system_social_cost_usd"] / len(df)
    return out


def _rate(df: pd.DataFrame, outcome: str) -> float:
    if df is None or len(df) == 0:
        return 0.0
    return float((df["outcome"] == outcome).mean())


def aggregate(summaries: Iterable[dict[str, Any]], confidence: float = 0.95) -> pd.DataFrame:
    """Aggregate replications into mean / sd / half-width of a normal CI.

    Returns a DataFrame indexed by metric with columns
    ``mean``, ``sd``, ``n``, ``ci_low``, ``ci_high``, ``ci_halfwidth``.
    """
    df = pd.DataFrame(list(summaries))
    if df.empty:
        return pd.DataFrame(columns=["mean", "sd", "n", "ci_low", "ci_high", "ci_halfwidth"])
    numeric = df.select_dtypes(include=[np.number])
    numeric = numeric.drop(columns=[c for c in ("seed",) if c in numeric.columns])
    n = len(numeric)
    mean = numeric.mean()
    sd = numeric.std(ddof=1) if n > 1 else numeric.std(ddof=0).fillna(0.0)
    # Normal approximation; with n>=10 replications the t correction is <5%.
    z = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}.get(confidence, 1.960)
    half = z * sd / np.sqrt(n) if n > 1 else pd.Series(0.0, index=mean.index)
    out = pd.DataFrame(
        {
            "mean": mean,
            "sd": sd.fillna(0.0),
            "n": n,
            "ci_low": mean - half,
            "ci_high": mean + half,
            "ci_halfwidth": half.fillna(0.0),
        }
    )
    return out


def compare(
    baseline: pd.DataFrame, scenario: pd.DataFrame, metrics: Sequence[str] | None = None
) -> pd.DataFrame:
    """Percentage change of ``scenario`` against ``baseline`` for each metric."""
    metrics = list(metrics or HEADLINE_METRICS)
    rows = []
    for m in metrics:
        if m not in baseline.index or m not in scenario.index:
            continue
        b = float(baseline.loc[m, "mean"])
        s = float(scenario.loc[m, "mean"])
        pct = (s - b) / b * 100.0 if b else float("nan")
        # Overlapping CIs -> the difference is not distinguishable from noise.
        significant = not (
            scenario.loc[m, "ci_low"] <= baseline.loc[m, "ci_high"]
            and baseline.loc[m, "ci_low"] <= scenario.loc[m, "ci_high"]
        )
        rows.append(
            {
                "metric": m,
                "baseline": b,
                "scenario": s,
                "abs_change": s - b,
                "pct_change": pct,
                "improved": (s < b) if m in LOWER_IS_BETTER else (s > b),
                "significant": bool(significant),
            }
        )
    return pd.DataFrame(rows).set_index("metric")


def occupancy_by_segment(
    result: SimulationResult, t_from: float | None = None, t_to: float | None = None
) -> pd.Series:
    """Mean occupancy of each curb segment over a time window.

    This is the calibration target: it is the quantity a city actually observes
    from meter transactions and in-ground sensors, at the spatial resolution it
    observes it at.
    """
    samples = result.occupancy_samples
    if t_from is not None:
        samples = [s for s in samples if s["t_min"] >= t_from]
    if t_to is not None:
        samples = [s for s in samples if s["t_min"] <= t_to]
    if not samples:
        return pd.Series(dtype=float)
    return pd.DataFrame([s["per_segment"] for s in samples]).mean()


def rmse(a, b) -> float:
    """Root mean squared error between two aligned sequences."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0:
        return float("nan")
    return float(np.mean(np.abs(a - b)))
