"""Calibration of behavioural parameters against observed curb occupancy.

What is being calibrated
------------------------
A city can observe how full each curb segment is over time (meter transactions,
in-ground sensors) far more easily than it can observe *why* drivers behave the
way they do. Calibration therefore treats the segment-level occupancy profile as
the observable and searches the behavioural parameter vector for the values that
best reproduce it.

Calibration / validation split
------------------------------
Fitting and reporting on the same data would measure memorisation, not model
quality. Two independent held-out tests are used:

1. **Temporal hold-out.** The observation window is split in two: parameters are
   fitted to the first period only, and the fitted model is scored on the
   second, which has a different demand level because of the time-of-day
   profile.
2. **Behavioural hold-out.** The illegal-parking rate - the analogue of the
   citation record, a completely different data source from occupancy - is never
   used in fitting and is only ever scored.

A model that fits period 1 well and fails on period 2, or that matches occupancy
while getting illegal parking badly wrong, has been overfitted. Both numbers are
reported side by side for exactly that reason.

Synthetic observations
----------------------
No real meter data is used or implied (see README "Data and limitations"). The
"observed" series is generated from the same engine at a known, hidden parameter
vector plus measurement noise and partial sensor coverage, which makes parameter
recovery checkable: we know the answer, so we can report how close the estimate
got.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

from src.calibration import parameters as P
from src.config import load_scenario
from src.experiments.metrics import mae, occupancy_by_segment, rmse, summarise
from src.experiments.provenance import Manifest
from src.simulation.engine import run_simulation

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

#: The "true" parameters used to generate the synthetic observations. Chosen to
#: sit away from the config defaults so that recovering them is a real test.
GROUND_TRUTH: dict[str, float] = {
    "passenger_alpha_walk": 0.0112,
    "passenger_gamma_search": 0.52,
    "passenger_delta_occupancy": 1.65,
    "passenger_search_radius": 205.0,
    "passenger_compliance": 0.78,
    "delivery_illegal_prob": 0.68,
    "ridehail_gamma_search": 0.85,
}

#: Measurement model for the synthetic sensor network.
SENSOR_NOISE_SD = 0.035  # absolute occupancy points
SENSOR_COVERAGE = 0.85  # share of segments with a working sensor


@dataclass
class Windows:
    """Calibration and validation time windows, in simulation minutes."""

    warmup_min: float
    horizon_min: float

    @property
    def split_min(self) -> float:
        return self.warmup_min + (self.horizon_min - self.warmup_min) / 2.0

    @property
    def calibration(self) -> tuple[float, float]:
        return (self.warmup_min, self.split_min)

    @property
    def validation(self) -> tuple[float, float]:
        return (self.split_min, self.horizon_min)


def make_windows(scenario: str = "baseline") -> Windows:
    cfg = load_scenario(scenario, seed=0)
    return Windows(warmup_min=cfg.warmup_min, horizon_min=cfg.horizon_min)


# ---------------------------------------------------------------------------
# Simulating the observable
# ---------------------------------------------------------------------------
def simulate_occupancy(
    theta: Sequence[float],
    seeds: Sequence[int],
    windows: Windows,
    scenario: str = "baseline",
    extra_overrides: dict[str, Any] | None = None,
) -> tuple[pd.Series, pd.Series, dict[str, float]]:
    """Run the model at ``theta`` and return (calibration, validation) occupancy.

    The third return value carries the behavioural hold-out metrics that are
    scored but never fitted. ``extra_overrides`` is merged on top of the
    parameter vector, which is how tests shorten the horizon without touching
    the shipped configuration.
    """
    overrides = P.to_overrides(theta)
    if extra_overrides:
        overrides = {**overrides, **extra_overrides}
    cal_frames, val_frames, aux = [], [], []
    for seed in seeds:
        cfg = load_scenario(scenario, seed=int(seed), overrides=overrides)
        result = run_simulation(cfg)
        cal_frames.append(occupancy_by_segment(result, *windows.calibration))
        val_frames.append(occupancy_by_segment(result, *windows.validation))
        s = summarise(result)
        aux.append(
            {
                "illegal_parking_rate": s.get("illegal_parking_rate", 0.0),
                "passenger_illegal_rate": s.get("passenger_illegal_rate", 0.0),
                "delivery_illegal_rate": s.get("delivery_illegal_rate", 0.0),
                "passenger_search_time_min": s.get("passenger_search_time_min", 0.0),
            }
        )
    cal = pd.concat(cal_frames, axis=1).mean(axis=1)
    val = pd.concat(val_frames, axis=1).mean(axis=1)
    aux_mean = {k: float(np.mean([a[k] for a in aux])) for k in aux[0]}
    return cal, val, aux_mean


# ---------------------------------------------------------------------------
# Synthetic observations
# ---------------------------------------------------------------------------
def make_synthetic_observations(
    seeds: Sequence[int] = (101, 102, 103, 104, 105),
    scenario: str = "baseline",
    out_dir: Path | None = None,
    rng_seed: int = 7,
) -> pd.DataFrame:
    """Generate and persist a synthetic 'observed' occupancy dataset.

    Emulates a real sensor deployment: partial spatial coverage and additive
    measurement error on every reading.
    """
    windows = make_windows(scenario)
    theta = P.from_dict(GROUND_TRUTH)
    cal, val, aux = simulate_occupancy(theta, seeds, windows, scenario)

    rng = np.random.default_rng(rng_seed)
    segments = list(cal.index)
    covered = rng.random(len(segments)) < SENSOR_COVERAGE

    rows = []
    for i, seg in enumerate(segments):
        if not covered[i]:
            continue
        for window, series in (("calibration", cal), ("validation", val)):
            value = float(series[seg]) + float(rng.normal(0.0, SENSOR_NOISE_SD))
            rows.append(
                {
                    "curb_id": seg,
                    "window": window,
                    "observed_occupancy": float(np.clip(value, 0.0, 1.0)),
                }
            )
    df = pd.DataFrame(rows)

    out_dir = Path(out_dir) if out_dir is not None else DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "observed_occupancy.csv", index=False)
    meta = {
        "note": (
            "SYNTHETIC data generated by src/calibration/calibrate.py. It is not "
            "empirical meter data and must not be interpreted as such."
        ),
        "generator_seeds": list(seeds),
        "sensor_noise_sd": SENSOR_NOISE_SD,
        "sensor_coverage": SENSOR_COVERAGE,
        "ground_truth_parameters": GROUND_TRUTH,
        "held_out_behavioural_targets": aux,
        "windows": {"calibration": windows.calibration, "validation": windows.validation},
    }
    (out_dir / "observed_occupancy_meta.json").write_text(json.dumps(meta, indent=2))
    return df


def load_observations(path: Path | None = None) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    """Load the observed occupancy series and its metadata."""
    d = Path(path) if path is not None else DATA_DIR
    df = pd.read_csv(d / "observed_occupancy.csv")
    meta = json.loads((d / "observed_occupancy_meta.json").read_text())
    cal = df.loc[df["window"] == "calibration"].set_index("curb_id")["observed_occupancy"]
    val = df.loc[df["window"] == "validation"].set_index("curb_id")["observed_occupancy"]
    return cal, val, meta


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------
def occupancy_error(observed: pd.Series, simulated: pd.Series) -> dict[str, float]:
    """RMSE / MAE / bias / correlation between observed and simulated occupancy."""
    common = observed.index.intersection(simulated.index)
    o = observed.loc[common].to_numpy(dtype=float)
    s = simulated.loc[common].to_numpy(dtype=float)
    corr = (
        float(np.corrcoef(o, s)[0, 1])
        if len(common) > 2 and o.std() > 0 and s.std() > 0
        else float("nan")
    )
    return {
        "rmse": rmse(o, s),
        "mae": mae(o, s),
        "bias": float(np.mean(s - o)),
        "corr": corr,
        "n_segments": int(len(common)),
    }


class CalibrationObjective:
    """Callable objective with a memo cache and an evaluation log."""

    def __init__(
        self,
        observed_cal: pd.Series,
        seeds: Sequence[int],
        windows: Windows,
        scenario: str = "baseline",
        extra_overrides: dict[str, Any] | None = None,
    ) -> None:
        self.observed_cal = observed_cal
        self.seeds = list(seeds)
        self.windows = windows
        self.scenario = scenario
        self.extra_overrides = extra_overrides
        self.history: list[dict[str, Any]] = []
        self._cache: dict[tuple, float] = {}

    def __call__(self, theta: Sequence[float]) -> float:
        key = tuple(np.round(np.asarray(theta, dtype=float), 6))
        if key in self._cache:
            return self._cache[key]
        cal, _val, _aux = simulate_occupancy(
            theta, self.seeds, self.windows, self.scenario, self.extra_overrides
        )
        err = occupancy_error(self.observed_cal, cal)
        value = err["rmse"]
        self._cache[key] = value
        self.history.append({"theta": list(map(float, theta)), "rmse": value, "mae": err["mae"]})
        return value

    @property
    def n_evaluations(self) -> int:
        return len(self.history)


# ---------------------------------------------------------------------------
# Calibration drivers
# ---------------------------------------------------------------------------
@dataclass
class CalibrationReport:
    method: str
    theta: dict[str, float]
    calibration_error: dict[str, float]
    validation_error: dict[str, float]
    heldout_behavioural: dict[str, float]
    n_evaluations: int
    wall_time_s: float
    seeds: list[int]
    ground_truth: dict[str, float] = field(default_factory=dict)
    parameter_recovery: dict[str, float] = field(default_factory=dict)
    baseline_error: dict[str, float] = field(default_factory=dict)
    oracle_error: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def calibrate(
    seeds: Sequence[int] = (1, 2, 3),
    method: str = "differential_evolution",
    scenario: str = "baseline",
    maxiter: int = 12,
    popsize: int = 6,
    data_dir: Path | None = None,
    workers: int = 1,
    verbose: bool = True,
) -> CalibrationReport:
    """Fit the behavioural parameters and score them on both hold-outs."""
    observed_cal, observed_val, meta = load_observations(data_dir)
    windows = make_windows(scenario)
    objective = CalibrationObjective(observed_cal, seeds, windows, scenario)

    t0 = time.perf_counter()
    x0 = P.defaults()
    if method == "differential_evolution":
        # `updating="deferred"` is required for parallel evaluation: the
        # population is scored as a batch instead of being updated in place.
        res = differential_evolution(
            objective,
            bounds=P.bounds(),
            maxiter=maxiter,
            popsize=popsize,
            tol=0.01,
            seed=12345,
            polish=False,
            disp=verbose,
            workers=workers,
            updating="deferred" if workers != 1 else "immediate",
        )
        theta = res.x
        n_evals = int(getattr(res, "nfev", 0))
    elif method == "nelder_mead":
        res = minimize(
            objective,
            x0=x0,
            method="Nelder-Mead",
            bounds=P.bounds(),
            options={"maxiter": maxiter * 10, "xatol": 1e-3, "fatol": 1e-4, "disp": verbose},
        )
        theta = res.x
        n_evals = int(getattr(res, "nfev", 0))
    else:
        raise ValueError(f"unknown calibration method: {method}")
    wall = time.perf_counter() - t0

    # Score the fitted parameters: in-sample, out-of-sample, and on the
    # behavioural series that was never fitted.
    cal_sim, val_sim, aux = simulate_occupancy(theta, seeds, windows, scenario)
    cal_err = occupancy_error(observed_cal, cal_sim)
    val_err = occupancy_error(observed_val, val_sim)

    # Uncalibrated reference: how much did calibration actually buy?
    base_cal_sim, base_val_sim, _ = simulate_occupancy(x0, seeds, windows, scenario)
    base_err = {
        "calibration_rmse": occupancy_error(observed_cal, base_cal_sim)["rmse"],
        "validation_rmse": occupancy_error(observed_val, base_val_sim)["rmse"],
    }

    truth = meta.get("ground_truth_parameters", {})

    # Oracle reference: score the parameters the observations were *actually*
    # generated from. Any in-sample RMSE below this is the estimator fitting
    # measurement noise, and the gap between the oracle's validation error and
    # the fitted vector's validation error is the price of that overfitting.
    oracle_err: dict[str, float] = {}
    if truth:
        o_cal, o_val, _ = simulate_occupancy(P.from_dict(truth), seeds, windows, scenario)
        oracle_err = {
            "calibration_rmse": occupancy_error(observed_cal, o_cal)["rmse"],
            "validation_rmse": occupancy_error(observed_val, o_val)["rmse"],
            "measurement_noise_sd": SENSOR_NOISE_SD,
        }

    recovery = {}
    if truth:
        est = P.to_dict(theta)
        for name, true_value in truth.items():
            recovery[name] = float(est[name] - true_value)
        recovery["normalised_l2"] = float(
            np.linalg.norm(P.normalise(theta) - P.normalise(P.from_dict(truth)))
        )

    heldout = {f"sim_{k}": v for k, v in aux.items()}
    for k, v in (meta.get("held_out_behavioural_targets") or {}).items():
        heldout[f"observed_{k}"] = float(v)
        if f"sim_{k}" in heldout:
            heldout[f"abs_error_{k}"] = abs(heldout[f"sim_{k}"] - float(v))

    return CalibrationReport(
        method=method,
        theta=P.to_dict(theta),
        calibration_error=cal_err,
        validation_error=val_err,
        heldout_behavioural=heldout,
        n_evaluations=max(n_evals, objective.n_evaluations),
        wall_time_s=wall,
        seeds=list(seeds),
        ground_truth=truth,
        parameter_recovery=recovery,
        baseline_error=base_err,
        oracle_error=oracle_err,
    )


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m src.calibration.calibrate",
        description="Calibrate behavioural parameters against observed curb occupancy.",
    )
    p.add_argument("--generate", action="store_true", help="(re)generate synthetic observations")
    p.add_argument("--seeds", type=int, default=3, help="replications per objective evaluation")
    p.add_argument(
        "--method",
        default="differential_evolution",
        choices=["differential_evolution", "nelder_mead"],
    )
    p.add_argument("--maxiter", type=int, default=12)
    p.add_argument("--popsize", type=int, default=6)
    p.add_argument("--workers", type=int, default=1, help="parallel objective evaluations")
    p.add_argument("--out", default=str(RESULTS_DIR / "calibration"))
    args = p.parse_args(argv)

    if args.generate:
        df = make_synthetic_observations()
        print(f"Wrote {len(df)} synthetic observations to {DATA_DIR / 'observed_occupancy.csv'}")

    seeds = list(range(1, args.seeds + 1))
    report = calibrate(
        seeds=seeds,
        method=args.method,
        maxiter=args.maxiter,
        popsize=args.popsize,
        workers=args.workers,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration_report.json").write_text(report.to_json())
    manifest = Manifest.create(
        experiment="calibration",
        scenario="baseline",
        seeds=seeds,
        config={"method": args.method, "maxiter": args.maxiter, "popsize": args.popsize},
        extra={"parameters": list(P.PARAM_NAMES)},
    ).finish(report.wall_time_s)
    manifest.write(out / "manifest.json")

    print("\nCalibrated parameters")
    print("-" * 56)
    for k, v in report.theta.items():
        truth = report.ground_truth.get(k)
        extra = f"   (true {truth:.4g})" if truth is not None else ""
        print(f"{k:28s} {v:10.4f}{extra}")
    print("\nOccupancy fit")
    print("-" * 56)
    print(f"{'calibration RMSE':28s} {report.calibration_error['rmse']:10.4f}")
    print(f"{'validation RMSE (held out)':28s} {report.validation_error['rmse']:10.4f}")
    print(f"{'calibration MAE':28s} {report.calibration_error['mae']:10.4f}")
    print(f"{'validation MAE (held out)':28s} {report.validation_error['mae']:10.4f}")
    print(f"{'uncalibrated cal RMSE':28s} {report.baseline_error['calibration_rmse']:10.4f}")
    print(f"{'uncalibrated val RMSE':28s} {report.baseline_error['validation_rmse']:10.4f}")
    if report.oracle_error:
        print(f"{'oracle (true params) cal':28s} {report.oracle_error['calibration_rmse']:10.4f}")
        print(f"{'oracle (true params) val':28s} {report.oracle_error['validation_rmse']:10.4f}")
        print(f"{'measurement noise SD':28s} {report.oracle_error['measurement_noise_sd']:10.4f}")
    print(f"\nObjective evaluations: {report.n_evaluations} in {report.wall_time_s:.1f}s")
    print(f"Written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
