# Architecture notes

Why the code is shaped the way it is. These are the decisions that would be
invisible from the file listing but that determine whether the model can be
trusted, extended and re-run a year from now.

## The purity of `run_simulation`

```python
result = run_simulation(cfg)     # RunConfig in, SimulationResult out
```

No filesystem access, no module-level state, no logging side effects, no reading
of anything outside `cfg`. Three consequences follow directly, none of which had
to be built separately:

* **Parallelism.** Replications can be scattered across processes because there
  is nothing to share. Workers rebuild the config from a plain dict, so `fork`
  and `spawn` behave identically.
* **Reproducibility.** A run is a pure function of `(config, seed)`. The test
  asserting bit-identical output for a repeated seed is cheap to write precisely
  because there is no state to reset.
* **Composability.** Calibration and optimization are just objective functions
  wrapped around repeated calls. Neither needed the engine to grow a new API.

## Recording versus aggregating

`MetricsRecorder` stores facts: this trip searched 4.1 minutes, this segment was
72% full at t=140. `experiments/metrics.py` turns facts into statistics. The
split means a new metric can be computed from a stored run without re-simulating,
and metric definitions are unit-testable against hand-built inputs.

## Two backends, one behavioural model

`curb_choice_score` is a free function in `agents/base_agent.py`, used by the
SimPy agents and by the SUMO backend. A model whose behaviour differs between
backends is not one model, and the divergence would be invisible until the two
disagreed for reasons nobody could explain.

The same principle applies to the network: both backends are downstream of
`config/network.yaml`, and a test asserts the SUMO `parkingArea` capacities sum
to exactly the SimPy inventory's stall count.

## Why not `simpy.Resource` for curb stalls

`Resource` is a queueing primitive: an unmet request waits in line. Curb space
is a *balking* system — a driver who finds the block full drives to the next one,
and that cruising is the quantity the model exists to measure. Modelling stalls
as a queue would silently eliminate the phenomenon under study.

Secondarily, the time-of-day scenario changes a segment's regulation mix while
vehicles are parked in it. Occupancy is therefore tracked explicitly, and a
re-allocation is allowed to leave a pool temporarily over-subscribed, which
drains as those vehicles leave — the honest representation of a mid-day
regulation change, and something a fixed-capacity `Resource` cannot express.

## Constraints by construction, not by penalty

The optimizer searches an unconstrained box; `project_to_simplex` maps every
candidate onto the feasible simplex inside the objective. So an infeasible point
cannot be evaluated, cannot be returned, and cannot appear in a trace. Penalty
terms would have required tuning a penalty weight, and would have let the search
spend evaluations in regions that are not curb allocations at all.

## Noise is a first-class concern

Everything that touches the objective assumes it is stochastic:

* evaluations return a standard error alongside the value;
* common random numbers are used across candidate points;
* DE's local polish step is disabled, because it assumes a smooth deterministic
  function;
* candidate optima are re-evaluated on independent confirmation seeds before
  being compared with baselines, because a search's estimate of its own winner is
  biased low by selection;
* scenario comparisons flag non-overlapping confidence intervals rather than
  reporting a percentage change as though it were exact.

## Provenance

`Manifest` records the git commit and dirty flag, the resolved configuration and
its hash, the seeds, the Python version, the platform, the CPU count and the wall
time, next to every result. `RunConfig.fingerprint()` hashes everything except
the seed, so replications of one experiment group themselves automatically.

`scripts/render_results.py` closes the loop by generating the README's results
section from those files, with `--check` available in CI: the documented numbers
cannot silently drift from the numbers the code produces.

## Extension points

| to change | edit |
|---|---|
| a new vehicle class | subclass `BaseAgent`, add a config block, add it to `VEHICLE_CLASSES` |
| a different dispatch policy | `RidehailFleet._try_dispatch` (SimPy) / `TaxiDispatcher.match` (SUMO) |
| a real street network | replace `build_grid_network`; the curb inventory schema is unchanged |
| real observations | write a loader producing `data/observed_occupancy.csv`'s schema |
| a different objective | `config/optimization.yaml` weights, or `CurbObjective._components` |
| segment-level allocation | widen the decision vector in `optimization/objective.py`; the engine already supports per-segment allocations |
