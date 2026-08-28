# results/published/

A committed snapshot of the reference run, so the repository demonstrates itself
without anyone having to spend a CPU-hour first. Everything here is reproduced by
`make all`; the manifest records the commit and configuration it came from.

| file | contents |
|---|---|
| `report.html` | the full static report — open it in a browser, no server needed |
| `baseline_aggregate.csv` | baseline metrics, mean / sd / 95% CI over 30 seeds |
| `baseline_manifest.json` | provenance for the run these numbers came from |
| `scenario_means.csv`, `scenario_ci_halfwidth.csv` | every metric × every policy scenario |
| `optimization_summary.json` | baselines, the three searches, and the confirmation runs |
| `pareto_samples.csv` | sampled allocations with the non-dominated set labelled |
| `calibration_report.json` | fitted parameters, both hold-outs, parameter recovery, oracle reference |
| `cross_modal_elasticities.csv`, `demand_levels.csv` | sensitivity study |
| `parallel_benchmark.csv` | wall time and speedup by worker count |
| `sumo_summary_seed1.json` | one replication through the SUMO/TraCI backend |

These are model output on a synthetic district. See the project README's
*Data and limitations*.
