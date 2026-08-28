# results/

Experiment output. Everything here is generated and git-ignored; regenerate with
`make all` (or the individual commands in the project README).

```
results/
├── <scenario>/
│   ├── per_seed.csv        one row per replication, every metric
│   ├── aggregate.csv       mean, sd, n, 95% CI per metric
│   └── manifest.json       git commit, config + hash, seeds, platform, wall time
├── all_scenarios_per_seed.csv
├── scenario_means.csv          metric × scenario
├── scenario_ci_halfwidth.csv   matching CI half-widths
├── parallel_benchmark.csv      wall time and speedup by worker count
├── sensitivity/
│   ├── cross_modal_elasticities.csv
│   └── demand_levels.csv
├── calibration/
│   ├── calibration_report.json   fitted parameters, both hold-outs, recovery
│   └── manifest.json
├── optimization/
│   ├── optimization_summary.json  baselines, search results, confirmation runs
│   ├── confirmed_allocations.csv
│   ├── trace_<method>.csv         every point each search visited
│   └── pareto_samples.csv
├── sumo/                          generated SUMO network + backend summaries
└── report.html                    self-contained static report
```

Do not hand-edit anything here, and do not quote a number from these files
without its manifest: the manifest is what ties a result to the commit and
configuration that produced it.
