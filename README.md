# Anomaly Agent

An agentic AI tool that autonomously devises, executes and evaluates tests to find anomalies in the cosmic microwave background (CMB).

## Contributors

Harry Chambers
Nura Ismail
Adam Moss

[More to be added...]

## Run analysis

After a run has produced one folder per successful test, compute the run-level
look-elsewhere correction and effective number of tests with:

```bash
python scripts/analyse_run.py data/output/anomaly_agent/default_run
```

The script reads each `result_summary.json`, `planck_statistic.npy`, and
`simulation_statistics.npy` file, recomputes per-test empirical p-values, and
compares Planck's smallest p-value with the distribution of smallest p-values
from the simulated skies. It also rank-normalizes the simulation-statistic
matrix to estimate the effective number of independent tests from the test
correlation matrix. Outputs are written to `<run_dir>/run_analysis/` by default,
including global p-value and effective-test bootstrap histograms when those
bootstrap calculations are enabled.

Use `--n-bootstrap` for the global p-value bootstrap count and
`--n-effective-bootstrap` for the effective-test bootstrap count. Set either to
`0` to disable that uncertainty calculation.
