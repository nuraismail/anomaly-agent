# Anomaly Agent

An agentic AI tool that autonomously devises, executes and evaluates tests to find anomalies in the cosmic microwave background (CMB).

## Setup

Use Python 3.11 or newer. From the repository root:

```bash
conda create -n cmb-agents python=3.11
conda activate cmb-agents
python -m pip install -e .
```

For development tools:

```bash
python -m pip install -e ".[dev]"
```

Set your OpenRouter API key in the shell or in a local `.env` file:

```bash
cp .env.example .env
```

Then edit `.env` so it contains:

```bash
OPENROUTER_API_KEY=your_key_here
```

The current code expects to be run from the repository root because config and
data paths are repository-relative. Place the required CMB inputs under
`data/input/`.

## Run

```bash
python AnomalyAgent.py
```

After installing the project, this equivalent entry point is also available:

```bash
anomaly-agent
```

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

## Contributors

Harry Chambers
Nura Ismail
Adam Moss

[More to be added...]
