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

Download the Planck SMICA map used by the agent and validation scripts with:

```bash
mkdir -p data/input

curl -L -C - \
  -o data/input/COM_CMB_IQU-smica_2048_R3.00_full.fits \
  https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/component-maps/cmb/COM_CMB_IQU-smica_2048_R3.00_full.fits
```

## Generate simulations

Generate LCDM simulation maps with CAMB and healpy. By default, cosmological
parameters are sampled from Planck-parameter uncertainties:

```bash
python scripts/generate_simulations.py \
  --n-maps 1000 \
  --output data/input/CMBmapsPlanckLCDM256.npy \
  --seed 12345
```

The default output is an agent-ready NumPy stack with shape
`(n_maps, 12*nside^2)` at `nside=256`. Resolution defaults to `auto`: the
generation `nside` and `lmax` are derived from the beam FWHM, output nside, a
beam-sampling target, and a beam-power cutoff. For the default 5 arcmin beam and
`nside=256` output, this recommends synthesis at `nside=2048` with `lmax=4255`,
then downgrades to the output resolution. The script also writes a manifest JSON
next to the map file containing the random seed, cosmological parameters, beam,
derived resolution settings, and any manual overrides.

To instead use fixed Planck mean parameters while varying only the simulated sky
realization:

```bash
python scripts/generate_simulations.py \
  --cosmology-mode fixed \
  --n-maps 1000 \
  --output data/input/CMBmapsPlanckLCDM256_fixed.npy \
  --seed 12345
```

For a cheaper notebook-style run using the original `lmax=1000` choice:

```bash
python scripts/generate_simulations.py \
  --resolution-mode manual \
  --n-maps 1000 \
  --nside-generate 2048 \
  --nside-out 256 \
  --lmax 1000
```

The sampled parameter-uncertainty model follows the original notebook
convention: Planck parameters are sampled independently from Gaussian
uncertainties. This does not include the full Planck parameter covariance.

## Validate simulations

After generating maps, compute pseudo-spectra from the saved output-resolution
simulation maps and compare them with Planck SMICA processed in the same way:

```bash
python scripts/validate_simulations.py \
  --sim-maps data/input/CMBmapsPlanckLCDM256.npy \
  --planck-map data/input/COM_CMB_IQU-smica_2048_R3.00_full.fits \
  --output-dir data/output/simulation_validation \
  --overwrite
```

The validation script downgrades SMICA and the Planck mask to the simulation
`nside`, applies the same binary mask convention as the agent, computes
output-map pseudo-`D_l` spectra with `healpy.anafast`, bins the spectra, and
reports a reduced chi-square plus an empirical upper-tail p-value. It writes the
Planck and simulation spectra, binned spectra, ell bins, and `summary.json` to
the output directory.

For a full-sky diagnostic without the Planck mask, use the diagonal per-ell
chi-square mode:

```bash
python scripts/validate_simulations.py \
  --sim-maps data/input/CMBmapsPlanckLCDM256.npy \
  --planck-map data/input/COM_CMB_IQU-smica_2048_R3.00_full.fits \
  --mask-mode none \
  --chi2-mode diagonal \
  --ell-min 2 \
  --ell-max 767 \
  --output-dir data/output/simulation_validation_unmasked \
  --overwrite
```

This mode sums `(D_l^SMICA - mean(D_l^sim))^2 / Var(D_l^sim)` over individual
ell values and writes `planck_chi2_per_ell.npy` for inspecting which multipoles
contribute most.

## Run

```bash
python anomaly_agent.py
```

To use a specific simulation stack without renaming files, pass `--sim-maps`:

```bash
python anomaly_agent.py \
  --sim-maps data/input/CMBmapsPlanckLCDM256_n10000.npy
```

The same setting can be placed in a run config YAML:

```yaml
paths:
  sim_maps_path: data/input/CMBmapsPlanckLCDM256_n10000.npy
```

After installing the project, this equivalent entry point is also available:

```bash
anomaly-agent
```

Each run writes the effective merged configuration to
`<run_output_dir>/run_config.yaml`. The saved config includes `agent.mode`,
which is `exploratory`, `blind`, or `canonical`.

## Run a blinded spherical-field control

To test how much the prompts steer the agent toward known CMB anomaly families,
use the blinded agent:

```bash
python blind_agent.py \
  --config configs/blind_run_config.example.yaml
```

The blinded agent subclasses the main agent. It describes the inputs to the LLM
as generic masked scalar HEALPix maps on the sphere, disables web/arXiv search,
and avoids exposing Planck/CMB labels in the planner, implementation,
hypothesis, summary, and execution-output prompts. The execution, empirical
p-value calculation, plotting, novelty checks, and output format are shared with
the exploratory agent. Use a fresh `thread_id` for blinded runs, since prior
test names and rejected proposals from a CMB-aware run would leak domain
context.

After installing the project, this equivalent entry point is also available:

```bash
blind-anomaly-agent \
  --config configs/blind_run_config.example.yaml
```

## Run a canonical anomaly

To implement a specified published anomaly rather than ask the agent to invent a
new test, use the canonical agent:

```bash
python canonical_agent.py "cold spot" \
  --config configs/canonical_run_config.example.yaml
```

The canonical agent subclasses the main agent. It replaces the planner with one
that researches the named anomaly and writes an implementation-ready canonical
specification. It also adds a canonical review gate after execution, so an
implementation can be sent back for revision if it materially differs from the
specified literature test. The implementation, execution, plotting, and summary
machinery are shared with the exploratory agent.

After installing the project, this equivalent entry point is also available:

```bash
canonical-anomaly-agent "cold spot" \
  --config configs/canonical_run_config.example.yaml
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

Harry Chambers,
Nura Ismail,
Adam Moss
