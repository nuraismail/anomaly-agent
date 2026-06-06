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

## Contributors

Harry Chambers
Nura Ismail
Adam Moss

[More to be added...]
