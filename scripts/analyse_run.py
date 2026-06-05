#!/usr/bin/env python3
"""Post-run aggregate analysis for anomaly-agent outputs.

This script reads one completed run directory, computes the distribution of the
minimum per-test p-value across simulations, compares Planck's minimum per-test
p-value against that null distribution, and estimates the effective number of
independent tests from the simulation-statistic correlation matrix.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm, rankdata

try:
    import yaml
except ImportError:  # pragma: no cover - only used when PyYAML is unavailable.
    yaml = None


DEFAULT_PLOT_CONFIG = {
    "figure.figsize": (7.2, 4.8),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "DejaVu Serif",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.grid": False,
}


def load_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


@dataclass(frozen=True)
class TestResult:
    index: int
    name: str
    directory: Path
    planck_stat: float
    simulation_statistics: np.ndarray
    tail: str
    planck_p_value: float
    summary: dict[str, Any]


def finite_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    return values[np.isfinite(values)]


def load_finite_array(path: Path) -> np.ndarray:
    values = np.asarray(np.load(path), dtype=float).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite values found in {path}")
    return values


def load_plot_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists() or yaml is None:
        return DEFAULT_PLOT_CONFIG

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    config = dict(DEFAULT_PLOT_CONFIG)
    config.update(loaded)
    return config


def parse_tail(tail: str) -> tuple[str, str | None]:
    normalized = tail.lower().replace("_", "-")

    if "two" in normalized:
        return "two", None

    if "one" not in normalized:
        raise ValueError(f"Cannot identify one- or two-tailed test from tail={tail!r}")

    if "upper" in normalized:
        return "one", "upper"

    if "lower" in normalized:
        return "one", "lower"

    raise ValueError(f"One-tailed test is missing upper/lower direction: tail={tail!r}")


def empirical_p_value(
    value: float,
    reference: np.ndarray,
    tail: str,
    *,
    add_one: bool = True,
) -> float:
    mode, direction = parse_tail(tail)
    reference = finite_values(reference)

    if reference.size == 0:
        raise ValueError("Cannot compute empirical p-value from an empty reference sample")

    exceed_count = int(np.sum(reference >= value))
    below_count = int(np.sum(reference <= value))

    if add_one:
        denominator = reference.size + 1.0
        p_upper = (exceed_count + 1.0) / denominator
        p_lower = (below_count + 1.0) / denominator
    else:
        denominator = float(reference.size)
        p_upper = exceed_count / denominator
        p_lower = below_count / denominator

    if mode == "two":
        return float(min(1.0, 2.0 * min(p_upper, p_lower)))

    if direction == "upper":
        return float(p_upper)

    if direction == "lower":
        return float(p_lower)

    raise ValueError(f"Unsupported tail specification: {tail!r}")


def p_value_for_simulation(
    test: TestResult,
    simulation_index: int,
    *,
    leave_one_out: bool,
) -> float:
    value = float(test.simulation_statistics[simulation_index])

    if not leave_one_out:
        reference = test.simulation_statistics
        return empirical_p_value(value, reference, test.tail, add_one=True)

    reference = np.delete(test.simulation_statistics, simulation_index)
    # Leave-one-out avoids counting the target simulation in its own tail count
    # while still keeping nonzero Monte Carlo p-values.
    return empirical_p_value(value, reference, test.tail, add_one=True)


def load_test_result(test_dir: Path) -> TestResult | None:
    summary_path = test_dir / "result_summary.json"
    sims_path = test_dir / "simulation_statistics.npy"
    planck_path = test_dir / "planck_statistic.npy"

    if not (summary_path.exists() and sims_path.exists() and planck_path.exists()):
        return None

    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    simulation_statistics = load_finite_array(sims_path)
    if simulation_statistics.size == 0:
        raise ValueError(f"No finite simulation statistics found in {sims_path}")

    planck_arr = finite_values(np.load(planck_path))
    if planck_arr.size == 0:
        planck_stat = float(summary["planck_stat"])
    else:
        planck_stat = float(planck_arr[0])

    tail = str(summary.get("tail", "")).strip()
    if not tail:
        raise ValueError(f"Missing tail field in {summary_path}")

    stored_planck_p_value = summary.get("p_value")
    planck_p_value = empirical_p_value(
        planck_stat,
        simulation_statistics,
        tail,
        add_one=True,
    )

    if stored_planck_p_value is not None and not np.isclose(
        float(stored_planck_p_value),
        planck_p_value,
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        print(
            f"Recomputed Planck p-value for {test_dir.name}: "
            f"stored={float(stored_planck_p_value):.6g}, recomputed={planck_p_value:.6g}"
        )

    return TestResult(
        index=int(summary.get("saved_test_index", 0) or 0),
        name=str(summary.get("test_name") or test_dir.name),
        directory=test_dir,
        planck_stat=planck_stat,
        simulation_statistics=simulation_statistics,
        tail=tail,
        planck_p_value=planck_p_value,
        summary=summary,
    )


def load_run(run_dir: Path) -> list[TestResult]:
    tests = []

    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue

        test = load_test_result(child)
        if test is not None:
            tests.append(test)

    tests.sort(key=lambda item: (item.index if item.index > 0 else 10**9, item.directory.name))
    return tests


def common_simulation_count(tests: list[TestResult]) -> int:
    if not tests:
        raise ValueError("No completed test directories found")

    n_sims = min(test.simulation_statistics.size for test in tests)
    if n_sims <= 0:
        raise ValueError("No simulations available for global p-value analysis")

    return int(n_sims)


def compute_simulation_min_p_values(
    tests: list[TestResult],
    *,
    leave_one_out: bool,
) -> tuple[np.ndarray, np.ndarray]:
    n_sims = common_simulation_count(tests)
    min_p_values = np.empty(n_sims, dtype=float)
    min_test_indices = np.empty(n_sims, dtype=int)

    for simulation_index in range(n_sims):
        p_values = np.asarray(
            [
                p_value_for_simulation(test, simulation_index, leave_one_out=leave_one_out)
                for test in tests
            ],
            dtype=float,
        )
        winner = int(np.argmin(p_values))
        min_p_values[simulation_index] = float(p_values[winner])
        min_test_indices[simulation_index] = winner

    return min_p_values, min_test_indices


def bootstrap_global_p_value(
    min_p_values: np.ndarray,
    planck_min_p_value: float,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any] | None:
    if n_bootstrap <= 0:
        return None

    rng = np.random.default_rng(seed)
    n_sims = min_p_values.size
    boot_global_p_values = np.empty(n_bootstrap, dtype=float)

    for idx in range(n_bootstrap):
        sample = rng.choice(min_p_values, size=n_sims, replace=True)
        below_count = int(np.sum(sample <= planck_min_p_value))
        boot_global_p_values[idx] = float((below_count + 1.0) / (n_sims + 1.0))

    return {
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "mean": float(np.mean(boot_global_p_values)),
        "std": float(np.std(boot_global_p_values, ddof=1)),
        "median": float(np.median(boot_global_p_values)),
        "interval_68": [
            float(np.percentile(boot_global_p_values, 16)),
            float(np.percentile(boot_global_p_values, 84)),
        ],
        "interval_95": [
            float(np.percentile(boot_global_p_values, 2.5)),
            float(np.percentile(boot_global_p_values, 97.5)),
        ],
        "values": boot_global_p_values,
    }


def simulation_statistic_matrix(tests: list[TestResult]) -> np.ndarray:
    return np.column_stack([test.simulation_statistics for test in tests])


def rank_normalize_columns(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a two-dimensional matrix, got shape {matrix.shape}")

    scores = np.empty_like(matrix, dtype=float)

    for column_index in range(matrix.shape[1]):
        column = matrix[:, column_index]
        ranks = rankdata(column, method="average")
        quantiles = (ranks - 0.5) / column.size
        quantiles = np.clip(quantiles, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
        z_scores = norm.ppf(quantiles)
        z_scores -= np.mean(z_scores)
        std = float(np.std(z_scores))
        scores[:, column_index] = z_scores / std if std > 0.0 and np.isfinite(std) else 0.0

    return scores


def correlation_from_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError(f"Expected a two-dimensional score matrix, got shape {scores.shape}")

    n_tests = scores.shape[1]
    if n_tests == 1:
        valid = float(np.std(scores[:, 0])) > 0.0
        return np.asarray([[1.0 if valid else 0.0]], dtype=float)

    corr = np.corrcoef(scores, rowvar=False)
    corr = np.atleast_2d(np.asarray(corr, dtype=float))
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = 0.5 * (corr + corr.T)

    valid_columns = np.std(scores, axis=0) > 0.0
    for idx, valid in enumerate(valid_columns):
        corr[idx, idx] = 1.0 if bool(valid) else 0.0

    return corr


def effective_tests_from_corr(corr: np.ndarray) -> tuple[float, np.ndarray]:
    eigvals = np.linalg.eigvalsh(corr)
    eigvals = np.clip(np.real(eigvals), 0.0, None)
    total = float(np.sum(eigvals))
    squared_total = float(np.sum(eigvals**2))

    if total <= 0.0 or squared_total <= 0.0:
        raise ValueError("Degenerate correlation matrix eigenspectrum")

    n_eff = (total**2) / squared_total
    return float(n_eff), np.sort(eigvals)[::-1]


def estimate_effective_tests(matrix: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    scores = rank_normalize_columns(matrix)
    corr = correlation_from_scores(scores)
    n_eff, eigvals = effective_tests_from_corr(corr)
    return n_eff, eigvals, corr


def bootstrap_effective_tests(
    matrix: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any] | None:
    if n_bootstrap <= 0:
        return None

    rng = np.random.default_rng(seed)
    n_sims = matrix.shape[0]
    bootstrap_n_eff = np.empty(n_bootstrap, dtype=float)

    for idx in range(n_bootstrap):
        row_idx = rng.integers(0, n_sims, size=n_sims)
        try:
            bootstrap_n_eff[idx], _, _ = estimate_effective_tests(matrix[row_idx, :])
        except ValueError:
            bootstrap_n_eff[idx] = np.nan

    valid_values = finite_values(bootstrap_n_eff)
    if valid_values.size == 0:
        raise ValueError("No valid effective-test bootstrap replicates")

    return {
        "n_bootstrap": int(n_bootstrap),
        "n_valid_bootstrap": int(valid_values.size),
        "seed": int(seed),
        "mean": float(np.mean(valid_values)),
        "std": float(np.std(valid_values, ddof=1)) if valid_values.size > 1 else 0.0,
        "median": float(np.median(valid_values)),
        "interval_68": [
            float(np.percentile(valid_values, 16)),
            float(np.percentile(valid_values, 84)),
        ],
        "interval_95": [
            float(np.percentile(valid_values, 2.5)),
            float(np.percentile(valid_values, 97.5)),
        ],
        "values": bootstrap_n_eff,
    }


def novelty_verdict_matrix(tests: list[TestResult]) -> dict[str, Any]:
    novelty_labels = ["repeat", "variation", "novel"]
    verdict_labels = ["no anomalies found", "borderline", "anomalies found"]
    matrix = np.zeros((len(novelty_labels), len(verdict_labels)), dtype=int)

    novelty_lookup = {label: idx for idx, label in enumerate(novelty_labels)}
    verdict_lookup = {label: idx for idx, label in enumerate(verdict_labels)}

    for test in tests:
        test_summary = test.summary.get("test_summary", {})
        if not isinstance(test_summary, dict):
            continue

        novelty = str(test_summary.get("Test novelty", "")).strip().lower()
        verdict = str(test_summary.get("Verdict", "")).strip().lower()

        if novelty in novelty_lookup and verdict in verdict_lookup:
            matrix[novelty_lookup[novelty], verdict_lookup[verdict]] += 1

    return {
        "novelty_labels": novelty_labels,
        "verdict_labels": verdict_labels,
        "matrix": matrix,
    }


def save_histogram(
    min_p_values: np.ndarray,
    planck_min_p_value: float,
    global_p_value: float,
    output_dir: Path,
    plot_config: dict[str, Any],
) -> None:
    plt = load_pyplot()

    with plt.rc_context(plot_config):
        fig, ax = plt.subplots()
        ax.hist(
            min_p_values,
            bins="auto",
            density=True,
            alpha=0.82,
            color="#4C78A8",
            edgecolor="white",
            linewidth=0.6,
            label="Simulated skies",
        )
        ax.axvline(
            planck_min_p_value,
            color="#C1121F",
            linewidth=2.4,
            label=f"Planck min p = {planck_min_p_value:.4g}",
        )
        ax.axvline(
            float(np.mean(min_p_values)),
            color="#2F855A",
            linestyle="--",
            linewidth=1.8,
            label="Simulation mean",
        )
        ax.set_title("Global minimum p-value distribution")
        ax.set_xlabel("Minimum per-test p-value")
        ax.set_ylabel("Density")
        ax.text(
            0.98,
            0.96,
            f"global p = {global_p_value:.4g}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
        )
        ax.legend(frameon=False, loc="best")
        fig.tight_layout()
        fig.savefig(output_dir / "global_min_pvalue_histogram.png", dpi=300, bbox_inches="tight")
        fig.savefig(output_dir / "global_min_pvalue_histogram.pdf", bbox_inches="tight")
        plt.close(fig)


def save_global_bootstrap_plot(
    global_p_value: float,
    bootstrap: dict[str, Any] | None,
    output_dir: Path,
    plot_config: dict[str, Any],
) -> None:
    if bootstrap is None:
        return

    plt = load_pyplot()

    values = finite_values(np.asarray(bootstrap["values"], dtype=float))
    if values.size == 0:
        return

    interval_68 = bootstrap["interval_68"]
    interval_95 = bootstrap["interval_95"]

    with plt.rc_context(plot_config):
        fig, ax = plt.subplots()
        ax.hist(
            values,
            bins=35,
            color="#4C78A8",
            edgecolor="white",
            linewidth=0.7,
            alpha=0.9,
        )
        ax.axvline(
            global_p_value,
            color="#C1121F",
            linewidth=2.0,
            label=f"Point estimate = {global_p_value:.4g}",
        )
        ax.axvspan(
            interval_68[0],
            interval_68[1],
            color="#F4A261",
            alpha=0.22,
            label=f"68% interval: {interval_68[0]:.4g}-{interval_68[1]:.4g}",
        )
        ax.axvspan(
            interval_95[0],
            interval_95[1],
            color="#6C757D",
            alpha=0.12,
            label=f"95% interval: {interval_95[0]:.4g}-{interval_95[1]:.4g}",
        )
        ax.set_title("Global p-value bootstrap uncertainty")
        ax.set_xlabel("Global p-value")
        ax.set_ylabel("Bootstrap replicates")
        ax.legend(frameon=False, loc="best")
        fig.tight_layout()
        fig.savefig(output_dir / "global_pvalue_bootstrap_distribution.png", dpi=300, bbox_inches="tight")
        fig.savefig(output_dir / "global_pvalue_bootstrap_distribution.pdf", bbox_inches="tight")
        plt.close(fig)


def save_test_index_histogram(
    min_test_indices: np.ndarray,
    tests: list[TestResult],
    output_dir: Path,
    plot_config: dict[str, Any],
) -> None:
    plt = load_pyplot()

    labels = [f"{idx + 1}" for idx in range(len(tests))]
    counts = np.bincount(min_test_indices, minlength=len(tests))

    with plt.rc_context(plot_config):
        fig, ax = plt.subplots()
        ax.bar(labels, counts, color="#4C78A8", edgecolor="white", linewidth=0.6)
        ax.set_title("Test producing smallest simulated p-value")
        ax.set_xlabel("Test index")
        ax.set_ylabel("Simulation count")
        fig.tight_layout()
        fig.savefig(output_dir / "global_min_pvalue_test_index_histogram.png", dpi=300, bbox_inches="tight")
        fig.savefig(output_dir / "global_min_pvalue_test_index_histogram.pdf", bbox_inches="tight")
        plt.close(fig)


def save_novelty_verdict_matrix(
    matrix_info: dict[str, Any],
    output_dir: Path,
    plot_config: dict[str, Any],
) -> None:
    plt = load_pyplot()

    matrix = np.asarray(matrix_info["matrix"], dtype=int)

    with plt.rc_context(plot_config):
        fig, ax = plt.subplots()
        image = ax.imshow(matrix, cmap="Blues", vmin=0)
        fig.colorbar(image, ax=ax, label="No. of tests")
        ax.set_xticks(np.arange(len(matrix_info["verdict_labels"])))
        ax.set_xticklabels(matrix_info["verdict_labels"], rotation=25, ha="right")
        ax.set_yticks(np.arange(len(matrix_info["novelty_labels"])))
        ax.set_yticklabels(matrix_info["novelty_labels"])
        ax.set_title("Test novelty vs verdict")

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")

        fig.tight_layout()
        fig.savefig(output_dir / "global_novelty_verdict_matrix.png", dpi=300, bbox_inches="tight")
        fig.savefig(output_dir / "global_novelty_verdict_matrix.pdf", bbox_inches="tight")
        plt.close(fig)


def save_effective_tests_bootstrap_plot(
    n_eff_point_estimate: float,
    bootstrap: dict[str, Any] | None,
    output_dir: Path,
    plot_config: dict[str, Any],
) -> None:
    if bootstrap is None:
        return

    plt = load_pyplot()

    values = finite_values(np.asarray(bootstrap["values"], dtype=float))
    if values.size == 0:
        return

    interval_68 = bootstrap["interval_68"]
    interval_95 = bootstrap["interval_95"]

    with plt.rc_context(plot_config):
        fig, ax = plt.subplots()
        ax.hist(
            values,
            bins=35,
            color="#4C78A8",
            edgecolor="white",
            linewidth=0.7,
            alpha=0.9,
        )
        ax.axvline(
            n_eff_point_estimate,
            color="#C1121F",
            linewidth=2.0,
            label=f"Point estimate = {n_eff_point_estimate:.2f}",
        )
        ax.axvspan(
            interval_68[0],
            interval_68[1],
            color="#F4A261",
            alpha=0.22,
            label=f"68% interval: {interval_68[0]:.2f}-{interval_68[1]:.2f}",
        )
        ax.axvspan(
            interval_95[0],
            interval_95[1],
            color="#6C757D",
            alpha=0.12,
            label=f"95% interval: {interval_95[0]:.2f}-{interval_95[1]:.2f}",
        )
        ax.set_title("Effective number of tests")
        ax.set_xlabel("Effective number of independent tests")
        ax.set_ylabel("Bootstrap replicates")
        ax.legend(frameon=False, loc="best")
        fig.tight_layout()
        fig.savefig(output_dir / "effective_tests_bootstrap_distribution.png", dpi=300, bbox_inches="tight")
        fig.savefig(output_dir / "effective_tests_bootstrap_distribution.pdf", bbox_inches="tight")
        plt.close(fig)


def save_effective_tests_correlation_matrix(
    corr: np.ndarray,
    output_dir: Path,
    plot_config: dict[str, Any],
) -> None:
    plt = load_pyplot()

    n_tests = corr.shape[0]
    tick_step = max(1, int(np.ceil(n_tests / 25)))
    tick_positions = np.arange(0, n_tests, tick_step)
    tick_labels = [str(idx + 1) for idx in tick_positions]
    figsize = (max(6.0, min(12.0, 0.28 * n_tests + 3.0)), max(5.0, min(12.0, 0.28 * n_tests + 2.5)))

    with plt.rc_context(plot_config):
        fig, ax = plt.subplots(figsize=figsize)
        image = ax.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        fig.colorbar(image, ax=ax, label="Rank-normalized correlation")
        ax.set_title("Simulation-statistic correlation matrix")
        ax.set_xlabel("Test index")
        ax.set_ylabel("Test index")
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=90 if n_tests > 15 else 0)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)
        fig.tight_layout()
        fig.savefig(output_dir / "effective_tests_correlation_matrix.png", dpi=300, bbox_inches="tight")
        fig.savefig(output_dir / "effective_tests_correlation_matrix.pdf", bbox_inches="tight")
        plt.close(fig)


def serializable_bootstrap(bootstrap: dict[str, Any] | None) -> dict[str, Any] | None:
    if bootstrap is None:
        return None

    return {key: value for key, value in bootstrap.items() if key != "values"}


def write_summary(
    output_dir: Path,
    run_dir: Path,
    tests: list[TestResult],
    min_p_values: np.ndarray,
    min_test_indices: np.ndarray,
    planck_min_index: int,
    global_p_value: float,
    global_sigma: float,
    bootstrap: dict[str, Any] | None,
    matrix_info: dict[str, Any],
    leave_one_out: bool,
) -> dict[str, Any]:
    test_summaries = []

    for idx, test in enumerate(tests):
        test_summaries.append(
            {
                "index": int(idx + 1),
                "saved_test_index": int(test.index),
                "name": test.name,
                "directory": str(test.directory),
                "tail": test.tail,
                "planck_stat": float(test.planck_stat),
                "planck_p_value": float(test.planck_p_value),
                "n_sims": int(test.simulation_statistics.size),
            }
        )

    summary = {
        "run_dir": str(run_dir),
        "n_tests": int(len(tests)),
        "n_sims_used": int(min_p_values.size),
        "simulation_p_values": "leave-one-out" if leave_one_out else "include-own-simulation",
        "planck_min_p_value": float(tests[planck_min_index].planck_p_value),
        "planck_min_test_index": int(planck_min_index + 1),
        "planck_min_test_name": tests[planck_min_index].name,
        "global_p_value": float(global_p_value),
        "global_sigma": float(global_sigma),
        "min_p_values_mean": float(np.mean(min_p_values)),
        "min_p_values_std": float(np.std(min_p_values, ddof=1)),
        "min_p_values_median": float(np.median(min_p_values)),
        "min_p_values_q16": float(np.percentile(min_p_values, 16)),
        "min_p_values_q84": float(np.percentile(min_p_values, 84)),
        "tests": test_summaries,
        "bootstrap": serializable_bootstrap(bootstrap),
        "novelty_verdict": {
            "novelty_labels": matrix_info["novelty_labels"],
            "verdict_labels": matrix_info["verdict_labels"],
            "matrix": np.asarray(matrix_info["matrix"], dtype=int).tolist(),
        },
        "outputs": {
            "summary_json": str(output_dir / "global_pvalue_summary.json"),
            "min_p_values_npy": str(output_dir / "global_min_p_values.npy"),
            "min_test_indices_npy": str(output_dir / "global_min_test_indices.npy"),
            "bootstrap_npy": str(output_dir / "global_bootstrap_p_values.npy") if bootstrap is not None else None,
            "histogram_png": str(output_dir / "global_min_pvalue_histogram.png"),
            "histogram_pdf": str(output_dir / "global_min_pvalue_histogram.pdf"),
            "bootstrap_distribution_png": str(output_dir / "global_pvalue_bootstrap_distribution.png")
            if bootstrap is not None
            else None,
            "bootstrap_distribution_pdf": str(output_dir / "global_pvalue_bootstrap_distribution.pdf")
            if bootstrap is not None
            else None,
            "test_index_histogram_png": str(output_dir / "global_min_pvalue_test_index_histogram.png"),
            "test_index_histogram_pdf": str(output_dir / "global_min_pvalue_test_index_histogram.pdf"),
            "novelty_verdict_png": str(output_dir / "global_novelty_verdict_matrix.png"),
            "novelty_verdict_pdf": str(output_dir / "global_novelty_verdict_matrix.pdf"),
        },
    }

    with (output_dir / "global_pvalue_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary


def analyse_effective_tests(
    tests: list[TestResult],
    output_dir: Path,
    plot_config: dict[str, Any],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    matrix = simulation_statistic_matrix(tests)
    n_eff, eigvals, corr = estimate_effective_tests(matrix)
    bootstrap = bootstrap_effective_tests(matrix, n_bootstrap=n_bootstrap, seed=seed)

    metadata = [
        {
            "column_index": int(idx),
            "test_index": int(idx + 1),
            "saved_test_index": int(test.index),
            "test_name": test.name,
            "directory": str(test.directory),
        }
        for idx, test in enumerate(tests)
    ]

    np.save(output_dir / "effective_tests_eigenvalues.npy", eigvals)
    np.save(output_dir / "effective_tests_correlation_matrix.npy", corr)
    if bootstrap is not None:
        np.save(output_dir / "effective_tests_bootstrap.npy", bootstrap["values"])

    with (output_dir / "effective_tests_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    save_effective_tests_bootstrap_plot(n_eff, bootstrap, output_dir, plot_config)
    save_effective_tests_correlation_matrix(corr, output_dir, plot_config)

    summary = {
        "method": (
            "Simulation statistics are rank-normalized column by column, their "
            "correlation matrix is eigendecomposed, and N_eff is estimated as "
            "(sum(lambda)^2) / sum(lambda^2). Bootstrap intervals resample "
            "simulation rows with replacement."
        ),
        "n_tests": int(matrix.shape[1]),
        "n_simulations": int(matrix.shape[0]),
        "n_eff_point_estimate": float(n_eff),
        "n_eff_fraction_of_tests": float(n_eff / matrix.shape[1]) if matrix.shape[1] else 0.0,
        "eigenvalues": eigvals.tolist(),
        "bootstrap": serializable_bootstrap(bootstrap),
        "outputs": {
            "summary_json": str(output_dir / "effective_tests_summary.json"),
            "metadata_json": str(output_dir / "effective_tests_metadata.json"),
            "bootstrap_npy": str(output_dir / "effective_tests_bootstrap.npy") if bootstrap is not None else None,
            "eigenvalues_npy": str(output_dir / "effective_tests_eigenvalues.npy"),
            "correlation_matrix_npy": str(output_dir / "effective_tests_correlation_matrix.npy"),
            "bootstrap_distribution_png": str(output_dir / "effective_tests_bootstrap_distribution.png")
            if bootstrap is not None
            else None,
            "bootstrap_distribution_pdf": str(output_dir / "effective_tests_bootstrap_distribution.pdf")
            if bootstrap is not None
            else None,
            "correlation_matrix_png": str(output_dir / "effective_tests_correlation_matrix.png"),
            "correlation_matrix_pdf": str(output_dir / "effective_tests_correlation_matrix.pdf"),
        },
    }

    with (output_dir / "effective_tests_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary


def write_run_analysis_summary(
    output_dir: Path,
    global_summary: dict[str, Any],
    effective_summary: dict[str, Any],
) -> dict[str, Any]:
    summary = {
        "run_dir": global_summary["run_dir"],
        "n_tests": global_summary["n_tests"],
        "n_sims_used": global_summary["n_sims_used"],
        "global_pvalue": {
            "planck_min_p_value": global_summary["planck_min_p_value"],
            "planck_min_test_index": global_summary["planck_min_test_index"],
            "planck_min_test_name": global_summary["planck_min_test_name"],
            "global_p_value": global_summary["global_p_value"],
            "global_sigma": global_summary["global_sigma"],
        },
        "effective_tests": {
            "n_eff_point_estimate": effective_summary["n_eff_point_estimate"],
            "n_eff_fraction_of_tests": effective_summary["n_eff_fraction_of_tests"],
            "bootstrap": effective_summary["bootstrap"],
        },
        "outputs": {
            "run_analysis_summary_json": str(output_dir / "run_analysis_summary.json"),
            "global_pvalue_summary_json": global_summary["outputs"]["summary_json"],
            "effective_tests_summary_json": effective_summary["outputs"]["summary_json"],
        },
    }

    with (output_dir / "run_analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary


def analyse_run(
    run_dir: Path,
    *,
    output_dir: Path | None,
    plot_config_path: Path | None,
    n_bootstrap: int,
    n_effective_bootstrap: int,
    seed: int,
    leave_one_out: bool,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    if output_dir is None:
        output_dir = run_dir / "run_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    tests = load_run(run_dir)
    if not tests:
        raise ValueError(f"No completed test outputs found under {run_dir}")

    n_sims = common_simulation_count(tests)
    if any(test.simulation_statistics.size != n_sims for test in tests):
        print(f"Using first {n_sims} simulations from each test because counts differ.")
        tests = [
            TestResult(
                index=test.index,
                name=test.name,
                directory=test.directory,
                planck_stat=test.planck_stat,
                simulation_statistics=test.simulation_statistics[:n_sims],
                tail=test.tail,
                planck_p_value=test.planck_p_value,
                summary=test.summary,
            )
            for test in tests
        ]

    planck_p_values = np.asarray([test.planck_p_value for test in tests], dtype=float)
    planck_min_index = int(np.argmin(planck_p_values))
    planck_min_p_value = float(planck_p_values[planck_min_index])

    min_p_values, min_test_indices = compute_simulation_min_p_values(
        tests,
        leave_one_out=leave_one_out,
    )

    global_below_count = int(np.sum(min_p_values <= planck_min_p_value))
    global_p_value = float((global_below_count + 1.0) / (min_p_values.size + 1.0))
    global_sigma = 0.0 if global_p_value >= 1.0 else float(norm.isf(global_p_value))

    bootstrap = bootstrap_global_p_value(
        min_p_values,
        planck_min_p_value,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    matrix_info = novelty_verdict_matrix(tests)
    plot_config = load_plot_config(plot_config_path)

    np.save(output_dir / "global_min_p_values.npy", min_p_values)
    np.save(output_dir / "global_min_test_indices.npy", min_test_indices)
    if bootstrap is not None:
        np.save(output_dir / "global_bootstrap_p_values.npy", bootstrap["values"])

    save_histogram(min_p_values, planck_min_p_value, global_p_value, output_dir, plot_config)
    save_global_bootstrap_plot(global_p_value, bootstrap, output_dir, plot_config)
    save_test_index_histogram(min_test_indices, tests, output_dir, plot_config)
    save_novelty_verdict_matrix(matrix_info, output_dir, plot_config)

    global_summary = write_summary(
        output_dir,
        run_dir,
        tests,
        min_p_values,
        min_test_indices,
        planck_min_index,
        global_p_value,
        global_sigma,
        bootstrap,
        matrix_info,
        leave_one_out,
    )

    effective_summary = analyse_effective_tests(
        tests,
        output_dir,
        plot_config,
        n_bootstrap=n_effective_bootstrap,
        seed=seed,
    )

    return write_run_analysis_summary(output_dir, global_summary, effective_summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute run-level aggregate diagnostics for an anomaly-agent output directory.",
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Run output directory containing Test_* subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for aggregate run-analysis outputs. Defaults to <run_dir>/run_analysis.",
    )
    parser.add_argument(
        "--plot-config",
        type=Path,
        default=Path("configs/plot_config.yaml"),
        help="YAML matplotlib rcParams file. Defaults to configs/plot_config.yaml.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=50000,
        help="Number of bootstrap resamples for global p-value uncertainty. Use 0 to disable.",
    )
    parser.add_argument(
        "--n-effective-bootstrap",
        type=int,
        default=5000,
        help="Number of bootstrap resamples for effective-test uncertainty. Use 0 to disable.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for bootstrap resampling.",
    )
    parser.add_argument(
        "--include-own-simulation",
        action="store_true",
        help="For simulated-sky p-values, include that sky in its own reference distribution.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = analyse_run(
        args.run_dir,
        output_dir=args.output_dir,
        plot_config_path=args.plot_config,
        n_bootstrap=args.n_bootstrap,
        n_effective_bootstrap=args.n_effective_bootstrap,
        seed=args.seed,
        leave_one_out=not args.include_own_simulation,
    )

    print(f"Analysed {summary['n_tests']} tests with {summary['n_sims_used']} simulations.")
    print(
        "Planck minimum p-value: "
        f"{summary['global_pvalue']['planck_min_p_value']:.6g} "
        f"({summary['global_pvalue']['planck_min_test_name']})"
    )
    print(
        "Global p-value: "
        f"{summary['global_pvalue']['global_p_value']:.6g}; "
        f"global sigma: {summary['global_pvalue']['global_sigma']:.3f}"
    )
    print(
        "Effective tests: "
        f"{summary['effective_tests']['n_eff_point_estimate']:.3f} / {summary['n_tests']} "
        f"({summary['effective_tests']['n_eff_fraction_of_tests']:.3f} of nominal count)"
    )
    print(f"Wrote {summary['outputs']['run_analysis_summary_json']}")


if __name__ == "__main__":
    main()
