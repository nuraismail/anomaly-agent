from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def get_threshold_specs(summary: dict | None = None):
    if not isinstance(summary, dict):
        return []

    specs = []

    threshold_specs = summary.get("threshold_specs", [])
    thresholds = summary.get("thresholds", [])
    labels = summary.get("threshold_labels", [])

    if isinstance(threshold_specs, list):
        for item in threshold_specs:
            if not (isinstance(item, dict) and "value" in item):
                continue

            label = str(item.get("label", "")).strip()

            if not label:
                continue

            specs.append({
                "value": float(item["value"]),
                "label": label,
                "color": item.get("color", "#6C757D"),
                "linestyle": item.get("linestyle", "--"),
                "linewidth": float(item.get("linewidth", 1.2)),
            })

    if specs:
        return specs

    if isinstance(thresholds, list) and isinstance(labels, list):
        for idx, value in enumerate(thresholds):
            if idx >= len(labels):
                continue
            label = str(labels[idx]).strip()
            if not label:
                continue
            specs.append({
                "value": float(value),
                "label": label,
                "color": "#6C757D",
                "linestyle": "--",
                "linewidth": 1.2,
            })

    return specs

def plot_histogram(ax, planck_stat: float, sim_results: np.ndarray, summary: dict | None = None, test_config: dict = None):
    plot_spec = summary.get("plot_spec", {}) if isinstance(summary, dict) else {}
    default_bins = test_config.get("plot_bins", None) if isinstance(test_config, dict) else 60
    bins = plot_spec.get("bins", default_bins)
    bins = int(bins) if type(bins) != str else bins if bins in ['auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', 'sqrt'] else default_bins
    ax.hist(
        sim_results,
        bins=bins,
        density=True,
        alpha=0.82,
        color="#4C78A8",
        edgecolor="white",
        linewidth=0.6,
        label="Simulations",
    )
    ax.axvline(planck_stat, color="#C1121F", linewidth=2.4, label="Planck")
    
    for spec in get_threshold_specs(summary):
        ax.axvline(
            spec["value"],
            color=spec["color"],
            linestyle=spec["linestyle"],
            linewidth=spec["linewidth"],
            label=spec["label"],
        )

    ax.set_ylabel("Density")
    ax.legend(frameon=False, loc="best")

def plot_ecdf(ax, planck_stat: float, sim_results: np.ndarray):
    sorted_vals = np.sort(sim_results)
    y = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax.step(sorted_vals, y, where="post", color="#4C78A8", linewidth=2.2, label="Simulations")
    planck_rank = np.searchsorted(sorted_vals, planck_stat, side="right") / len(sorted_vals)
    ax.axvline(planck_stat, color="#C1121F", linewidth=2.2, label="Planck")
    ax.axhline(planck_rank, color="#C1121F", linestyle=":", linewidth=1.3)
    ax.set_ylabel("Empirical CDF")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="best")

def plot_rank(ax, planck_stat: float, sim_results: np.ndarray):
    sorted_vals = np.sort(sim_results)
    x = np.arange(1, len(sorted_vals) + 1)
    ax.plot(x, sorted_vals, color="#4C78A8", linewidth=2.0, label="Simulations")
    insertion = np.searchsorted(sorted_vals, planck_stat, side="right")
    ax.scatter(
        max(1, insertion),
        planck_stat,
        color="#C1121F",
        s=54,
        zorder=3,
        label="Planck",
    )
    ax.axhline(planck_stat, color="#C1121F", linestyle=":", linewidth=1.3)
    ax.set_ylabel("Statistic value")
    ax.set_xlabel("Simulation rank")
    ax.legend(frameon=False, loc="best")

def plot_results(planck_stat: float, sim_results: np.ndarray, output_dir: Path, summary: dict | None = None, plot_config: dict = None, test_config: dict = None):
    if isinstance(summary, dict):
        plot_spec = summary.get("plot_spec", {})
        requested = plot_spec.get("kind")

        if requested in {"histogram", "ecdf", "rank"}:
            kind = requested

        unique_count = len(np.unique(np.round(sim_results, decimals=10)))

        if unique_count <= max(12, len(sim_results) // 20):
            kind = "rank"

        q25, q75 = np.percentile(sim_results, [25, 75])
        iqr = q75 - q25
        spread = np.std(sim_results)

        if spread > 0 and abs(np.mean(sim_results) - np.median(sim_results)) > 0.35 * spread:
            kind = "ecdf"
        elif iqr == 0:
            kind = "rank"
        else:
            kind = "histogram"

    plot_spec = summary.get("plot_spec", {}) if isinstance(summary, dict) else {}
    meta = {
        "title": plot_spec.get("title", "Simulation distribution vs Planck"),
        "xlabel": plot_spec.get("xlabel", "Statistic value"),
        "ylabel": plot_spec.get("ylabel"),
    }

    with plt.rc_context(plot_config):
        fig, ax = plt.subplots()
        ax.grid(alpha=0.18, linewidth=0.6)

        if kind == "ecdf":
            plot_ecdf(ax, planck_stat, sim_results, summary)
        elif kind == "rank":
            plot_rank(ax, planck_stat, sim_results, summary)
        else:
            plot_histogram(ax, planck_stat, sim_results, summary, test_config)

        ax.set_title(meta["title"])
        ax.set_xlabel(meta["xlabel"])
        
        if meta["ylabel"]:
            ax.set_ylabel(meta["ylabel"])

        png_path = output_dir / "statistic_figure.png"
        pdf_path = output_dir / "statistic_figure.pdf"
        fig.tight_layout()
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)

    return {"png": str(png_path), "pdf": str(pdf_path), "kind": kind}