import file_paths
import numpy as np
import yaml

def build_default_summary(planck_stat: float, sim_results: np.ndarray, source: str = "fallback_missing"):
    sim_results = np.asarray(sim_results, dtype=float)
    valid_sim = sim_results[np.isfinite(sim_results)]

    if valid_sim.size:
        sim_mean = float(np.mean(valid_sim))
        sim_std = float(np.std(valid_sim))
        sim_min = float(np.min(valid_sim))
        sim_max = float(np.max(valid_sim))
        q025, q975 = np.percentile(valid_sim, [2.5, 97.5])
        thresholds = [float(q025), float(q975)]
    else:
        sim_mean = None
        sim_std = None
        sim_min = None
        sim_max = None
        thresholds = []

    z_score = None
    if np.isfinite(planck_stat) and sim_mean is not None:
        if sim_std and sim_std > 0:
            z_score = float((planck_stat - sim_mean) / sim_std)
        elif np.isclose(planck_stat, sim_mean):
            z_score = 0.0

    is_null_test = bool(
        (not np.isfinite(planck_stat))
        or valid_sim.size == 0
        or (sim_std is not None and np.isclose(sim_std, 0.0) and sim_mean is not None and np.isclose(planck_stat, sim_mean))
    )

    return {
        "is_null_test": is_null_test,
        "summary_source": source,
        "n_valid_simulations": int(valid_sim.size),
        "sim_mean": sim_mean,
        "sim_std": sim_std,
        "sim_min": sim_min,
        "sim_max": sim_max,
        "z_score": z_score,
        "thresholds": thresholds,
    }

def compact_summary(summary: dict | None):
    if not isinstance(summary, dict):
        return summary

    with open(file_paths.cmb_dict_dir) as stream:
        redundant_keys = yaml.safe_load(stream)["redundant_keys"]
        return {key: value for key, value in summary.items() if key not in redundant_keys}