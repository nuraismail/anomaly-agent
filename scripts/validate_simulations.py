#!/usr/bin/env python3
"""Validate generated simulation maps against a downgraded Planck SMICA map."""

from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np


def import_healpy():
    try:
        import healpy as hp
    except ImportError as exc:
        raise SystemExit(
            "healpy is required to validate maps. Install project dependencies with "
            "`python -m pip install -e .`."
        ) from exc

    return hp


def json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def dl_from_cl(cl: np.ndarray) -> np.ndarray:
    ell = np.arange(cl.size, dtype=np.float64)
    factor = np.ones_like(ell)
    factor[2:] = ell[2:] * (ell[2:] + 1.0) / (2.0 * np.pi)
    dl = np.asarray(cl, dtype=np.float64) * factor
    dl[:2] = 0.0
    return dl


def has_glob_char(path: str) -> bool:
    return any(char in path for char in "*?[")


def load_simulation_arrays(sim_maps: str) -> list[tuple[str, np.ndarray]]:
    paths = sorted(glob(sim_maps)) if has_glob_char(sim_maps) else [sim_maps]
    if not paths:
        raise FileNotFoundError(f"No simulation files matched: {sim_maps}")

    arrays = []
    for path in paths:
        array = np.load(path, allow_pickle=False, mmap_mode="r")
        if array.ndim not in (1, 2):
            raise ValueError(
                f"Expected 1D or 2D simulation array in {path}, "
                f"got shape {array.shape}"
            )
        arrays.append((path, array))
    return arrays


def first_simulation_map(arrays: list[tuple[str, np.ndarray]]) -> np.ndarray:
    first = arrays[0][1]
    return np.asarray(first if first.ndim == 1 else first[0], dtype=np.float64)


def iter_simulation_maps(arrays: list[tuple[str, np.ndarray]], max_maps: int | None):
    count = 0
    for path, array in arrays:
        if array.ndim == 1:
            yield f"{path}[0]", np.asarray(array, dtype=np.float64)
            count += 1
        else:
            for index in range(array.shape[0]):
                yield f"{path}[{index}]", np.asarray(array[index], dtype=np.float64)
                count += 1
                if max_maps is not None and count >= max_maps:
                    return
        if max_maps is not None and count >= max_maps:
            return


def infer_nside_from_map(cmb_map: np.ndarray) -> int:
    hp = import_healpy()
    return int(hp.npix2nside(cmb_map.size))


def read_planck_map_and_mask(
    planck_map_path: Path,
    *,
    nside: int,
    planck_field: int,
    mask_mode: str,
    mask_field: int,
    mask_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    hp = import_healpy()

    planck_map_full = hp.read_map(planck_map_path, field=planck_field)
    planck_map = hp.ud_grade(planck_map_full, nside) * 1.0e6

    if mask_mode == "planck":
        planck_mask_full = hp.read_map(planck_map_path, field=mask_field)
        planck_mask = hp.ud_grade(planck_mask_full, nside) >= mask_threshold
    elif mask_mode == "none":
        planck_mask = np.ones_like(planck_map, dtype=bool)
    else:
        raise ValueError(f"Unknown mask mode: {mask_mode}")

    return np.asarray(planck_map, dtype=np.float64), np.asarray(planck_mask, dtype=bool)


def pseudo_dl(
    cmb_map: np.ndarray,
    mask: np.ndarray,
    *,
    lmax: int,
    subtract_unmasked_mean: bool,
) -> np.ndarray:
    hp = import_healpy()

    working = np.asarray(cmb_map, dtype=np.float64).copy()
    if subtract_unmasked_mean:
        working[mask] -= float(np.mean(working[mask]))
    working[~mask] = 0.0
    return dl_from_cl(hp.anafast(working, lmax=lmax))


def make_ell_bins(ell_min: int, ell_max: int, bin_width: int) -> list[tuple[int, int]]:
    if ell_min < 0:
        raise ValueError("ell-min must be non-negative")
    if ell_max < ell_min:
        raise ValueError("ell-max must be greater than or equal to ell-min")
    if bin_width <= 0:
        raise ValueError("bin-width must be positive")

    return [
        (start, min(start + bin_width - 1, ell_max))
        for start in range(ell_min, ell_max + 1, bin_width)
    ]


def bin_spectra(spectra: np.ndarray, bins: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    ell = np.arange(spectra.shape[-1], dtype=np.float64)
    binned = []
    ell_centers = []
    for start, stop in bins:
        selector = slice(start, stop + 1)
        weights = 2.0 * ell[selector] + 1.0
        binned.append(np.average(spectra[..., selector], axis=-1, weights=weights))
        ell_centers.append(np.average(ell[selector], weights=weights))
    return np.stack(binned, axis=-1), np.asarray(ell_centers, dtype=np.float64)


def covariance_chi_square_summary(
    simulation_binned: np.ndarray,
    planck_binned: np.ndarray,
    *,
    cov_rcond: float,
) -> dict[str, Any]:
    n_sims, n_bins = simulation_binned.shape
    if n_sims < 2:
        raise ValueError("At least two simulations are needed to estimate a covariance")

    mean = np.mean(simulation_binned, axis=0)
    covariance = np.cov(simulation_binned, rowvar=False, ddof=1)
    covariance = np.atleast_2d(covariance)
    covariance_inverse = np.linalg.pinv(covariance, rcond=cov_rcond)

    hartlap_factor = 1.0
    if n_sims > n_bins + 2:
        hartlap_factor = float((n_sims - n_bins - 2) / (n_sims - 1))
    precision = hartlap_factor * covariance_inverse

    planck_delta = planck_binned - mean
    sim_delta = simulation_binned - mean
    planck_chi2 = float(planck_delta @ precision @ planck_delta)
    simulation_chi2 = np.einsum("ij,jk,ik->i", sim_delta, precision, sim_delta)
    empirical_p_value = float(
        (np.count_nonzero(simulation_chi2 >= planck_chi2) + 1) / (n_sims + 1)
    )

    return {
        "chi2_mode": "covariance",
        "n_sims": int(n_sims),
        "n_bins": int(n_bins),
        "n_dof": int(n_bins),
        "hartlap_factor": float(hartlap_factor),
        "cov_rcond": float(cov_rcond),
        "planck_chi2": planck_chi2,
        "planck_reduced_chi2": float(planck_chi2 / n_bins),
        "simulation_chi2_mean": float(np.mean(simulation_chi2)),
        "simulation_chi2_std": float(np.std(simulation_chi2, ddof=1)),
        "simulation_chi2_min": float(np.min(simulation_chi2)),
        "simulation_chi2_max": float(np.max(simulation_chi2)),
        "empirical_p_value_upper": empirical_p_value,
    }


def diagonal_chi_square_summary(
    simulation_spectra: np.ndarray,
    planck_spectrum: np.ndarray,
    *,
    ell_min: int,
    ell_max: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    n_sims = simulation_spectra.shape[0]
    if n_sims < 2:
        raise ValueError("At least two simulations are needed to estimate per-ell variances")

    ell = np.arange(planck_spectrum.size, dtype=np.int32)
    mean = np.mean(simulation_spectra, axis=0)
    variance = np.var(simulation_spectra, axis=0, ddof=1)
    selected = (ell >= ell_min) & (ell <= ell_max)
    valid = selected & np.isfinite(planck_spectrum) & np.isfinite(mean) & (variance > 0.0)
    if not np.any(valid):
        raise ValueError("No valid ell values are available for diagonal chi-square")

    planck_contributions = np.full(planck_spectrum.shape, np.nan, dtype=np.float64)
    planck_contributions[valid] = (planck_spectrum[valid] - mean[valid]) ** 2 / variance[valid]

    sim_delta = simulation_spectra[:, valid] - mean[valid]
    simulation_chi2 = np.sum((sim_delta**2) / variance[valid], axis=1)
    planck_chi2 = float(np.nansum(planck_contributions[valid]))
    n_dof = int(np.count_nonzero(valid))
    empirical_p_value = float(
        (np.count_nonzero(simulation_chi2 >= planck_chi2) + 1) / (n_sims + 1)
    )

    summary = {
        "chi2_mode": "diagonal",
        "n_sims": int(n_sims),
        "ell_min": int(ell_min),
        "ell_max": int(ell_max),
        "n_ell": n_dof,
        "n_dof": n_dof,
        "n_excluded_ell": int(np.count_nonzero(selected & ~valid)),
        "planck_chi2": planck_chi2,
        "planck_reduced_chi2": float(planck_chi2 / n_dof),
        "simulation_chi2_mean": float(np.mean(simulation_chi2)),
        "simulation_chi2_std": float(np.std(simulation_chi2, ddof=1)),
        "simulation_chi2_min": float(np.min(simulation_chi2)),
        "simulation_chi2_max": float(np.max(simulation_chi2)),
        "empirical_p_value_upper": empirical_p_value,
    }
    diagnostics = {
        "simulation_spectrum_mean_dl": mean,
        "simulation_spectrum_variance_dl": variance,
        "planck_chi2_per_ell": planck_contributions,
        "valid_ell_mask": valid,
    }
    return summary, diagnostics


def validate_simulations(args: argparse.Namespace) -> dict[str, Any]:
    hp = import_healpy()

    arrays = load_simulation_arrays(str(args.sim_maps))
    first_map = first_simulation_map(arrays)
    nside = int(args.nside) if args.nside is not None else infer_nside_from_map(first_map)
    expected_npix = hp.nside2npix(nside)
    if first_map.size != expected_npix:
        raise ValueError(f"Simulation map has {first_map.size} pixels, expected {expected_npix}")

    lmax = int(args.lmax) if args.lmax is not None else 3 * nside - 1
    ell_max = int(args.ell_max) if args.ell_max is not None else lmax
    if ell_max > lmax:
        raise ValueError("ell-max cannot exceed lmax")

    planck_map, mask = read_planck_map_and_mask(
        args.planck_map,
        nside=nside,
        planck_field=args.planck_field,
        mask_mode=args.mask_mode,
        mask_field=args.mask_field,
        mask_threshold=args.mask_threshold,
    )

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}. Use --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)

    planck_dl = pseudo_dl(
        planck_map,
        mask,
        lmax=lmax,
        subtract_unmasked_mean=not args.keep_unmasked_mean,
    )

    simulation_spectra = []
    map_ids = []
    for map_index, (map_id, sim_map) in enumerate(
        iter_simulation_maps(arrays, args.max_maps),
        start=1,
    ):
        if sim_map.size != expected_npix:
            raise ValueError(f"{map_id} has {sim_map.size} pixels, expected {expected_npix}")
        simulation_spectra.append(
            pseudo_dl(
                sim_map,
                mask,
                lmax=lmax,
                subtract_unmasked_mean=not args.keep_unmasked_mean,
            )
        )
        map_ids.append(map_id)
        if map_index == 1 or map_index % args.progress_every == 0:
            print(f"Computed spectra for {map_index} simulations")

    if not simulation_spectra:
        raise ValueError("No simulation maps were found")

    simulation_dl = np.asarray(simulation_spectra, dtype=np.float32)
    bins = make_ell_bins(args.ell_min, ell_max, args.bin_width)
    simulation_binned, ell_centers = bin_spectra(simulation_dl, bins)
    planck_binned, _ = bin_spectra(planck_dl, bins)

    diagnostics = {
        "simulation_spectrum_mean_dl": np.mean(simulation_dl, axis=0),
        "simulation_spectrum_variance_dl": np.var(simulation_dl, axis=0, ddof=1),
    }
    if args.chi2_mode == "covariance":
        summary = covariance_chi_square_summary(
            np.asarray(simulation_binned, dtype=np.float64),
            np.asarray(planck_binned, dtype=np.float64),
            cov_rcond=args.cov_rcond,
        )
    elif args.chi2_mode == "diagonal":
        summary, diagnostics = diagonal_chi_square_summary(
            np.asarray(simulation_dl, dtype=np.float64),
            np.asarray(planck_dl, dtype=np.float64),
            ell_min=args.ell_min,
            ell_max=ell_max,
        )
    else:
        raise ValueError(f"Unknown chi2 mode: {args.chi2_mode}")

    ell = np.arange(lmax + 1, dtype=np.int32)
    bin_edges = np.asarray(bins, dtype=np.int32)
    np.save(output_dir / "ell.npy", ell)
    np.save(output_dir / "ell_bin_centers.npy", ell_centers.astype(np.float32))
    np.save(output_dir / "ell_bin_edges.npy", bin_edges)
    np.save(output_dir / "planck_spectrum_dl.npy", planck_dl.astype(np.float32))
    np.save(output_dir / "simulation_spectra_dl.npy", simulation_dl)
    np.save(
        output_dir / "planck_spectrum_binned_dl.npy",
        np.asarray(planck_binned, dtype=np.float32),
    )
    np.save(
        output_dir / "simulation_spectra_binned_dl.npy",
        np.asarray(simulation_binned, dtype=np.float32),
    )
    np.save(
        output_dir / "simulation_spectrum_mean_dl.npy",
        np.asarray(diagnostics["simulation_spectrum_mean_dl"], dtype=np.float32),
    )
    np.save(
        output_dir / "simulation_spectrum_variance_dl.npy",
        np.asarray(diagnostics["simulation_spectrum_variance_dl"], dtype=np.float32),
    )
    if args.chi2_mode == "diagonal":
        np.save(
            output_dir / "planck_chi2_per_ell.npy",
            np.asarray(diagnostics["planck_chi2_per_ell"], dtype=np.float32),
        )
        np.save(output_dir / "valid_ell_mask.npy", diagnostics["valid_ell_mask"])

    manifest = {
        "description": (
            "Pseudo-D_l validation of generated simulation maps against Planck SMICA "
            "processed at the same output resolution."
        ),
        "sim_maps": str(args.sim_maps),
        "planck_map": str(args.planck_map),
        "output_dir": str(output_dir),
        "map_ids": map_ids,
        "nside": int(nside),
        "npix": int(expected_npix),
        "lmax": int(lmax),
        "ell_min": int(args.ell_min),
        "ell_max": int(ell_max),
        "bin_width": int(args.bin_width),
        "n_bins": int(len(bins)),
        "mask_mode": str(args.mask_mode),
        "chi2_mode": str(args.chi2_mode),
        "planck_field": int(args.planck_field),
        "mask_field": int(args.mask_field),
        "mask_threshold": float(args.mask_threshold),
        "f_sky": float(np.mean(mask)),
        "subtract_unmasked_mean": not args.keep_unmasked_mean,
        "summary": summary,
        "outputs": {
            "ell": str(output_dir / "ell.npy"),
            "ell_bin_centers": str(output_dir / "ell_bin_centers.npy"),
            "ell_bin_edges": str(output_dir / "ell_bin_edges.npy"),
            "planck_spectrum_dl": str(output_dir / "planck_spectrum_dl.npy"),
            "simulation_spectra_dl": str(output_dir / "simulation_spectra_dl.npy"),
            "planck_spectrum_binned_dl": str(output_dir / "planck_spectrum_binned_dl.npy"),
            "simulation_spectra_binned_dl": str(
                output_dir / "simulation_spectra_binned_dl.npy"
            ),
            "simulation_spectrum_mean_dl": str(output_dir / "simulation_spectrum_mean_dl.npy"),
            "simulation_spectrum_variance_dl": str(
                output_dir / "simulation_spectrum_variance_dl.npy"
            ),
            "summary": str(output_dir / "summary.json"),
        },
    }
    if args.chi2_mode == "diagonal":
        manifest["outputs"]["planck_chi2_per_ell"] = str(
            output_dir / "planck_chi2_per_ell.npy"
        )
        manifest["outputs"]["valid_ell_mask"] = str(output_dir / "valid_ell_mask.npy")

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(json_ready(manifest), handle, indent=2)

    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate generated CMB simulation maps against downgraded "
            "Planck SMICA spectra."
        ),
    )
    parser.add_argument(
        "--sim-maps",
        default="data/input/CMBmapsPlanckLCDM256.npy",
        help="Simulation .npy stack or glob. Defaults to data/input/CMBmapsPlanckLCDM256.npy.",
    )
    parser.add_argument(
        "--planck-map",
        type=Path,
        default=Path("data/input/COM_CMB_IQU-smica_2048_R3.00_full.fits"),
        help=(
            "Planck SMICA FITS file. Defaults to "
            "data/input/COM_CMB_IQU-smica_2048_R3.00_full.fits."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/output/simulation_validation"),
        help="Directory for validation outputs. Defaults to data/output/simulation_validation.",
    )
    parser.add_argument(
        "--nside",
        type=int,
        default=None,
        help="Output nside. Defaults to the nside inferred from the simulation map pixels.",
    )
    parser.add_argument(
        "--lmax",
        type=int,
        default=None,
        help="Maximum multipole for anafast. Defaults to 3*nside - 1.",
    )
    parser.add_argument(
        "--ell-min",
        type=int,
        default=2,
        help="Minimum ell included in binned goodness-of-fit calculations. Defaults to 2.",
    )
    parser.add_argument(
        "--ell-max",
        type=int,
        default=None,
        help="Maximum ell included in binned goodness-of-fit calculations. Defaults to lmax.",
    )
    parser.add_argument(
        "--bin-width",
        type=int,
        default=20,
        help="Width of ell bins used for covariance and chi-square calculations. Defaults to 20.",
    )
    parser.add_argument(
        "--mask-mode",
        choices=["planck", "none"],
        default="planck",
        help="Mask mode. Use planck for the Planck confidence mask or none for full sky.",
    )
    parser.add_argument(
        "--chi2-mode",
        choices=["covariance", "diagonal"],
        default="covariance",
        help=(
            "Goodness-of-fit mode. Use covariance for binned covariance chi-square "
            "or diagonal for a per-ell diagonal chi-square. Defaults to covariance."
        ),
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.9,
        help="Threshold for the downgraded Planck confidence mask. Defaults to 0.9.",
    )
    parser.add_argument(
        "--planck-field",
        type=int,
        default=0,
        help="FITS field containing the SMICA temperature map. Defaults to 0.",
    )
    parser.add_argument(
        "--mask-field",
        type=int,
        default=3,
        help="FITS field containing the Planck confidence mask. Defaults to 3.",
    )
    parser.add_argument(
        "--cov-rcond",
        type=float,
        default=1.0e-10,
        help="Relative cutoff for pseudo-inverting the binned covariance. Defaults to 1e-10.",
    )
    parser.add_argument(
        "--max-maps",
        type=int,
        default=None,
        help="Optional maximum number of simulation maps to validate.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N maps. Defaults to 100.",
    )
    parser.add_argument(
        "--keep-unmasked-mean",
        action="store_true",
        help="Do not subtract each map's mean over unmasked pixels before computing spectra.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty output directory.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.progress_every <= 0:
        raise ValueError("progress-every must be positive")
    if args.max_maps is not None and args.max_maps <= 0:
        raise ValueError("max-maps must be positive")

    manifest = validate_simulations(args)
    summary = manifest["summary"]
    print(f"Wrote validation outputs: {manifest['output_dir']}")
    print(f"Planck reduced chi2: {summary['planck_reduced_chi2']:.4g}")
    print(f"Empirical upper-tail p-value: {summary['empirical_p_value_upper']:.4g}")


if __name__ == "__main__":
    main()
