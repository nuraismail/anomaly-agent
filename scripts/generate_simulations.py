#!/usr/bin/env python3
"""Generate LCDM CMB temperature simulations with Planck parameters.

It can sample cosmological parameters from simple independent Gaussian Planck
uncertainties or hold them fixed at their Planck mean values. It generates
lensed TT spectra with CAMB, simulates CMB temperature maps with healpy.synfast,
and writes an agent-ready NumPy stack.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


PLANCK_PARAMETER_PRIORS = {
    "H0": {"mean": 67.66, "sigma": 0.42},
    "ombh2": {"mean": 0.02234, "sigma": 0.00014},
    "omch2": {"mean": 0.11907, "sigma": 0.00094},
    "tau": {"mean": 0.05520, "sigma": 0.00715},
    "As": {"mean": 2.097e-9, "sigma": 0.030e-9},
    "ns": {"mean": 0.9671, "sigma": 0.0038},
}

FIXED_PARAMETERS = {
    "mnu": 0.06,
    "omk": 0.0,
}


def import_camb():
    try:
        import camb
    except ImportError as exc:
        raise SystemExit(
            "CAMB is required to generate maps. Install project dependencies with "
            "`python -m pip install -e .`."
        ) from exc

    return camb


def import_healpy():
    try:
        import healpy as hp
    except ImportError as exc:
        raise SystemExit(
            "healpy is required to generate maps. Install project dependencies with "
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


def arcmin_to_rad(value: float) -> float:
    return float(np.deg2rad(value / 60.0))


def next_power_of_two(value: float) -> int:
    if value <= 1.0:
        return 1
    return int(2 ** np.ceil(np.log2(value)))


def default_manifest_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_manifest.json")


def default_lmax(nside_generate: int) -> int:
    return min(1000, 3 * nside_generate - 1)


def recommend_beamed_synfast_resolution(
    *,
    fwhm_arcmin: float,
    nside_out: int,
    pixels_per_fwhm: float,
    beam_power_cutoff: float,
) -> dict[str, Any]:
    if fwhm_arcmin <= 0.0:
        raise ValueError("fwhm-arcmin must be positive")
    if pixels_per_fwhm <= 0.0:
        raise ValueError("pixels-per-fwhm must be positive")
    if not (0.0 < beam_power_cutoff < 1.0):
        raise ValueError("beam-power-cutoff must be between 0 and 1")

    theta_fwhm = arcmin_to_rad(fwhm_arcmin)
    sigma_b = theta_fwhm / np.sqrt(8.0 * np.log(2.0))
    lmax_out = 3 * nside_out - 1

    l_beam = int(
        np.ceil(
            (np.sqrt(1.0 + 4.0 * np.log(1.0 / beam_power_cutoff) / sigma_b**2) - 1.0)
            / 2.0
        )
    )
    lmax_gen = max(lmax_out, l_beam)

    theta_pix_factor = np.sqrt(np.pi / 3.0)
    nside_from_beam_sampling = pixels_per_fwhm * theta_pix_factor / theta_fwhm
    nside_from_lmax = (lmax_gen + 1.0) / 3.0
    nside_generate = next_power_of_two(
        max(
            float(nside_out),
            nside_from_beam_sampling,
            nside_from_lmax,
        )
    )
    lmax_gen = min(lmax_gen, 3 * nside_generate - 1)

    return {
        "nside_generate": int(nside_generate),
        "lmax": int(lmax_gen),
        "lmax_camb": int(lmax_gen),
        "nside_out": int(nside_out),
        "lmax_out": int(lmax_out),
        "l_beam": int(l_beam),
        "pixels_per_fwhm": float(pixels_per_fwhm),
        "beam_power_cutoff": float(beam_power_cutoff),
        "theta_pix_generate_arcmin": float(np.rad2deg(theta_pix_factor / nside_generate) * 60.0),
        "theta_pix_out_arcmin": float(np.rad2deg(theta_pix_factor / nside_out) * 60.0),
        "nside_from_beam_sampling": float(nside_from_beam_sampling),
        "nside_from_lmax": float(nside_from_lmax),
    }


def resolve_resolution_settings(args: argparse.Namespace) -> dict[str, Any]:
    recommendation = recommend_beamed_synfast_resolution(
        fwhm_arcmin=args.fwhm_arcmin,
        nside_out=args.nside_out,
        pixels_per_fwhm=args.pixels_per_fwhm,
        beam_power_cutoff=args.beam_power_cutoff,
    )

    if args.resolution_mode == "auto":
        nside_generate = (
            int(args.nside_generate)
            if args.nside_generate is not None
            else int(recommendation["nside_generate"])
        )
        lmax = int(args.lmax) if args.lmax is not None else int(recommendation["lmax"])
    elif args.resolution_mode == "manual":
        nside_generate = int(args.nside_generate) if args.nside_generate is not None else 2048
        lmax = int(args.lmax) if args.lmax is not None else default_lmax(nside_generate)
    else:
        raise ValueError(f"Unknown resolution mode: {args.resolution_mode}")

    return {
        "mode": args.resolution_mode,
        "nside_generate": int(nside_generate),
        "lmax": int(lmax),
        "lmax_camb": int(lmax),
        "auto_recommendation": recommendation,
        "manual_overrides": {
            "nside_generate": args.nside_generate is not None,
            "lmax": args.lmax is not None,
        },
    }


def parse_dtype(name: str) -> np.dtype:
    dtype = np.dtype(name)
    if dtype.kind != "f":
        raise argparse.ArgumentTypeError("dtype must be a floating-point type")
    return dtype


def validate_nside(value: int, label: str) -> None:
    hp = import_healpy()
    if not hp.isnsideok(value):
        raise ValueError(f"{label} must be a valid HEALPix nside, got {value}")


def draw_lcdm_parameters(rng: np.random.Generator) -> dict[str, float]:
    params = {
        name: float(rng.normal(spec["mean"], spec["sigma"]))
        for name, spec in PLANCK_PARAMETER_PRIORS.items()
    }
    params.update(FIXED_PARAMETERS)
    return params


def fixed_lcdm_parameters() -> dict[str, float]:
    params = {
        name: float(spec["mean"])
        for name, spec in PLANCK_PARAMETER_PRIORS.items()
    }
    params.update(FIXED_PARAMETERS)
    return params


def choose_lcdm_parameters(
    rng: np.random.Generator,
    cosmology_mode: str,
) -> dict[str, float]:
    if cosmology_mode == "sampled":
        return draw_lcdm_parameters(rng)
    if cosmology_mode == "fixed":
        return fixed_lcdm_parameters()
    raise ValueError(f"Unknown cosmology mode: {cosmology_mode}")


def generate_lensed_tt_dl(
    params: dict[str, float],
    *,
    lmax: int,
    accuracy_boost: float,
) -> np.ndarray:
    camb = import_camb()

    pars = camb.set_params(
        H0=params["H0"],
        ombh2=params["ombh2"],
        omch2=params["omch2"],
        mnu=params["mnu"],
        omk=params["omk"],
        tau=params["tau"],
        As=params["As"],
        ns=params["ns"],
        halofit_version="mead",
        lmax=lmax,
        AccuracyBoost=accuracy_boost,
    )
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")
    dl = np.asarray(powers["total"][: lmax + 1, 0], dtype=np.float64)
    if dl.shape[0] != lmax + 1:
        raise RuntimeError(f"CAMB returned {dl.shape[0]} TT values for lmax={lmax}")
    dl[:2] = 0.0
    return dl


def dl_to_cl(dl: np.ndarray) -> np.ndarray:
    ell = np.arange(dl.size, dtype=np.float64)
    factor = np.ones_like(ell)
    factor[2:] = ell[2:] * (ell[2:] + 1.0) / (2.0 * np.pi)
    cl = np.asarray(dl, dtype=np.float64) / factor
    cl[:2] = 0.0
    return cl


def open_temp_memmap(
    output_path: Path,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype,
    overwrite: bool,
) -> tuple[Path, np.memmap]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f"{output_path.stem}.partial.npy")

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )
    if temp_path.exists() and not overwrite:
        raise FileExistsError(
            f"Temporary output already exists: {temp_path}. Use --overwrite to replace it."
        )
    if temp_path.exists():
        temp_path.unlink()

    array = np.lib.format.open_memmap(temp_path, mode="w+", dtype=dtype, shape=shape)
    return temp_path, array


def simulate_temperature_map(
    cl: np.ndarray,
    *,
    nside_generate: int,
    nside_out: int,
    lmax: int,
    fwhm_rad: float,
    map_seed: int,
    dtype: np.dtype,
) -> np.ndarray:
    hp = import_healpy()

    # healpy.synfast uses NumPy's legacy global RNG internally.  Seeding per map
    # makes the stochastic sky realization reproducible while leaving parameter
    # draws under the Generator above.
    np.random.seed(int(map_seed))
    generated = hp.synfast(cl, nside=nside_generate, lmax=lmax, fwhm=fwhm_rad, new=True)

    if nside_generate != nside_out:
        generated = hp.ud_grade(generated, nside_out)

    return np.asarray(generated, dtype=dtype)


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(manifest), handle, indent=2)


def generate_maps(args: argparse.Namespace) -> dict[str, Any]:
    hp = import_healpy()

    validate_nside(args.nside_out, "nside-out")
    resolution = resolve_resolution_settings(args)
    nside_generate = resolution["nside_generate"]
    lmax = resolution["lmax"]
    lmax_camb = resolution["lmax_camb"]

    validate_nside(nside_generate, "nside-generate")
    if nside_generate < args.nside_out:
        raise ValueError("nside-generate must be greater than or equal to nside-out")
    if args.n_maps <= 0:
        raise ValueError("n-maps must be positive")
    if lmax <= 1:
        raise ValueError("lmax must be greater than 1")
    if lmax > 3 * nside_generate - 1:
        raise ValueError(
            f"lmax={lmax} is too high for nside-generate={nside_generate}; "
            f"use lmax <= {3 * nside_generate - 1}"
        )

    dtype = parse_dtype(args.dtype)
    output_path = args.output.resolve()
    manifest_path = (args.manifest or default_manifest_path(output_path)).resolve()
    fwhm_rad = arcmin_to_rad(args.fwhm_arcmin)
    npix_out = hp.nside2npix(args.nside_out)
    rng = np.random.default_rng(args.seed)

    temp_maps_path, maps = open_temp_memmap(
        output_path,
        shape=(args.n_maps, npix_out),
        dtype=dtype,
        overwrite=args.overwrite,
    )

    spectra = None
    temp_spectra_path = None
    if args.spectra_output is not None:
        temp_spectra_path, spectra = open_temp_memmap(
            args.spectra_output.resolve(),
            shape=(args.n_maps, lmax + 1),
            dtype=np.dtype("float32"),
            overwrite=args.overwrite,
        )

    parameter_draws: list[dict[str, Any]] = []
    description_by_mode = {
        "sampled": (
            "LCDM CMB temperature simulations generated from CAMB spectra with "
            "independently sampled Gaussian Planck parameter uncertainties."
        ),
        "fixed": (
            "LCDM CMB temperature simulations generated from CAMB spectra with "
            "fixed Planck mean cosmological parameters."
        ),
    }

    manifest = {
        "description": description_by_mode[args.cosmology_mode],
        "output_path": str(output_path),
        "manifest_path": str(manifest_path),
        "n_maps": int(args.n_maps),
        "nside_out": int(args.nside_out),
        "nside_generate": int(nside_generate),
        "npix_out": int(npix_out),
        "lmax": int(lmax),
        "lmax_camb": int(lmax_camb),
        "fwhm_arcmin": float(args.fwhm_arcmin),
        "fwhm_rad": float(fwhm_rad),
        "accuracy_boost": float(args.accuracy_boost),
        "resolution": resolution,
        "dtype": str(dtype),
        "seed": int(args.seed),
        "cosmology_mode": str(args.cosmology_mode),
        "parameter_priors": PLANCK_PARAMETER_PRIORS,
        "parameter_means": fixed_lcdm_parameters(),
        "fixed_parameters": FIXED_PARAMETERS,
        "spectra_output": str(args.spectra_output.resolve()) if args.spectra_output else None,
        "completed_maps": 0,
        "parameter_draws": parameter_draws,
    }

    try:
        print(
            "Using "
            f"cosmology_mode={args.cosmology_mode}, "
            f"resolution_mode={resolution['mode']}, "
            f"nside_generate={nside_generate}, "
            f"nside_out={args.nside_out}, "
            f"lmax={lmax}"
        )
        fixed_params = None
        fixed_dl = None
        fixed_cl = None
        if args.cosmology_mode == "fixed":
            fixed_params = choose_lcdm_parameters(rng, args.cosmology_mode)
            fixed_dl = generate_lensed_tt_dl(
                fixed_params,
                lmax=lmax_camb,
                accuracy_boost=args.accuracy_boost,
            )
            fixed_cl = dl_to_cl(fixed_dl)

        for map_index in range(args.n_maps):
            if args.cosmology_mode == "fixed":
                params = dict(fixed_params)
                dl = fixed_dl
                cl = fixed_cl
            else:
                params = choose_lcdm_parameters(rng, args.cosmology_mode)
                dl = generate_lensed_tt_dl(
                    params,
                    lmax=lmax_camb,
                    accuracy_boost=args.accuracy_boost,
                )
                cl = dl_to_cl(dl)

            map_seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
            sim_map = simulate_temperature_map(
                cl,
                nside_generate=nside_generate,
                nside_out=args.nside_out,
                lmax=lmax,
                fwhm_rad=fwhm_rad,
                map_seed=map_seed,
                dtype=dtype,
            )

            maps[map_index, :] = sim_map
            if spectra is not None:
                spectra[map_index, :] = dl.astype(np.float32)

            record = dict(params)
            record["map_index"] = int(map_index)
            record["map_seed"] = int(map_seed)
            record["cosmology_mode"] = str(args.cosmology_mode)
            parameter_draws.append(record)
            manifest["completed_maps"] = int(map_index + 1)

            if (
                (map_index + 1) == 1
                or (map_index + 1) % args.progress_every == 0
                or (map_index + 1) == args.n_maps
            ):
                print(f"Generated {map_index + 1}/{args.n_maps} maps")

        maps.flush()
        del maps
        os.replace(temp_maps_path, output_path)

        if spectra is not None:
            spectra.flush()
            del spectra
            os.replace(temp_spectra_path, args.spectra_output.resolve())

        write_manifest(manifest_path, manifest)
        return manifest
    except Exception:
        write_manifest(manifest_path, manifest)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Planck-parameter LCDM CMB simulation maps for anomaly-agent.",
    )
    parser.add_argument(
        "--n-maps",
        type=int,
        default=1000,
        help="Number of simulated maps to generate. Defaults to 1000.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/input/CMBmapsPlanckLCDM256.npy"),
        help="Output .npy stack. Defaults to data/input/CMBmapsPlanckLCDM256.npy.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest JSON path. Defaults to <output_stem>_manifest.json next to the output.",
    )
    parser.add_argument(
        "--nside-out",
        type=int,
        default=256,
        help="Output HEALPix nside. Defaults to 256.",
    )
    parser.add_argument(
        "--resolution-mode",
        choices=["auto", "manual"],
        default="auto",
        help=(
            "Resolution selection mode. In auto mode, nside-generate and lmax are "
            "derived from the beam and nside-out unless explicitly overridden. "
            "Defaults to auto."
        ),
    )
    parser.add_argument(
        "--nside-generate",
        type=int,
        default=None,
        help=(
            "Generation HEALPix nside before optional downgrade. In auto mode, "
            "defaults to the beam-derived recommendation; in manual mode, "
            "defaults to 2048."
        ),
    )
    parser.add_argument(
        "--lmax",
        type=int,
        default=None,
        help=(
            "Maximum multipole. In auto mode, defaults to the beam-derived "
            "recommendation; in manual mode, defaults to min(1000, 3*nside_generate - 1)."
        ),
    )
    parser.add_argument(
        "--fwhm-arcmin",
        type=float,
        default=5.0,
        help="Gaussian beam FWHM in arcmin passed to healpy.synfast. Defaults to 5.",
    )
    parser.add_argument(
        "--pixels-per-fwhm",
        type=float,
        default=2.5,
        help="Auto-mode beam sampling requirement. Defaults to 2.5.",
    )
    parser.add_argument(
        "--beam-power-cutoff",
        "--eps-power",
        dest="beam_power_cutoff",
        type=float,
        default=1.0e-3,
        help="Auto-mode B_l^2 cutoff used to choose lmax. Defaults to 1e-3.",
    )
    parser.add_argument(
        "--accuracy-boost",
        type=float,
        default=1.5,
        help="CAMB AccuracyBoost value. Defaults to 1.5.",
    )
    parser.add_argument(
        "--cosmology-mode",
        choices=["sampled", "fixed"],
        default="sampled",
        help=(
            "Cosmological parameter mode. Use sampled for independent Gaussian "
            "Planck draws, or fixed for Planck mean parameters. Defaults to sampled."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for sampled parameter draws and per-map sky seeds.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        help="Output map dtype. Must be a floating-point dtype. Defaults to float32.",
    )
    parser.add_argument(
        "--spectra-output",
        type=Path,
        default=None,
        help="Optional output .npy file for input CAMB D_l spectra.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N maps. Defaults to 10.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output and temporary partial files.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.progress_every <= 0:
        raise ValueError("progress-every must be positive")

    manifest = generate_maps(args)
    print(f"Wrote maps: {manifest['output_path']}")
    print(f"Wrote manifest: {manifest['manifest_path']}")


if __name__ == "__main__":
    main()
