import builtins
from types import SimpleNamespace

import healpy as hp
import numpy as np
import pytest
import yaml
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate

import anomaly_agent
import file_paths
from scan_agent import (
    SCAN_FRAMEWORK_MARKER,
    ScanAnomalyAgent,
    audit_scan_code,
    build_scan_analysis_code,
    parse_scan_proposal,
    scan_run_config,
    validate_scan_spec,
)


def template_variables(template: str) -> set:
    return set(PromptTemplate.from_template(template).input_variables)


def test_scan_planner_prompt_matches_planner_node_variables():
    with open(file_paths.scan_planner_dir) as stream:
        template = yaml.safe_load(stream)["template"]

    assert template_variables(template) == {
        "search_instruction",
        "tested",
        "prior_catalog_text",
        "rejected_proposals_text",
        "rotation_guidance",
        "search_count",
        "search_results",
        "planner_feedback",
    }


def test_scan_implement_prompt_matches_implement_node_variables():
    with open(file_paths.scan_implement_dir) as stream:
        file = yaml.safe_load(stream)

    assert template_variables(file["template"]) == {
        "test_name",
        "test_description",
        "scan_spec",
        "previous_code",
        "guidance",
    }
    assert template_variables(file["additional_template"]) == {"previous_error"}


def test_scan_review_prompt_matches_review_variables():
    with open(file_paths.scan_review_dir) as stream:
        template = yaml.safe_load(stream)["template"]

    assert template_variables(template) == {
        "scan_spec",
        "test_description",
        "analysis_code",
    }


def test_scan_run_config_marks_scan_mode():
    runtime_configs = {
        "agent": {"model": "test-model"},
        "test": {"tests_to_run": 1},
        "plot": {},
        "paths": {},
    }
    config = scan_run_config(
        runtime_configs,
        model="test-model",
        thread_id="scan_test",
        base_url="http://localhost",
        reasoning_effort=None,
        sim_maps_path=None,
    )

    assert config["agent"]["mode"] == "scan"
    assert "a_posteriori_parameters" in config["scan"]
    assert config["test"]["scan_runtime_safety_factor"] == 1.25
    assert config["test"]["scan_min_span_factor"] == 3.0
    assert config["test"]["scan_min_relative_width"] == 0.5
    assert config["test"]["scan_max_static_grid_points"] == 100_000


def valid_scan_spec(reduction="minimum"):
    return {
        "reduction": reduction,
        "parameters": [
            {
                "name": "position",
                "role": "position",
                "treatment": "scanned",
                "domain": "full_unmasked_sky",
                "grid": {"type": "all_unmasked_pixels"},
            },
            {
                "name": "smoothing_scale",
                "role": "scale",
                "treatment": "scanned",
                "domain": "2_to_64_degrees",
                "grid": {
                    "type": "log",
                    "minimum": 2.0,
                    "maximum": 64.0,
                    "count": 6,
                },
            },
            {
                "name": "harmonic_limit",
                "role": "resolution_limit",
                "treatment": "fixed",
                "value": "2*nside",
                "justification_type": "resolution_derived",
                "justification": "The bandwidth is derived only from map resolution.",
            },
        ],
    }


def valid_scan_description():
    return (
        "Evaluate a smoothed field over all usable positions and scales. "
        "PARAMETER ACCOUNTING: position scanned over the full unmasked sky; "
        "smoothing scale scanned from 2 to 64 degrees; harmonic limit fixed "
        "from the input resolution."
    )


def proposal_text(scan_spec=None, description=None):
    return (
        "TEST_NAME: Valid scan\n"
        "SCAN_SPEC:\n"
        + "\n".join(
            f"  {line}" if line else line
            for line in yaml.safe_dump(
                scan_spec or valid_scan_spec(), sort_keys=False
            ).splitlines()
        )
        + "\nDESCRIPTION: "
        + (description or valid_scan_description())
    )


def test_parse_and_validate_scan_proposal():
    test_name, description, scan_spec = parse_scan_proposal(proposal_text())

    assert test_name == "Valid scan"
    assert description == valid_scan_description()
    assert validate_scan_spec(scan_spec, description) == []


def test_scan_policy_rejects_fixed_scientific_parameter():
    scan_spec = valid_scan_spec()
    scan_spec["parameters"][0] = {
        "name": "known_cold_spot_center",
        "role": "position",
        "treatment": "fixed",
        "value": {"longitude_deg": 209, "latitude_deg": -57},
        "justification_type": "structural_definition",
        "justification": "This is the previously reported Cold Spot center.",
    }
    description = (
        "Measure the known feature. PARAMETER ACCOUNTING: known cold spot center "
        "fixed from the literature; smoothing scale scanned; harmonic limit fixed."
    )

    errors = validate_scan_spec(scan_spec, description)

    assert any("must be scanned, not fixed" in error for error in errors)


def test_scan_policy_rejects_narrow_scale_grid():
    scan_spec = valid_scan_spec()
    scan_spec["parameters"][1]["grid"]["maximum"] = 4.0

    errors = validate_scan_spec(scan_spec, valid_scan_description())

    assert any("spans a factor 2.00" in error for error in errors)


def test_scan_policy_rejects_tightly_clustered_threshold_grid():
    scan_spec = {
        "reduction": "maximum",
        "parameters": [
            {
                "name": "threshold",
                "role": "threshold",
                "treatment": "scanned",
                "domain": "dimensionless_threshold",
                "grid": {"type": "values", "values": [2.9, 3.0, 3.1]},
            }
        ],
    }
    description = "Statistic. PARAMETER ACCOUNTING: threshold scanned over its grid."

    errors = validate_scan_spec(scan_spec, description)

    assert any("relative width" in error for error in errors)


def test_scan_policy_requires_absolute_width_for_standardized_thresholds():
    scan_spec = {
        "reduction": "maximum",
        "parameters": [
            {
                "name": "threshold",
                "role": "threshold",
                "treatment": "scanned",
                "domain": "standardized_sigma_units",
                "grid": {"type": "values", "values": [-0.5, 0.0, 0.5]},
            }
        ],
    }
    description = "Statistic. PARAMETER ACCOUNTING: threshold scanned over its grid."

    errors = validate_scan_spec(scan_spec, description)

    assert any("Standardized grid" in error for error in errors)


def test_scan_policy_caps_static_cartesian_grid_size():
    scan_spec = {
        "reduction": "maximum",
        "parameters": [
            {
                "name": "direction",
                "role": "direction",
                "treatment": "scanned",
                "domain": "full_sphere",
                "grid": {"type": "healpix", "nside": 8},
            },
            {
                "name": "threshold",
                "role": "threshold",
                "treatment": "scanned",
                "domain": "dimensionless_threshold",
                "grid": {"type": "values", "values": list(range(200))},
            },
        ],
    }
    description = (
        "Statistic. PARAMETER ACCOUNTING: direction scanned over the full sphere; "
        "threshold scanned over its grid."
    )

    errors = validate_scan_spec(scan_spec, description)

    assert any("Static Cartesian scan grid has 153600 values" in error for error in errors)


def test_scan_policy_caps_projected_position_grid_size():
    scan_spec = valid_scan_spec()
    scan_spec["parameters"].insert(
        2,
        {
            "name": "threshold",
            "role": "threshold",
            "treatment": "scanned",
            "domain": "standardized_sigma_units",
            "grid": {"type": "values", "values": [-1.0, 0.0, 1.0]},
        },
    )
    description = (
        "Statistic. PARAMETER ACCOUNTING: position scanned over the full sky; "
        "smoothing scale scanned broadly; threshold scanned in sigma units; "
        "harmonic limit fixed from resolution."
    )

    errors = validate_scan_spec(scan_spec, description)

    assert any("Projected scan grid has 14155776 values" in error for error in errors)


def test_planner_retries_when_scan_manifest_is_missing(tmp_path):
    class FakePlanner:
        def invoke(self, prompt):
            return AIMessage(
                content=(
                    "TEST_NAME: Fixed known feature\n"
                    "DESCRIPTION: Measure a 10 degree disc at the known Cold Spot. "
                    "PARAMETER ACCOUNTING: center fixed."
                )
            )

    agent = ScanAnomalyAgent.__new__(ScanAnomalyAgent)
    agent.allow_search_tools = False
    agent.planner_prompt_path = file_paths.scan_planner_dir
    agent.test_output_dir = tmp_path
    agent.test_config = {
        "max_searches_per_test": 0,
        "family_soft_cap": 7,
        "family_hard_cap": 9,
    }
    agent.prompt_llm = lambda **kwargs: FakePlanner()

    result = agent.planner_node(
        {"tested_anomalies": [], "search_count": 0, "messages": []}
    )

    assert result["node_retry"] is True
    assert "Missing SCAN_SPEC" in result["messages"][-1].content


def test_scan_code_audit_blocks_scalar_override_and_external_access():
    code = """
import os

def evaluate_scan(m):
    noise = np.random.normal(size=m.size)
    return np.load("known_planck_values.npy")

def analyze_map(m):
    return -1.0

test_description = "bad"
"""

    errors = audit_scan_code(code)

    assert any("Do not define analyze_map" in error for error in errors)
    assert any("Imports are not allowed" in error for error in errors)
    assert any("External data access" in error for error in errors)
    assert any("Randomness is not allowed" in error for error in errors)


def test_scan_execution_environment_has_restricted_builtins(tmp_path):
    agent = make_minimal_scan_agent(tmp_path)

    environment = agent.code_execution_environment()

    assert "open" not in environment["__builtins__"]
    assert "__import__" in environment["__builtins__"]
    assert environment["__builtins__"]["__import__"] is not builtins.__import__
    assert environment["__builtins__"]["__import__"]("numpy") is np
    with pytest.raises(ImportError, match="may not import module 'os'"):
        environment["__builtins__"]["__import__"]("os")
    assert environment["np"] is np
    assert environment["hp"] is hp


@pytest.mark.parametrize(
    ("reduction", "expected"),
    [
        ("minimum", -4.0),
        ("maximum", 3.0),
        ("maximum_absolute", 4.0),
    ],
)
def test_framework_owns_scan_reduction(reduction, expected):
    code = """
def evaluate_scan(m):
    return np.asarray([1.0, -4.0, 3.0, np.nan])

test_description = "framework reduction test"
"""
    scan_spec = {
        "reduction": reduction,
        "parameters": [
            {
                "name": "threshold",
                "role": "threshold",
                "treatment": "scanned",
                "domain": "standardized_amplitude",
                "grid": {"type": "values", "values": [-4.0, 1.0, 3.0, 5.0]},
            }
        ],
    }
    wrapped_code = build_scan_analysis_code(code, scan_spec)
    namespace = {"np": np}

    exec(wrapped_code, namespace, namespace)

    assert namespace["analyze_map"](np.ones(12)) == expected
    assert SCAN_FRAMEWORK_MARKER in wrapped_code


def test_framework_rejects_wrong_grid_size():
    code = """
def evaluate_scan(m):
    return np.asarray([1.0, 2.0])

test_description = "wrong size"
"""
    scan_spec = {
        "reduction": "maximum",
        "parameters": [
            {
                "name": "threshold",
                "role": "threshold",
                "treatment": "scanned",
                "domain": "standardized_amplitude",
                "grid": {"type": "values", "values": [-1.0, 0.0, 1.0]},
            }
        ],
    }
    namespace = {"np": np}
    exec(build_scan_analysis_code(code, scan_spec), namespace, namespace)

    with pytest.raises(ValueError, match="SCAN_SPEC requires 3"):
        namespace["analyze_map"](np.ones(12))


def make_minimal_scan_agent(tmp_path):
    agent = ScanAnomalyAgent.__new__(ScanAnomalyAgent)
    agent.agent_mode = "scan"
    agent.test_output_dir = tmp_path
    agent.test_config = {"max_test_minutes": 1, "mask_threshold": 0.9, "plot_bins": 11}
    agent.plot_config = {}
    agent.python_env = {
        "analyze_map": None,
        "summarize_results": None,
        "test_description": None,
        "last_error": None,
        "last_sigma": None,
        "last_result": None,
        "scan_spec": None,
        "scan_review": None,
    }
    agent.observed_map_label = "Planck map"
    agent.observed_statistic_label = "Observed Planck statistic"
    agent.show_simulation_sources = True
    return agent


def test_semantic_review_fails_closed_on_invalid_verdict(tmp_path):
    class InvalidReviewer:
        def invoke(self, prompt):
            return AIMessage(content="Everything seems fine.")

    agent = make_minimal_scan_agent(tmp_path)
    agent.scan_review_prompt_path = file_paths.scan_review_dir
    agent.llm = InvalidReviewer()

    review = agent.review_scan_implementation(
        scan_spec=valid_scan_spec(),
        test_description=valid_scan_description(),
        code=(
            "def evaluate_scan(m):\n"
            "    return m[np.isfinite(m)]\n\n"
            'test_description = "scan"\n'
        ),
    )

    assert review["accepted"] is False
    assert review["verdict"] == "invalid"


def registration_runtime(scan_spec, description, code):
    message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "register_analysis",
                "args": {"code": code},
                "id": "scan-register-1",
                "type": "tool_call",
            }
        ],
    )
    return SimpleNamespace(
        state={
            "messages": [message],
            "scan_spec": scan_spec,
            "current_test_description": description,
        }
    )


def test_registration_reviews_wraps_and_preflights_scan_code(tmp_path, monkeypatch):
    agent = make_minimal_scan_agent(tmp_path)
    planck_map = np.full(12, 5.0)
    mask = np.ones(12, dtype=bool)
    sim_maps = [
        (f"sim[{index}]", np.full(12, float(index + 1))) for index in range(4)
    ]
    monkeypatch.setattr(agent, "prepare_planck_data", lambda target_nside: (planck_map, mask))
    monkeypatch.setattr(agent, "iter_simulation_maps", lambda: iter(sim_maps))
    monkeypatch.setattr(agent, "simulation_map_count", lambda: len(sim_maps))
    warmup_maps = []
    original_warmup = agent.warm_up_registration_probe
    monkeypatch.setattr(
        agent,
        "warm_up_registration_probe",
        lambda analyze_fn, sample_map, sample_mask: (
            warmup_maps.append(np.asarray(sample_map).copy()),
            original_warmup(analyze_fn, sample_map, sample_mask),
        )[-1],
    )
    monkeypatch.setattr(
        agent,
        "review_scan_implementation",
        lambda **kwargs: {
            "accepted": True,
            "verdict": "accept",
            "reason": "The implementation matches the manifest.",
            "revision_guidance": "None",
        },
    )
    code = """
def evaluate_scan(m):
    values = m[np.isfinite(m)]
    return np.concatenate([values for _ in range(6)])

test_description = "registered scan"
"""
    scan_spec = valid_scan_spec()

    command = agent.register(
        code,
        registration_runtime(scan_spec, valid_scan_description(), code),
    )

    assert command.update["node_retry"] is False
    assert agent.python_env["last_error"] is None
    assert command.update["scan_spec"] == scan_spec
    assert command.update["scan_review"]["accepted"] is True
    assert command.update["scan_review"]["policy_version"] == 1
    assert len(command.update["scan_review"]["registered_code_sha256"]) == 64
    assert SCAN_FRAMEWORK_MARKER in command.update["code"][-1].content
    assert agent.python_env["analyze_map"](planck_map) == 5.0
    assert len(warmup_maps) == 1
    np.testing.assert_allclose(warmup_maps[0], sim_maps[0][1])


def test_probe_timeout_becomes_actionable_fix_cycle_feedback(tmp_path, monkeypatch):
    agent = make_minimal_scan_agent(tmp_path)
    agent._scan_preflight_probe_limit_seconds = 0.432
    agent._scan_preflight_simulation_count = 10_000
    monkeypatch.setattr(
        agent,
        "review_scan_implementation",
        lambda **kwargs: {
            "accepted": True,
            "verdict": "accept",
            "reason": "Matches.",
            "revision_guidance": "None",
        },
    )

    def timeout_register(self, code, runtime):
        raise RuntimeError("Restarting test: exceeded 0.432 seconds.")

    monkeypatch.setattr(anomaly_agent.AnomalyAgent, "register", timeout_register)
    code = """
def evaluate_scan(m):
    values = m[np.isfinite(m)]
    return np.concatenate([values for _ in range(6)])

test_description = "registered scan"
"""

    command = agent.register(
        code,
        registration_runtime(valid_scan_spec(), valid_scan_description(), code),
    )
    feedback = command.update["messages"][-1].content

    assert command.update["node_retry"] is True
    assert feedback.startswith("SCAN RUNTIME REJECTED")
    assert "warmed per-map timing probe exceeded 0.432 seconds" in feedback
    assert "Vectorize evaluate_scan" in feedback
    assert "Restarting test" not in feedback


def test_scan_preflight_budget_uses_actual_stack_size(tmp_path, monkeypatch):
    agent = make_minimal_scan_agent(tmp_path)
    agent.test_config = {
        "max_test_minutes": 90,
        "scan_runtime_safety_factor": 1.25,
    }
    monkeypatch.setattr(agent, "simulation_map_count", lambda: 10_000)

    limit_minutes = agent.preflight_probe_max_minutes()

    assert limit_minutes == pytest.approx(90 / (10_001 * 1.25))
    assert limit_minutes * 60 < 0.432


def inject_spot(m, nside, lon_deg, lat_deg, sigma_deg, amplitude):
    npix = hp.nside2npix(nside)
    vec = hp.ang2vec(lon_deg, lat_deg, lonlat=True)
    pixvec = np.asarray(hp.pix2vec(nside, np.arange(npix)))
    ang = np.arccos(np.clip(vec @ pixvec, -1.0, 1.0))
    return m + amplitude * np.exp(-0.5 * (ang / np.radians(sigma_deg)) ** 2)


def test_framework_wrapped_harmonic_position_scan_runs_end_to_end(tmp_path, monkeypatch):
    agent = make_minimal_scan_agent(tmp_path)

    nside = 16
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(5)
    mask = np.ones(npix, dtype=bool)
    planck_map = inject_spot(
        rng.standard_normal(npix), nside, 90.0, 30.0, sigma_deg=10.0, amplitude=-6.0
    )
    sim_maps = [(f"sim[{i}]", rng.standard_normal(npix)) for i in range(4)]

    monkeypatch.setattr(agent, "prepare_planck_data", lambda target_nside: (planck_map, mask))
    monkeypatch.setattr(agent, "iter_simulation_maps", lambda: iter(sim_maps))
    monkeypatch.setattr(
        anomaly_agent,
        "plot_results",
        lambda planck_stat, sim_results, output_dir, summary=None, plot_config=None, test_config=None: {
            "png": str(output_dir / "statistic_figure.png"),
            "pdf": str(output_dir / "statistic_figure.pdf"),
            "kind": "histogram",
        },
    )

    implementation_code = """
def evaluate_scan(m):
    nside = hp.get_nside(m)
    lmax = 2 * nside
    mask = np.isfinite(m)
    alm = hp.map2alm(np.where(mask, m, 0.0), lmax=lmax, iter=0)
    scan_values = []
    for fwhm_deg in [8.0, 16.0, 32.0]:
        bl = hp.gauss_beam(np.radians(fwhm_deg), lmax=lmax)
        filtered = hp.alm2map(hp.almxfl(alm, bl), nside)
        values = filtered[mask]
        values = (values - values.mean()) / values.std()
        scan_values.append(values)
    return np.concatenate(scan_values)

test_description = "Most extreme cold smoothed value over a position and scale scan."
"""
    scan_spec = valid_scan_spec()
    scan_spec["parameters"][1]["grid"] = {
        "type": "values",
        "values": [8.0, 16.0, 32.0],
    }
    scan_spec["parameters"][1]["domain"] = "8_to_32_degrees"
    scan_description = (
        "Evaluate a smoothed field over all usable positions and three scales. "
        "PARAMETER ACCOUNTING: position scanned over the full unmasked sky; "
        "smoothing scale scanned over 8, 16, and 32 degrees; harmonic limit "
        "fixed from the input resolution."
    )
    code = build_scan_analysis_code(implementation_code, scan_spec)
    state = {
        "code": [AIMessage(content=code)],
        "current_test_name": "Scanned cold feature",
        "current_test_description": scan_description,
        "test_hypothesis": [AIMessage(content="The observed minimum is unusually low.")],
        "test_type": [AIMessage(content="one-tailed lower")],
        "justification": [AIMessage(content="Lower tail empirical comparison.")],
        "tested_anomalies": [],
        "scan_spec": scan_spec,
        "scan_review": {
            "accepted": True,
            "verdict": "accept",
            "reason": "Matches the scan specification.",
            "revision_guidance": "None",
        },
    }

    output = agent.run_registered_analysis(state)
    result = agent.python_env["last_result"]

    assert output.startswith("CODE OUTPUT")
    assert agent.python_env["last_error"] is None
    assert result["n_sims"] == 4
    assert result["scan_reduction_control"] == "framework"
    assert (tmp_path / "Test_01_scanned_cold_feature" / "scan_manifest.yaml").exists()
    assert (tmp_path / "Test_01_scanned_cold_feature" / "scan_policy_review.json").exists()
    sim_stats = np.load(tmp_path / "Test_01_scanned_cold_feature" / "simulation_statistics.npy")
    assert result["planck_stat"] < sim_stats.min()
