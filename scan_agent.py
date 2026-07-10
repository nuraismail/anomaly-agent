"""Scan-enforced CMB anomaly agent.

This module reuses the exploratory AnomalyAgent workflow but requires that
every assumed parameter of a proposed statistic — sky positions, directions,
angular scales, multipole ranges, thresholds, region shapes, and so on — is
either scanned over a broad a priori grid or fixed by an a priori structural
justification, never taken from prior knowledge of the observed sky. The
single scalar statistic is the extremum over the scan grid, so the
look-elsewhere cost of the parameter selection is paid inside the test and
identically on every simulation. A structured manifest is validated before
implementation, generated code is audited before execution, a fail-closed
semantic review checks code/spec agreement, and a trusted framework wrapper
owns the final extremum reduction. Registration uses the actual simulation
count when rejecting implementations too slow for the configured stack.
These controls are policy guardrails for model-generated code, not a security
boundary against deliberately adversarial Python.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import hashlib
import json
import re
from glob import glob
from pathlib import Path

import file_paths
import healpy as hp
import numpy as np
import yaml
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import PromptTemplate
from langgraph.types import Command

from anomaly_agent import (
    AnomalyAgent,
    effective_run_config,
    load_runtime_configs,
    normalize_optional_config_value,
)
from utils.string_utils import message_content_to_text, parse_test_metadata, text_to_dict


SCAN_FRAMEWORK_MARKER = "# --- scan framework: generated, do not edit ---"
SCAN_POLICY_VERSION = 1
SCAN_REDUCTIONS = {"minimum", "maximum", "maximum_absolute"}
SCIENTIFIC_PARAMETER_ROLES = {
    "position",
    "direction",
    "axis",
    "orientation",
    "scale",
    "multipole",
    "threshold",
    "region_shape",
    "region_size",
    "weight",
    "amplitude",
    "other_scientific",
}
STRUCTURAL_PARAMETER_ROLES = {
    "resolution_limit",
    "input_mask",
    "coordinate_system",
    "numerical_method",
    "normalization",
    "statistic_form",
    "symmetry",
}
FIXED_JUSTIFICATION_TYPES = {
    "resolution_derived",
    "input_defined",
    "coordinate_convention",
    "numerical_stability",
    "normalization_convention",
    "structural_definition",
    "symmetry_definition",
}
GRID_TYPES = {
    "linear",
    "log",
    "integer",
    "values",
    "healpix",
    "all_unmasked_pixels",
}


def _is_plain_scan_data(value, seen: set[int] | None = None) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return bool(np.isfinite(value))
    if not isinstance(value, (list, dict)):
        return False

    seen = set() if seen is None else seen
    value_id = id(value)
    if value_id in seen:
        return False
    seen.add(value_id)
    if isinstance(value, list):
        valid = all(_is_plain_scan_data(item, seen) for item in value)
    else:
        valid = all(
            isinstance(key, str) and _is_plain_scan_data(item, seen)
            for key, item in value.items()
        )
    seen.remove(value_id)
    return valid


def parse_scan_proposal(text: str) -> tuple[str, str, dict]:
    """Parse the planner's human description and machine-readable scan manifest."""
    if re.search(r"(?im)^[ \t]*TEST_NAME[ \t]*:[ \t]*\S", text) is None:
        raise ValueError("Missing non-empty TEST_NAME field.")
    match = re.search(
        r"(?ims)^[ \t]*SCAN_SPEC[ \t]*:[ \t]*\n(?P<spec>.*?)(?=^[ \t]*DESCRIPTION[ \t]*:)",
        text,
    )
    if match is None:
        raise ValueError("Missing SCAN_SPEC block before DESCRIPTION.")

    try:
        scan_spec = yaml.safe_load(match.group("spec"))
    except yaml.YAMLError as exc:
        raise ValueError(f"SCAN_SPEC is not valid YAML: {exc}") from exc

    if not isinstance(scan_spec, dict):
        raise ValueError("SCAN_SPEC must be a YAML mapping.")

    test_name, test_description = parse_test_metadata(text)
    if not test_description:
        raise ValueError("Missing non-empty DESCRIPTION field.")
    return test_name, test_description, scan_spec


def _grid_point_count(grid: dict) -> int | None:
    grid_type = str(grid.get("type", "")).strip().lower()
    if grid_type == "values":
        values = grid.get("values")
        return len(values) if isinstance(values, list) else None
    if grid_type in {"linear", "log", "integer"}:
        count = grid.get("count")
        return int(count) if isinstance(count, int) and not isinstance(count, bool) else None
    if grid_type == "healpix":
        nside = grid.get("nside")
        if isinstance(nside, int) and not isinstance(nside, bool) and nside > 0:
            return 12 * nside**2
    return None


def _numeric_grid_bounds(grid: dict) -> tuple[float, float] | None:
    grid_type = str(grid.get("type", "")).strip().lower()
    if grid_type == "values":
        values = grid.get("values")
        if not isinstance(values, list) or not values:
            return None
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            return None
        return float(min(values)), float(max(values))

    minimum = grid.get("minimum")
    maximum = grid.get("maximum")
    if all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in (minimum, maximum)
    ):
        return float(minimum), float(maximum)
    return None


def validate_scan_spec(
    scan_spec: dict,
    description: str,
    *,
    min_scan_points: int = 3,
    min_span_factor: float = 3.0,
    min_relative_width: float = 0.5,
    min_standardized_width: float = 2.0,
    min_direction_points: int = 48,
    max_direction_points: int = 768,
    max_static_grid_points: int = 100_000,
    position_grid_upper_bound: int = 786_432,
    max_total_grid_values: int = 5_000_000,
) -> list[str]:
    """Return deterministic policy errors for a proposed scan manifest."""
    errors: list[str] = []
    if not _is_plain_scan_data(scan_spec):
        return [
            "SCAN_SPEC may contain only finite numbers, strings, booleans, null, "
            "lists, and string-keyed mappings."
        ]
    if min_scan_points < 2:
        errors.append("scan_min_points must be at least 2.")
    if not np.isfinite(min_span_factor) or min_span_factor <= 1.0:
        errors.append("scan_min_span_factor must be greater than 1.")
    if not np.isfinite(min_relative_width) or not 0.0 < min_relative_width <= 2.0:
        errors.append("scan_min_relative_width must be finite and in (0, 2].")
    if not np.isfinite(min_standardized_width) or min_standardized_width <= 0.0:
        errors.append("scan_min_standardized_width must be finite and positive.")
    if min_direction_points < 2:
        errors.append("scan_min_direction_points must be at least 2.")
    if max_direction_points < min_direction_points:
        errors.append(
            "scan_max_direction_points must be at least scan_min_direction_points."
        )
    if max_static_grid_points < 2:
        errors.append("scan_max_static_grid_points must be at least 2.")
    if position_grid_upper_bound < 2:
        errors.append("scan_position_grid_upper_bound must be at least 2.")
    if max_total_grid_values < 2:
        errors.append("scan_max_total_grid_values must be at least 2.")
    reduction = str(scan_spec.get("reduction", "")).strip().lower()
    if reduction not in SCAN_REDUCTIONS:
        errors.append(
            "SCAN_SPEC.reduction must be minimum, maximum, or maximum_absolute."
        )

    parameters = scan_spec.get("parameters")
    if not isinstance(parameters, list) or not parameters:
        return errors + ["SCAN_SPEC.parameters must be a non-empty list."]

    known_roles = SCIENTIFIC_PARAMETER_ROLES | STRUCTURAL_PARAMETER_ROLES
    names: set[str] = set()
    scanned_count = 0
    full_map_grid_count = 0
    static_grid_count = 1
    accounting_match = re.search(r"(?is)PARAMETER\s+ACCOUNTING\s*:\s*(.*)$", description)
    accounting = accounting_match.group(1).lower() if accounting_match else ""
    if accounting_match is None:
        errors.append("DESCRIPTION must end with a PARAMETER ACCOUNTING section.")

    for index, parameter in enumerate(parameters, start=1):
        prefix = f"Parameter {index}"
        if not isinstance(parameter, dict):
            errors.append(f"{prefix} must be a mapping.")
            continue

        name = str(parameter.get("name", "")).strip()
        role = str(parameter.get("role", "")).strip().lower()
        treatment = str(parameter.get("treatment", "")).strip().lower()
        if not name:
            errors.append(f"{prefix} is missing a name.")
        else:
            normalized_name = re.sub(r"[_-]+", " ", name.lower())
            if normalized_name in names:
                errors.append(f"Parameter name '{name}' is duplicated.")
            names.add(normalized_name)
            if accounting and normalized_name not in re.sub(r"[_-]+", " ", accounting):
                errors.append(
                    f"Parameter '{name}' is missing from DESCRIPTION's parameter accounting."
                )
            if accounting and treatment and treatment not in accounting:
                errors.append(
                    f"Parameter accounting does not state that '{name}' is {treatment}."
                )

        if role not in known_roles:
            errors.append(
                f"{prefix} role '{role or '<missing>'}' is not an allowed parameter role."
            )

        if treatment == "fixed":
            if role in SCIENTIFIC_PARAMETER_ROLES:
                errors.append(
                    f"Scientific parameter '{name or index}' ({role}) must be scanned, not fixed."
                )
            if role not in STRUCTURAL_PARAMETER_ROLES:
                errors.append(
                    f"Fixed parameter '{name or index}' must use a structural role."
                )
            justification_type = str(
                parameter.get("justification_type", "")
            ).strip().lower()
            if justification_type not in FIXED_JUSTIFICATION_TYPES:
                errors.append(
                    f"Fixed parameter '{name or index}' has no allowed justification_type."
                )
            justification = str(parameter.get("justification", "")).strip()
            if len(justification) < 12:
                errors.append(
                    f"Fixed parameter '{name or index}' needs a substantive a priori justification."
                )
            if "value" not in parameter:
                errors.append(f"Fixed parameter '{name or index}' must declare its value.")
            continue

        if treatment != "scanned":
            errors.append(
                f"Parameter '{name or index}' treatment must be scanned or fixed."
            )
            continue

        scanned_count += 1
        domain = str(parameter.get("domain", "")).strip().lower()
        grid = parameter.get("grid")
        if not domain:
            errors.append(f"Scanned parameter '{name or index}' must declare its domain.")
        if not isinstance(grid, dict):
            errors.append(f"Scanned parameter '{name or index}' must declare a grid mapping.")
            continue

        grid_type = str(grid.get("type", "")).strip().lower()
        if grid_type not in GRID_TYPES:
            errors.append(
                f"Scanned parameter '{name or index}' has unsupported grid type "
                f"'{grid_type or '<missing>'}'."
            )
            continue

        point_count = _grid_point_count(grid)
        if grid_type == "all_unmasked_pixels":
            full_map_grid_count += 1
        else:
            if point_count is None:
                errors.append(
                    f"Grid for '{name or index}' must provide a valid point count."
                )
            elif point_count < min_scan_points:
                errors.append(
                    f"Grid for '{name or index}' has {point_count} points; "
                    f"at least {min_scan_points} are required."
                )
            if point_count is not None and point_count > 0:
                static_grid_count *= point_count

        if grid_type in {"linear", "log", "integer"}:
            bounds = _numeric_grid_bounds(grid)
            if bounds is None or not bounds[0] < bounds[1]:
                errors.append(
                    f"Grid for '{name or index}' needs numeric minimum < maximum."
                )
            if grid_type == "log" and bounds is not None and bounds[0] <= 0:
                errors.append(f"Log grid for '{name or index}' must be strictly positive.")
        elif grid_type == "values":
            values = grid.get("values")
            if not isinstance(values, list):
                errors.append(f"Values grid for '{name or index}' must contain a list.")
            elif any(isinstance(value, (list, dict)) for value in values):
                errors.append(
                    f"Values grid for '{name or index}' must contain scalar entries."
                )
            elif len(set(values)) != len(values):
                errors.append(f"Values grid for '{name or index}' contains duplicates.")
        elif grid_type == "healpix":
            nside = grid.get("nside")
            if not isinstance(nside, int) or isinstance(nside, bool) or nside <= 0:
                errors.append(f"HEALPix grid for '{name or index}' needs a positive nside.")

        if role == "position":
            if domain != "full_unmasked_sky" or grid_type != "all_unmasked_pixels":
                errors.append(
                    "Position scans must cover full_unmasked_sky using all_unmasked_pixels."
                )
        elif role == "direction":
            if domain != "full_sphere" or grid_type != "healpix":
                errors.append("Direction scans must use a full_sphere HEALPix grid.")
            elif point_count is not None and point_count < min_direction_points:
                errors.append(
                    f"Direction grid has {point_count} points; at least "
                    f"{min_direction_points} are required."
                )
            elif point_count is not None and point_count > max_direction_points:
                errors.append(
                    f"Direction grid has {point_count} points; cap is {max_direction_points}."
                )
        elif role == "axis":
            if domain not in {"full_sphere", "full_projective_sphere"} or grid_type != "healpix":
                errors.append(
                    "Axis scans must use a full_sphere or full_projective_sphere HEALPix grid."
                )
            elif point_count is not None and point_count < min_direction_points:
                errors.append(
                    f"Axis grid has {point_count} points; at least "
                    f"{min_direction_points} are required."
                )
            elif point_count is not None and point_count > max_direction_points:
                errors.append(
                    f"Axis grid has {point_count} points; cap is {max_direction_points}."
                )

        if role in {
            "orientation",
            "scale",
            "multipole",
            "threshold",
            "region_size",
            "weight",
            "amplitude",
        }:
            bounds = _numeric_grid_bounds(grid)
            if bounds is None:
                errors.append(
                    f"Grid for numeric parameter '{name or index}' must contain numeric bounds."
                )
            elif role in {"scale", "multipole", "region_size"} and bounds[0] <= 0:
                errors.append(
                    f"Broadness of '{name or index}' requires a strictly positive grid."
                )
            elif (
                role in {"scale", "multipole", "region_size"}
                and bounds[1] / bounds[0] < min_span_factor
            ):
                errors.append(
                    f"Grid for '{name or index}' spans a factor {bounds[1] / bounds[0]:.2f}; "
                    f"at least {min_span_factor:g} is required."
                )
            elif role in {"orientation", "threshold", "weight", "amplitude"}:
                width = bounds[1] - bounds[0]
                reference = max(abs(bounds[0]), abs(bounds[1]))
                relative_width = width / reference if reference > 0.0 else np.inf
                if relative_width < min_relative_width:
                    errors.append(
                        f"Grid for '{name or index}' has relative width "
                        f"{relative_width:.2f}; at least {min_relative_width:g} is required."
                    )
                standardized_domain = "sigma" in domain or "standardized" in domain
                if (
                    role in {"threshold", "amplitude"}
                    and standardized_domain
                    and width < min_standardized_width
                ):
                    errors.append(
                        f"Standardized grid for '{name or index}' spans {width:.2f}; "
                        f"at least {min_standardized_width:g} is required."
                    )

    if scanned_count == 0:
        errors.append("A scan-enforced test must contain at least one scanned parameter.")
    if full_map_grid_count > 1:
        errors.append("At most one parameter may use the all_unmasked_pixels grid.")
    if static_grid_count > max_static_grid_points:
        errors.append(
            f"Static Cartesian scan grid has {static_grid_count} values; cap is "
            f"{max_static_grid_points}."
        )
    projected_grid_values = static_grid_count * (
        position_grid_upper_bound if full_map_grid_count else 1
    )
    if projected_grid_values > max_total_grid_values:
        errors.append(
            f"Projected scan grid has {projected_grid_values} values; cap is "
            f"{max_total_grid_values}."
        )

    return errors


def audit_scan_code(code: str) -> list[str]:
    """Check enforceable implementation rules before any generated code executes."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"Implementation is not valid Python: {exc.msg} (line {exc.lineno})."]

    errors: list[str] = []
    all_function_defs = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if any(node.name == "analyze_map" for node in all_function_defs):
        errors.append(
            "Do not define analyze_map; define evaluate_scan and let the framework reduce it."
        )
    evaluate_defs = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "evaluate_scan"
    ]
    if len(evaluate_defs) != 1:
        errors.append("Implementation must define exactly one top-level evaluate_scan(m).")
    else:
        evaluate_def = evaluate_defs[0]
        if isinstance(evaluate_def, ast.AsyncFunctionDef):
            errors.append("evaluate_scan must be synchronous, not async.")
        positional = list(evaluate_def.args.posonlyargs) + list(evaluate_def.args.args)
        if (
            len(positional) != 1
            or positional[0].arg != "m"
            or evaluate_def.args.vararg is not None
            or evaluate_def.args.kwarg is not None
            or evaluate_def.args.kwonlyargs
        ):
            errors.append("evaluate_scan must accept exactly one argument: m.")

    has_description = any(
        isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Name) and target.id == "test_description"
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
        )
        for node in tree.body
    )
    if not has_description:
        errors.append("Implementation must assign test_description.")

    if any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree)):
        errors.append("Imports are not allowed; np and hp are already provided.")

    blocked_calls = {
        "open",
        "exec",
        "eval",
        "compile",
        "__import__",
        "input",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
    }
    blocked_attribute_calls = {
        ("np", "load"),
        ("np", "save"),
        ("np", "fromfile"),
        ("np", "memmap"),
        ("hp", "read_map"),
        ("hp", "write_map"),
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "__builtins__":
            errors.append("Direct access to __builtins__ is not allowed.")
        if isinstance(node, ast.Attribute) and node.attr in {
            "__builtins__",
            "__class__",
            "__globals__",
            "__mro__",
            "__subclasses__",
        }:
            errors.append("Dunder-based runtime introspection is not allowed.")
        if isinstance(node, ast.Attribute):
            path_parts = [node.attr]
            root = node.value
            while isinstance(root, ast.Attribute):
                path_parts.append(root.attr)
                root = root.value
            if isinstance(root, ast.Name):
                path_parts.append(root.id)
                dotted_path = ".".join(reversed(path_parts))
                if dotted_path.startswith("np.random"):
                    errors.append("Randomness is not allowed in scan implementations.")
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(
                isinstance(target, ast.Name)
                and target.id
                in {
                    "np",
                    "hp",
                    "evaluate_scan",
                    "_scan_framework_reduce",
                }
                for target in targets
            ):
                errors.append("Trusted scan-framework globals may not be reassigned.")
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in blocked_calls:
            errors.append(f"Call to {node.func.id}() is not allowed.")
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and (node.func.value.id, node.func.attr) in blocked_attribute_calls
        ):
            errors.append(
                f"External data access through {node.func.value.id}.{node.func.attr}() is not allowed."
            )

    if SCAN_FRAMEWORK_MARKER in code:
        errors.append("Generated scan framework code must not be supplied by the implementer.")
    return list(dict.fromkeys(errors))


def scan_grid_size_contract(scan_spec: dict) -> tuple[int, bool]:
    """Return the static grid multiplier and whether map positions are scanned."""
    static_count = 1
    scans_unmasked_pixels = False
    for parameter in scan_spec.get("parameters", []):
        if not isinstance(parameter, dict) or str(
            parameter.get("treatment", "")
        ).strip().lower() != "scanned":
            continue
        grid = parameter.get("grid", {})
        if str(grid.get("type", "")).strip().lower() == "all_unmasked_pixels":
            scans_unmasked_pixels = True
            continue
        point_count = _grid_point_count(grid)
        if point_count is None:
            raise ValueError(
                f"Cannot determine scan-grid size for parameter {parameter.get('name', '<unnamed>')}."
            )
        static_count *= point_count
    return static_count, scans_unmasked_pixels


def build_scan_analysis_code(code: str, scan_spec: dict) -> str:
    """Append the trusted scalar reduction used by registration and execution."""
    normalized_reduction = str(scan_spec.get("reduction", "")).strip().lower()
    if normalized_reduction not in SCAN_REDUCTIONS:
        raise ValueError(f"Unsupported scan reduction: {normalized_reduction}")
    static_grid_count, scans_unmasked_pixels = scan_grid_size_contract(scan_spec)

    wrapper = f'''\n\n{SCAN_FRAMEWORK_MARKER}
def _scan_framework_reduce(
    scan_values,
    expected_size,
    _np=np,
    _reduction={normalized_reduction!r},
):
    values = _np.asarray(scan_values, dtype=float)
    if values.ndim == 0:
        raise TypeError("evaluate_scan(m) must return an array over the declared scan grid")
    if values.size != expected_size:
        raise ValueError(
            f"evaluate_scan(m) returned {{values.size}} grid values; "
            f"SCAN_SPEC requires {{expected_size}}"
        )
    finite_values = values[_np.isfinite(values)]
    if finite_values.size < 2:
        raise ValueError("evaluate_scan(m) produced fewer than two finite scan-grid values")
    if _reduction == "minimum":
        result = _np.min(finite_values)
    elif _reduction == "maximum":
        result = _np.max(finite_values)
    else:
        result = _np.max(_np.abs(finite_values))
    return float(result)

def analyze_map(
    m,
    _evaluate=evaluate_scan,
    _reduce=_scan_framework_reduce,
    _np=np,
    _static_grid_count={static_grid_count},
    _scans_unmasked_pixels={scans_unmasked_pixels!r},
):
    position_count = int(_np.count_nonzero(_np.isfinite(m))) if _scans_unmasked_pixels else 1
    expected_size = _static_grid_count * position_count
    return _reduce(_evaluate(m), expected_size)
'''
    return code.rstrip() + wrapper


def strip_scan_framework(code: str) -> str:
    """Remove a previously appended trusted wrapper before asking for a revision."""
    return code.split(SCAN_FRAMEWORK_MARKER, 1)[0].rstrip()


class ScanAnomalyAgent(AnomalyAgent):
    """Agent variant that forbids a posteriori parameters via scan-and-maximize tests."""

    agent_mode = "scan"

    class State(AnomalyAgent.State):
        scan_spec: dict
        scan_review: dict

    def __init__(
        self,
        *args,
        scan_planner_path: str | Path | None = None,
        scan_implement_path: str | Path | None = None,
        scan_review_path: str | Path | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.planner_prompt_path = (
            Path(scan_planner_path)
            if scan_planner_path
            else file_paths.scan_planner_dir
        )
        self.implement_prompt_path = (
            Path(scan_implement_path)
            if scan_implement_path
            else file_paths.scan_implement_dir
        )
        self.scan_review_prompt_path = (
            Path(scan_review_path)
            if scan_review_path
            else file_paths.scan_review_dir
        )
        self.python_env["scan_spec"] = None
        self.python_env["scan_review"] = None

    def planner_node(self, state: State):
        result = super().planner_node(state)
        if result.get("node_retry") is not False or not result.get("messages"):
            return result

        proposal_text = message_content_to_text(result["messages"][-1].content)
        try:
            test_name, test_description, scan_spec = parse_scan_proposal(proposal_text)
            errors = validate_scan_spec(
                scan_spec,
                test_description,
                min_scan_points=int(self.test_config.get("scan_min_points", 3)),
                min_span_factor=float(
                    self.test_config.get("scan_min_span_factor", 3.0)
                ),
                min_relative_width=float(
                    self.test_config.get("scan_min_relative_width", 0.5)
                ),
                min_standardized_width=float(
                    self.test_config.get("scan_min_standardized_width", 2.0)
                ),
                min_direction_points=int(
                    self.test_config.get("scan_min_direction_points", 48)
                ),
                max_direction_points=int(
                    self.test_config.get("scan_max_direction_points", 768)
                ),
                max_static_grid_points=int(
                    self.test_config.get("scan_max_static_grid_points", 100_000)
                ),
                position_grid_upper_bound=int(
                    self.test_config.get("scan_position_grid_upper_bound", 786_432)
                ),
                max_total_grid_values=int(
                    self.test_config.get("scan_max_total_grid_values", 5_000_000)
                ),
            )
        except ValueError as exc:
            test_name = result.get("current_test_name", "Invalid scan proposal")
            test_description = result.get("current_test_description", "")
            scan_spec = {}
            errors = [str(exc)]

        if errors:
            feedback = (
                "REJECTED BY SCAN POLICY:\n- "
                + "\n- ".join(errors)
                + "\n\nProvide a corrected SCAN_SPEC and DESCRIPTION."
            )
            self.persist_rejected_proposal(
                state,
                test_name,
                test_description,
                rotation_issue=None,
                novelty_issue=None,
                planner_feedback=feedback,
            )
            return {
                "messages": [AIMessage(content=feedback)],
                "node_retry": True,
            }

        result["current_test_name"] = test_name
        result["current_test_description"] = test_description
        result["scan_spec"] = scan_spec
        return result

    def implement_node(self, state: State):
        test_name = state["current_test_name"]
        test_description = state["current_test_description"]
        previous_error = self.python_env.get("last_error") or state.get(
            "python_env", {}
        ).get("last_error")

        if previous_error:
            previous_code = strip_scan_framework(
                self.retrieve_state(state, "code", max_entries=1) or ""
            )
            previous_code = previous_code or "None"
        else:
            previous_code = ""

        with self.implement_prompt_path.open("r", encoding="utf-8") as stream:
            prompt_config = yaml.safe_load(stream)
            template = prompt_config["template"]
            additional_template = prompt_config["additional_template"]

        if previous_error:
            guidance = PromptTemplate.from_template(additional_template).format_prompt(
                previous_error=previous_error
            ).to_string()
        else:
            guidance = ""

        scan_spec_text = yaml.safe_dump(
            state.get("scan_spec", {}), sort_keys=False
        ).strip()
        prompt = PromptTemplate.from_template(template).format_prompt(
            test_name=test_name,
            test_description=test_description,
            scan_spec=scan_spec_text,
            previous_code=previous_code,
            guidance=guidance,
        )

        print("\n##### PROMPT #####\n")
        print(prompt.to_string())
        msg = self.implement_llm.invoke(prompt)

        if getattr(msg, "tool_calls", None):
            code_text = msg.tool_calls[0]["args"].get("code", "")
            return {
                "messages": [msg],
                "code": [AIMessage(content=code_text)],
                "node_retry": False,
            }
        should_retry = bool(
            getattr(msg, "invalid_tool_calls", None)
            or not getattr(msg, "tool_calls", None)
        )
        return {"messages": [msg], "node_retry": should_retry}

    def review_scan_implementation(
        self,
        *,
        scan_spec: dict,
        test_description: str,
        code: str,
    ) -> dict:
        with self.scan_review_prompt_path.open("r", encoding="utf-8") as stream:
            template = yaml.safe_load(stream)["template"]

        prompt = PromptTemplate.from_template(template).format_prompt(
            scan_spec=yaml.safe_dump(scan_spec, sort_keys=False).strip(),
            test_description=test_description,
            analysis_code=code,
        )
        print("\n##### SCAN POLICY REVIEW PROMPT #####\n")
        print(prompt.to_string())

        try:
            msg = self.llm.invoke(prompt)
            review_text = message_content_to_text(msg.content)
            review = text_to_dict(
                review_text, ["VERDICT", "REASON", "REVISION_GUIDANCE"]
            )
            verdict = review["VERDICT"].strip().lower()
            accepted = verdict == "accept"
            reason = review["REASON"].strip()
            guidance = review["REVISION_GUIDANCE"].strip()
            if verdict not in {"accept", "reject"}:
                reason = reason or "Policy reviewer did not return ACCEPT or REJECT."
            return {
                "accepted": accepted,
                "verdict": verdict or "invalid",
                "reason": reason,
                "revision_guidance": guidance,
            }
        except Exception as exc:
            return {
                "accepted": False,
                "verdict": "error",
                "reason": f"Scan policy review failed closed: {type(exc).__name__}: {exc}",
                "revision_guidance": "Retry the implementation policy review.",
            }

    def registration_rejection(
        self,
        runtime,
        errors: list[str],
        *,
        heading: str = "SCAN POLICY REJECTED",
    ) -> Command:
        tool_call_id = runtime.state["messages"][-1].tool_calls[0]["id"]
        error = f"{heading}:\n- " + "\n- ".join(errors)
        self.python_env["last_error"] = error
        self.python_env["analyze_map"] = None
        self.python_env["summarize_results"] = None
        self.python_env["test_description"] = None
        self.python_env["scan_spec"] = None
        self.python_env["scan_review"] = None
        return Command(
            update={
                "messages": [ToolMessage(error, tool_call_id=tool_call_id)],
                "node_retry": True,
                "python_env": self.python_env.copy(),
            }
        )

    def register(self, code: str, runtime) -> Command:
        scan_spec = runtime.state.get("scan_spec")
        test_description = runtime.state.get("current_test_description", "")
        if not isinstance(scan_spec, dict):
            return self.registration_rejection(
                runtime, ["No validated SCAN_SPEC is present in graph state."]
            )

        policy_errors = validate_scan_spec(
            scan_spec,
            test_description,
            min_scan_points=int(self.test_config.get("scan_min_points", 3)),
            min_span_factor=float(
                self.test_config.get("scan_min_span_factor", 3.0)
            ),
            min_relative_width=float(
                self.test_config.get("scan_min_relative_width", 0.5)
            ),
            min_standardized_width=float(
                self.test_config.get("scan_min_standardized_width", 2.0)
            ),
            min_direction_points=int(
                self.test_config.get("scan_min_direction_points", 48)
            ),
            max_direction_points=int(
                self.test_config.get("scan_max_direction_points", 768)
            ),
            max_static_grid_points=int(
                self.test_config.get("scan_max_static_grid_points", 100_000)
            ),
            position_grid_upper_bound=int(
                self.test_config.get("scan_position_grid_upper_bound", 786_432)
            ),
            max_total_grid_values=int(
                self.test_config.get("scan_max_total_grid_values", 5_000_000)
            ),
        )
        policy_errors.extend(audit_scan_code(code))
        if policy_errors:
            return self.registration_rejection(runtime, policy_errors)

        review = self.review_scan_implementation(
            scan_spec=scan_spec,
            test_description=test_description,
            code=code,
        )
        review = dict(review)
        review["policy_version"] = SCAN_POLICY_VERSION
        review["implementation_sha256"] = hashlib.sha256(code.encode("utf-8")).hexdigest()
        review["scan_spec_sha256"] = hashlib.sha256(
            json.dumps(scan_spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if not review["accepted"]:
            errors = [review["reason"] or "Semantic scan-policy review rejected the code."]
            if review["revision_guidance"]:
                errors.append(review["revision_guidance"])
            return self.registration_rejection(runtime, errors)

        wrapped_code = build_scan_analysis_code(code, scan_spec)
        review["registered_code_sha256"] = hashlib.sha256(
            wrapped_code.encode("utf-8")
        ).hexdigest()
        try:
            command = super().register(wrapped_code, runtime)
        except RuntimeError as exc:
            limit_seconds = getattr(
                self, "_scan_preflight_probe_limit_seconds", None
            )
            simulation_count = getattr(
                self, "_scan_preflight_simulation_count", None
            )
            if (
                "Restarting test" in str(exc)
                and limit_seconds is not None
                and simulation_count is not None
            ):
                timing_error = (
                    f"The warmed per-map timing probe exceeded {limit_seconds:.3f} "
                    f"seconds, the projected limit for {simulation_count} simulations."
                )
                timing_guidance = (
                    "Vectorize evaluate_scan or reduce grid density without violating "
                    "the validated broad-scan constraints."
                )
            else:
                timing_error = (
                    "The registration probe raised RuntimeError: " + str(exc)
                )
                timing_guidance = "Revise evaluate_scan to avoid this runtime error."
            return self.registration_rejection(
                runtime,
                [
                    timing_error,
                    timing_guidance,
                ],
                heading="SCAN RUNTIME REJECTED",
            )
        if self.python_env.get("last_error") is not None:
            return command

        self.python_env["scan_spec"] = scan_spec
        self.python_env["scan_review"] = review
        if isinstance(command.update, dict):
            command.update["code"] = [AIMessage(content=wrapped_code)]
            command.update["scan_spec"] = scan_spec
            command.update["scan_review"] = review
        return command

    def simulation_map_count(self) -> int:
        paths = sorted(glob(str(self.sim_maps_path)))
        if not paths:
            raise FileNotFoundError(f"No simulation files matched: {self.sim_maps_path}")

        count = 0
        for path in paths:
            maps = np.load(path, allow_pickle=False, mmap_mode="r")
            count += 1 if maps.ndim == 1 else int(maps.shape[0])
        return count

    def code_execution_environment(self) -> dict:
        safe_builtin_names = {
            "Exception",
            "RuntimeError",
            "TypeError",
            "ValueError",
            "abs",
            "all",
            "any",
            "bool",
            "dict",
            "enumerate",
            "filter",
            "float",
            "int",
            "isinstance",
            "len",
            "list",
            "map",
            "max",
            "min",
            "range",
            "reversed",
            "round",
            "set",
            "slice",
            "sorted",
            "str",
            "sum",
            "tuple",
            "zip",
        }
        safe_builtins = {
            name: getattr(builtins, name) for name in safe_builtin_names
        }

        def restricted_library_import(
            name,
            globals=None,
            locals=None,
            fromlist=(),
            level=0,
        ):
            package = str((globals or {}).get("__package__", ""))
            root_name = package.split(".", 1)[0] if level else str(name).split(".", 1)[0]
            if root_name not in {"numpy", "healpy"}:
                raise ImportError(
                    f"Generated scan code may not import module '{name}'."
                )
            return builtins.__import__(name, globals, locals, fromlist, level)

        # NumPy/healpy dispatchers perform internal imports at call time. Expose
        # only those library roots; generated import syntax/direct calls are
        # separately rejected by the AST policy.
        safe_builtins["__import__"] = restricted_library_import
        return {"np": np, "hp": hp, "__builtins__": safe_builtins}

    def preflight_probe_max_minutes(self) -> float:
        simulation_count = self.simulation_map_count()
        safety_factor = float(
            self.test_config.get("scan_runtime_safety_factor", 1.25)
        )
        if not np.isfinite(safety_factor) or safety_factor < 1.0:
            raise ValueError(
                "scan_runtime_safety_factor must be finite and at least 1.0"
            )
        probe_max_minutes = self.test_config["max_test_minutes"] / (
            (simulation_count + 1) * safety_factor
        )
        self._scan_preflight_simulation_count = simulation_count
        self._scan_preflight_probe_limit_seconds = probe_max_minutes * 60.0
        return probe_max_minutes

    def warm_up_registration_probe(self, analyze_fn, sample_map, mask) -> None:
        warmup_map = np.asarray(sample_map, dtype=float).copy()
        warmup_map[~mask] = np.nan
        analyze_fn(warmup_map)

    def run_registered_analysis(self, state: State) -> str:
        output = super().run_registered_analysis(state)
        result = self.python_env.get("last_result")
        if not isinstance(result, dict):
            return output

        scan_spec = state.get("scan_spec") or self.python_env.get("scan_spec") or {}
        scan_review = (
            state.get("scan_review") or self.python_env.get("scan_review") or {}
        )
        output_dir = Path(result["output_dir"])
        manifest_path = output_dir / "scan_manifest.yaml"
        review_path = output_dir / "scan_policy_review.json"
        with manifest_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(scan_spec, stream, sort_keys=False)
        with review_path.open("w", encoding="utf-8") as stream:
            json.dump(scan_review, stream, indent=2)

        result["scan_spec"] = scan_spec
        result["scan_review"] = scan_review
        result["scan_policy_version"] = SCAN_POLICY_VERSION
        result["scan_reduction_control"] = "framework"
        result["scan_manifest_path"] = str(manifest_path)
        result["scan_review_path"] = str(review_path)
        return output


def scan_run_config(runtime_configs: dict, **kwargs) -> dict:
    kwargs.setdefault("agent_mode", ScanAnomalyAgent.agent_mode)
    config = effective_run_config(runtime_configs, **kwargs)
    config["test"].setdefault("scan_runtime_safety_factor", 1.25)
    config["test"].setdefault("scan_min_points", 3)
    config["test"].setdefault("scan_min_span_factor", 3.0)
    config["test"].setdefault("scan_min_relative_width", 0.5)
    config["test"].setdefault("scan_min_standardized_width", 2.0)
    config["test"].setdefault("scan_min_direction_points", 48)
    config["test"].setdefault("scan_max_direction_points", 768)
    config["test"].setdefault("scan_max_static_grid_points", 100_000)
    config["test"].setdefault("scan_position_grid_upper_bound", 786_432)
    config["test"].setdefault("scan_max_total_grid_values", 5_000_000)
    config["scan"] = {
        "a_posteriori_parameters": (
            "disallowed; every assumed parameter is scanned over an a priori "
            "grid or fixed by an a priori structural justification"
        ),
        "scan_statistic": "extremum of the scanned statistic over the grid",
        "enforcement": {
            "policy_version": SCAN_POLICY_VERSION,
            "manifest_validation": "deterministic and fail-closed",
            "implementation_review": "semantic and fail-closed before map evaluation",
            "reduction_control": "framework-owned",
            "runtime_preflight": "projected from the configured simulation count",
        },
    }
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Run the scan-enforced CMB anomaly agent."
    )
    parser.add_argument(
        "--config",
        help="Optional run config YAML overriding agent, test, plot, and paths defaults.",
    )
    parser.add_argument("--model", help="Model name to use for all agent LLM calls.")
    parser.add_argument("--thread-id", help="Checkpoint/output thread id for this run.")
    parser.add_argument("--base-url", help="OpenAI-compatible API base URL.")
    parser.add_argument(
        "--reasoning-effort",
        help="Reasoning effort for supported models. Use 'none' to disable.",
    )
    parser.add_argument(
        "--sim-maps",
        help=(
            "Simulation map .npy stack or glob. Defaults to the repository "
            "path configured in file_paths.py."
        ),
    )
    args = parser.parse_args()

    runtime_configs = load_runtime_configs(args.config)
    agent_config = runtime_configs["agent"]
    paths_config = runtime_configs["paths"]

    model = args.model or agent_config.get("model")
    if not model:
        parser.error("model must be set in the config file or passed with --model")

    if args.thread_id:
        thread_id = args.thread_id
    elif args.config:
        thread_id = agent_config.get("thread_id", "scan_run")
    else:
        thread_id = "scan_run"

    base_url = args.base_url or agent_config.get(
        "base_url", "https://openrouter.ai/api/v1"
    )
    reasoning_effort_value = (
        args.reasoning_effort
        if args.reasoning_effort is not None
        else agent_config.get("reasoning_effort")
    )
    reasoning_effort = normalize_optional_config_value(reasoning_effort_value)
    sim_maps_path = (
        args.sim_maps
        or paths_config.get("sim_maps_path")
        or agent_config.get("sim_maps_path")
    )

    agent = ScanAnomalyAgent(
        model=model,
        thread_id=thread_id,
        base_url=base_url,
        reasoning_effort=reasoning_effort,
        sim_maps_path=sim_maps_path,
        test_config=runtime_configs["test"],
        plot_config=runtime_configs["plot"],
        run_config=scan_run_config(
            runtime_configs,
            model=model,
            thread_id=thread_id,
            base_url=base_url,
            reasoning_effort=reasoning_effort,
            sim_maps_path=sim_maps_path,
        ),
    )
    agent()


if __name__ == "__main__":
    main()
