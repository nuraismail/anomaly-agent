from pathlib import Path

import numpy as np
import pytest

from scripts.analyse_run import (
    TestResult as RunTestResult,
    compute_simulation_min_p_values,
    effective_tests_from_corr,
    empirical_p_value,
    parse_tail,
)


def make_result(name: str, values: list[float]) -> RunTestResult:
    return RunTestResult(
        index=0,
        name=name,
        directory=Path(name),
        planck_stat=0.0,
        simulation_statistics=np.asarray(values, dtype=float),
        tail="one-tailed upper",
        planck_p_value=1.0,
        summary={},
    )


def test_parse_tail_accepts_common_tail_labels():
    assert parse_tail("two-tailed") == ("two", None)
    assert parse_tail("one_tailed_upper") == ("one", "upper")
    assert parse_tail("one tailed lower") == ("one", "lower")

    with pytest.raises(ValueError, match="missing upper/lower"):
        parse_tail("one-tailed")


def test_empirical_p_value_uses_add_one_correction():
    reference = np.asarray([1.0, 2.0, 3.0, 4.0])

    assert empirical_p_value(5.0, reference, "one-tailed upper") == pytest.approx(0.2)
    assert empirical_p_value(0.0, reference, "one-tailed lower") == pytest.approx(0.2)
    assert empirical_p_value(3.5, reference, "two-tailed") == pytest.approx(0.8)


def test_compute_simulation_min_p_values_across_tests():
    tests = [
        make_result("ascending", [1.0, 2.0, 3.0]),
        make_result("descending", [3.0, 2.0, 1.0]),
    ]

    min_p_values, min_test_indices = compute_simulation_min_p_values(
        tests,
        leave_one_out=False,
    )

    np.testing.assert_allclose(min_p_values, [0.5, 0.75, 0.5])
    np.testing.assert_array_equal(min_test_indices, [1, 0, 0])


def test_effective_tests_from_known_correlation_matrices():
    assert effective_tests_from_corr(np.eye(3))[0] == pytest.approx(3.0)
    assert effective_tests_from_corr(np.ones((3, 3)))[0] == pytest.approx(1.0)
