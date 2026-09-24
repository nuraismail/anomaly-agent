import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analyse_run import (
    TestResult as RunTestResult,
    analyse_run,
    compute_simulation_min_p_values,
    effective_tests_from_corr,
    empirical_p_value,
    parse_tail,
    pooled_p_values,
)


def make_result(
    name: str,
    values: list[float],
    *,
    planck_stat: float = 0.0,
    tail: str = "one-tailed upper",
) -> RunTestResult:
    simulations = np.asarray(values, dtype=float)
    return RunTestResult(
        index=0,
        name=name,
        directory=Path(name),
        planck_stat=planck_stat,
        simulation_statistics=simulations,
        tail=tail,
        planck_p_value=empirical_p_value(planck_stat, simulations, tail),
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

    min_p_values, min_test_indices = compute_simulation_min_p_values(tests)

    np.testing.assert_allclose(min_p_values, [0.25, 0.5, 0.25])
    np.testing.assert_array_equal(min_test_indices, [1, 0, 0])


@pytest.mark.parametrize(
    "tail, expected",
    [
        ("one-tailed upper", [0.75, 1.0, 0.75, 0.25]),
        ("one-tailed lower", [0.75, 0.25, 0.75, 1.0]),
        ("two-tailed", [1.0, 0.5, 1.0, 0.5]),
    ],
)
def test_pooled_ranks_count_ties_inclusively_and_preserve_planck_p(tail, expected):
    test = make_result("ties", [1.0, 2.0, 4.0], planck_stat=2.0, tail=tail)

    p_values = pooled_p_values(test)

    np.testing.assert_allclose(p_values, expected)
    assert p_values[0] == test.planck_p_value


def test_single_two_tailed_test_preserves_both_extreme_skies():
    test = make_result("two tails", list(range(1, 100)), tail="two-tailed")

    min_p_values, _ = compute_simulation_min_p_values([test])
    global_p = (np.count_nonzero(min_p_values <= test.planck_p_value) + 1) / 100

    assert global_p == pytest.approx(0.02)
    assert global_p == test.planck_p_value


def test_duplicate_tests_do_not_add_a_multiple_testing_penalty():
    test = make_result("duplicate", [1.0, 2.0, 3.0], planck_stat=4.0)

    min_p_values, _ = compute_simulation_min_p_values([test, test])
    global_p = (np.count_nonzero(min_p_values <= test.planck_p_value) + 1) / 4

    assert global_p == test.planck_p_value


@pytest.mark.parametrize("tail", ["one-tailed upper", "one-tailed lower", "two-tailed"])
def test_global_p_values_are_calibrated_over_exchangeable_observed_labels(tail):
    # Fix the sky statistics and let each sky take a turn as the observation.
    # The old rule assigns p=1/5 to multiple distinct extreme skies and fails
    # this exact finite-sample calibration check.
    skies = np.asarray([[5, 1], [1, 5], [2, 2], [3, 3], [4, 4]], dtype=float)
    global_p_values = []
    for observed_index in range(len(skies)):
        simulations = np.delete(skies, observed_index, axis=0)
        tests = [
            make_result(
                str(column),
                simulations[:, column],
                planck_stat=skies[observed_index, column],
                tail=tail,
            )
            for column in range(skies.shape[1])
        ]
        planck_min_p = min(test.planck_p_value for test in tests)
        min_p_values, _ = compute_simulation_min_p_values(tests)
        global_p_values.append(
            (np.count_nonzero(min_p_values <= planck_min_p) + 1) / len(skies)
        )

    global_p_values = np.asarray(global_p_values)
    for alpha in np.unique(global_p_values):
        assert np.mean(global_p_values <= alpha) <= alpha + 1e-12


def test_pooling_requires_aligned_simulation_counts():
    tests = [make_result("short", [1, 2]), make_result("long", [1, 2, 3])]
    with pytest.raises(ValueError, match="Align simulation counts"):
        compute_simulation_min_p_values(tests)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_pooling_rejects_nonfinite_observed_statistics(bad_value):
    test = make_result("invalid observation", [1, 2, 3], planck_stat=bad_value)
    with pytest.raises(ValueError, match="Non-finite statistics"):
        pooled_p_values(test)


@pytest.mark.parametrize(
    "tests, expected_min_p, expected_global_p",
    [
        (
            [
                make_result("first", [2, 1], planck_stat=3),
                make_result("second", [3, 2], planck_stat=1),
            ],
            1 / 3,
            2 / 3,
        ),
        (
            [
                make_result("long", [10, 20, 30, 40] + [-1] * 96),
                make_result("short", [1, 2, 3, 4]),
            ],
            1.0,
            1.0,
        ),
    ],
    ids=["extreme-across-multiple-tests", "recompute-after-truncation"],
)
def test_analyse_run_saves_corrected_global_results(
    tmp_path, tests, expected_min_p, expected_global_p
):
    for index, test in enumerate(tests, start=1):
        test_dir = tmp_path / f"Test_{index:02d}"
        test_dir.mkdir()
        np.save(test_dir / "simulation_statistics.npy", test.simulation_statistics)
        np.save(test_dir / "planck_statistic.npy", [test.planck_stat])
        (test_dir / "result_summary.json").write_text(
            json.dumps({
                "saved_test_index": index,
                "test_name": test.name,
                "tail": test.tail,
                "planck_stat": test.planck_stat,
                "p_value": test.planck_p_value,
            }),
            encoding="utf-8",
        )

    summary = analyse_run(
        tmp_path,
        output_dir=None,
        plot_config_path=None,
        n_bootstrap=20,
        n_effective_bootstrap=0,
        seed=12345,
    )
    global_result = summary["global_pvalue"]
    assert global_result["global_p_value"] == pytest.approx(expected_global_p)
    assert global_result["planck_min_p_value"] == pytest.approx(expected_min_p)
    assert global_result["simulation_p_values"] == "pooled-planck-and-simulations"

    output_dir = tmp_path / "run_analysis"
    saved = json.loads((output_dir / "global_pvalue_summary.json").read_text())
    assert saved["global_p_value"] == pytest.approx(expected_global_p)
    assert min(test["planck_p_value"] for test in saved["tests"]) == pytest.approx(expected_min_p)
    assert saved["simulation_p_values"] == "pooled-planck-and-simulations"
    assert saved["bootstrap"]["n_bootstrap"] == 20
    min_p_values = np.load(output_dir / "global_min_p_values.npy")
    n_sims = min(test.simulation_statistics.size for test in tests)
    assert min_p_values.shape == (n_sims,)
    assert all(test["n_sims"] == n_sims for test in saved["tests"])


def test_effective_tests_from_known_correlation_matrices():
    assert effective_tests_from_corr(np.eye(3))[0] == pytest.approx(3.0)
    assert effective_tests_from_corr(np.ones((3, 3)))[0] == pytest.approx(1.0)
