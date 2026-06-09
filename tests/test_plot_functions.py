import numpy as np

from utils.plot_functions import plot_results


def test_plot_results_accepts_missing_summary(tmp_path):
    result = plot_results(
        5.0,
        np.asarray([1.0, 2.0, 3.0, 4.0]),
        tmp_path,
        summary=None,
        plot_config={},
    )

    assert result["kind"] == "histogram"
    assert (tmp_path / "statistic_figure.png").exists()
    assert (tmp_path / "statistic_figure.pdf").exists()


def test_plot_results_honors_explicit_plot_kind(tmp_path):
    result = plot_results(
        0.0,
        np.zeros(30),
        tmp_path,
        summary={"plot_spec": {"kind": "ecdf"}},
        plot_config={},
    )

    assert result["kind"] == "ecdf"
