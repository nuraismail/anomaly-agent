import numpy as np
import pytest
from langchain_core.messages import AIMessage

import anomaly_agent
from anomaly_agent import AnomalyAgent


def make_minimal_agent(tmp_path):
    agent = AnomalyAgent.__new__(AnomalyAgent)
    agent.agent_mode = "exploratory"
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
    }
    agent.observed_map_label = "Planck map"
    agent.observed_statistic_label = "Observed Planck statistic"
    agent.show_simulation_sources = True
    return agent


def test_run_registered_analysis_computes_empirical_upper_tail(tmp_path, monkeypatch):
    agent = make_minimal_agent(tmp_path)
    planck_map = np.full(12, 5.0)
    mask = np.ones(12, dtype=bool)
    sim_maps = [
        ("sim[0]", np.full(12, 1.0)),
        ("sim[1]", np.full(12, 2.0)),
        ("sim[2]", np.full(12, 3.0)),
        ("sim[3]", np.full(12, 4.0)),
    ]

    monkeypatch.setattr(agent, "prepare_planck_data", lambda target_nside: (planck_map, mask))
    monkeypatch.setattr(agent, "iter_simulation_maps", lambda: iter(sim_maps))

    def fake_plot_results(
        planck_stat,
        sim_results,
        output_dir,
        summary=None,
        plot_config=None,
        test_config=None,
    ):
        assert planck_stat == 5.0
        np.testing.assert_allclose(sim_results, [1.0, 2.0, 3.0, 4.0])
        assert test_config is agent.test_config
        return {
            "png": str(output_dir / "statistic_figure.png"),
            "pdf": str(output_dir / "statistic_figure.pdf"),
            "kind": "histogram",
        }

    monkeypatch.setattr(anomaly_agent, "plot_results", fake_plot_results)

    code = """
import numpy as np

def analyze_map(m):
    return float(np.nanmean(m))

def summarize_results(planck_stat, sim_results):
    return {"plot_spec": {"kind": "histogram"}, "extra_note": "ok"}
"""
    state = {
        "code": [AIMessage(content=code)],
        "current_test_name": "Mean statistic",
        "current_test_description": "Mean of unmasked pixels.",
        "test_hypothesis": [AIMessage(content="The observed mean is high.")],
        "test_type": [AIMessage(content="one-tailed upper")],
        "justification": [AIMessage(content="Upper tail empirical comparison.")],
        "tested_anomalies": [],
    }

    output = agent.run_registered_analysis(state)
    result = agent.python_env["last_result"]

    assert output.startswith("CODE OUTPUT")
    assert agent.python_env["last_error"] is None
    assert result["planck_stat"] == 5.0
    assert result["n_sims"] == 4
    assert result["p_value"] == pytest.approx(0.2)
    assert result["tail"] == "one-tailed upper"
    assert result["custom_summary"]["summary_source"] == "model"
    assert result["custom_summary"]["extra_note"] == "ok"
    assert (tmp_path / "Test_01_mean_statistic" / "simulation_statistics.npy").exists()
    assert (tmp_path / "Test_01_mean_statistic" / "planck_statistic.npy").exists()
