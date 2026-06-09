from pathlib import Path

import pytest

import anomaly_agent
from anomaly_agent import (
    load_runtime_configs,
    merge_config,
    normalize_optional_config_value,
)


def write_yaml(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_normalize_optional_config_value():
    assert normalize_optional_config_value(None) is None
    assert normalize_optional_config_value(" none ") is None
    assert normalize_optional_config_value("null") is None
    assert normalize_optional_config_value("high") == "high"


def test_merge_config_recurses_without_dropping_sibling_values():
    base = {"agent": {"model": "old", "thread_id": "run"}, "test": {"n": 1}}
    override = {"agent": {"model": "new"}, "paths": {"sim_maps_path": "maps.npy"}}

    merged = merge_config(base, override)

    assert merged == {
        "agent": {"model": "new", "thread_id": "run"},
        "test": {"n": 1},
        "paths": {"sim_maps_path": "maps.npy"},
    }
    assert base["agent"]["model"] == "old"


def test_load_runtime_configs_merges_sectioned_override(tmp_path, monkeypatch):
    agent_path = write_yaml(tmp_path / "agent.yaml", "model: old\nthread_id: base\n")
    test_path = write_yaml(tmp_path / "test.yaml", "tests_to_run: 5\n")
    plot_path = write_yaml(tmp_path / "plot.yaml", "figure.dpi: 100\n")
    override_path = write_yaml(
        tmp_path / "override.yaml",
        (
            "agent:\n"
            "  model: new\n"
            "test:\n"
            "  tests_to_run: 1\n"
            "paths:\n"
            "  sim_maps_path: custom.npy\n"
        ),
    )

    monkeypatch.setattr(anomaly_agent.file_paths, "agent_config_dir", agent_path)
    monkeypatch.setattr(anomaly_agent.file_paths, "test_config_dir", test_path)
    monkeypatch.setattr(anomaly_agent.file_paths, "plot_config_dir", plot_path)

    configs = load_runtime_configs(override_path)

    assert configs["agent"] == {"model": "new", "thread_id": "base"}
    assert configs["test"] == {"tests_to_run": 1}
    assert configs["plot"] == {"figure.dpi": 100}
    assert configs["paths"] == {"sim_maps_path": "custom.npy"}


def test_load_runtime_configs_rejects_unknown_sections(tmp_path, monkeypatch):
    monkeypatch.setattr(
        anomaly_agent.file_paths,
        "agent_config_dir",
        write_yaml(tmp_path / "agent.yaml", "model: old\n"),
    )
    monkeypatch.setattr(
        anomaly_agent.file_paths,
        "test_config_dir",
        write_yaml(tmp_path / "test.yaml", "tests_to_run: 5\n"),
    )
    monkeypatch.setattr(
        anomaly_agent.file_paths,
        "plot_config_dir",
        write_yaml(tmp_path / "plot.yaml", "{}\n"),
    )
    override_path = write_yaml(
        tmp_path / "override.yaml",
        "agent:\n  model: new\nunknown:\n  value: true\n",
    )

    with pytest.raises(ValueError, match="Unknown config section"):
        load_runtime_configs(override_path)
