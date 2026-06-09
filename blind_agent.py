"""Domain-blinded HEALPix scalar-field anomaly agent.

This module reuses the exploratory AnomalyAgent execution workflow but replaces
the LLM-facing prompts and labels so the agent is asked to treat the data as a
generic masked scalar field on the sphere rather than as CMB data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import file_paths

from anomaly_agent import (
    AnomalyAgent,
    effective_run_config,
    load_runtime_configs,
    normalize_optional_config_value,
)


class BlindAnomalyAgent(AnomalyAgent):
    """Agent variant for domain-blinded HEALPix scalar-field tests."""

    agent_mode = "blind"

    _TEXT_REPLACEMENTS = (
        ("Planck SMICA", "the observed map"),
        ("Planck", "observed"),
        ("SMICA", "observed"),
        ("CMB", "spherical-field"),
        ("LCDM", "null-model"),
        ("ΛCDM", "null-model"),
        ("microK", "arbitrary units"),
        ("microkelvin", "arbitrary units"),
        ("μK", "arbitrary units"),
    )

    _KEY_REPLACEMENTS = (
        ("planck", "observed"),
        ("cmb", "field"),
        ("smica", "observed"),
        ("lcdm", "null_model"),
    )

    def __init__(
        self,
        *args,
        blind_planner_path: str | Path | None = None,
        blind_implement_path: str | Path | None = None,
        blind_hypothesis_path: str | Path | None = None,
        blind_summary_path: str | Path | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.planner_prompt_path = (
            Path(blind_planner_path)
            if blind_planner_path
            else file_paths.blind_planner_dir
        )
        self.implement_prompt_path = (
            Path(blind_implement_path)
            if blind_implement_path
            else file_paths.blind_implement_dir
        )
        self.hypothesis_prompt_path = (
            Path(blind_hypothesis_path)
            if blind_hypothesis_path
            else file_paths.blind_hypothesis_dir
        )
        self.summary_prompt_path = (
            Path(blind_summary_path)
            if blind_summary_path
            else file_paths.blind_summary_dir
        )

        self.allow_search_tools = False
        self.observed_map_label = "observed map"
        self.observed_statistic_label = "Observed statistic"
        self.show_simulation_sources = False
        self.summary_stop_markers = [
            "Test name",
            "Description",
            "Results",
            "Interpretation",
            "Meets hypothesis?",
            "Comparison to previous tests",
            "Test novelty",
            "Verdict",
        ]

    def anonymize_prompt_value(self, value):
        if isinstance(value, dict):
            return {
                self.anonymize_prompt_key(key): self.anonymize_prompt_value(val)
                for key, val in value.items()
            }
        if isinstance(value, list):
            return [self.anonymize_prompt_value(item) for item in value]
        if isinstance(value, tuple):
            return [self.anonymize_prompt_value(item) for item in value]
        if isinstance(value, str):
            text = value
            for old, new in self._TEXT_REPLACEMENTS:
                text = text.replace(old, new)
                text = text.replace(old.lower(), new)
            return text
        return value

    def anonymize_prompt_key(self, key):
        text = str(key).lower()
        for old, new in self._KEY_REPLACEMENTS:
            text = text.replace(old, new)
        return text

    def result_payload_for_prompt(self, result_payload: dict) -> dict:
        payload = dict(result_payload)
        payload.pop("analysis_code", None)
        return self.anonymize_prompt_value(payload)


def blind_run_config(runtime_configs: dict, **kwargs) -> dict:
    kwargs.setdefault("agent_mode", BlindAnomalyAgent.agent_mode)
    config = effective_run_config(runtime_configs, **kwargs)
    config["blind"] = {
        "data_description": "generic masked HEALPix scalar field on the sphere",
        "search_tools_enabled": False,
        "domain_specific_prior_knowledge": "disabled in prompts",
    }
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Run the domain-blinded HEALPix scalar-field anomaly agent."
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
        thread_id = agent_config.get("thread_id", "blind_run")
    else:
        thread_id = "blind_run"

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

    agent = BlindAnomalyAgent(
        model=model,
        thread_id=thread_id,
        base_url=base_url,
        reasoning_effort=reasoning_effort,
        sim_maps_path=sim_maps_path,
        test_config=runtime_configs["test"],
        plot_config=runtime_configs["plot"],
        run_config=blind_run_config(
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
