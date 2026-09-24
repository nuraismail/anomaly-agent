import json

from langchain_core.messages import AIMessage

import anomaly_agent
import file_paths
from anomaly_agent import AnomalyAgent
from utils.family_novelty import extract_test_signature, family_rotation_check
from utils.string_utils import parse_declared_fields


def write_summary(run_dir, index, name, description="", declared_family=None):
    test_dir = run_dir / f"Test_{index:02d}"
    test_dir.mkdir()
    payload = {"test_name": name, "test_description": description}
    if declared_family:
        payload["declared_family"] = declared_family
    (test_dir / "result_summary.json").write_text(json.dumps(payload), encoding="utf-8")


def make_planner_agent(tmp_path, responses, novelty_mode="lexical", caps=(7, 9)):
    class ScriptedLLM:
        def __init__(self, texts):
            self.texts = list(texts)
            self.calls = 0

        def invoke(self, prompt):
            self.calls += 1
            text = self.texts.pop(0) if self.texts else self.texts_fallback
            return AIMessage(content=text)

        texts_fallback = "VERDICT: DISTINCT\nDUPLICATE_OF: None\nREASON: differs"

    llm = ScriptedLLM(responses)
    agent = AnomalyAgent.__new__(AnomalyAgent)
    agent.allow_search_tools = False
    agent.planner_prompt_path = file_paths.planner_structured_dir
    agent.test_output_dir = tmp_path
    agent.test_config = {
        "max_searches_per_test": 0,
        "family_soft_cap": caps[0],
        "family_hard_cap": caps[1],
        "novelty_mode": novelty_mode,
    }
    agent.llm = llm
    agent.prompt_llm = lambda **kwargs: llm
    return agent, llm


def test_parse_declared_fields_normalises_labels():
    text = "TEST_NAME: X\nFAMILY: Power_Spectrum\nSTATISTIC_FORM: **ratio**\nDESCRIPTION: d"

    assert parse_declared_fields(text) == {"family": "power spectrum", "statistic_form": "ratio"}
    assert parse_declared_fields("TEST_NAME: X\nDESCRIPTION: d") == {}


def test_declared_family_overrides_keyword_inference():
    signature = extract_test_signature(
        "Threshold-Exceedance Neighbour Fraction",
        "Does not re-estimate the power spectrum or the variance.",
        declared={"family": "morphology", "statistic_form": "count"},
    )

    assert signature["families"] == ["morphology"]
    assert signature["stat_form"] == "count"
    assert signature["declared"] is True


def test_catalogue_counts_declared_families(tmp_path):
    for index in range(1, 10):
        write_summary(tmp_path, index, f"Test {index}", "mentions power spectrum", declared_family="morphology")

    issue = family_rotation_check(
        tmp_path, {"family_soft_cap": 7, "family_hard_cap": 9}, "New power test", "about power", declared={"family": "power spectrum"}
    )

    assert issue is None  # power was never a declared family, so it is not capped
    blocked = family_rotation_check(
        tmp_path, {"family_soft_cap": 7, "family_hard_cap": 9}, "Another shape test", "shapes", declared={"family": "morphology"}
    )
    assert blocked["severity"] == "blocked"


def test_planner_accepts_discouraged_family_but_rejects_blocked(tmp_path):
    for index in range(1, 8):  # seven parity tests: at the soft cap, below the hard cap
        write_summary(tmp_path, index, f"Parity test {index}", f"Parity statistic number {index}.", declared_family="parity")

    proposal = "TEST_NAME: Mirror plane extremum\nFAMILY: parity\nSTATISTIC_FORM: extremum\nDESCRIPTION: Extremum of mirror symmetry over all great-circle planes."
    agent, _ = make_planner_agent(tmp_path, [proposal])
    result = agent.planner_node({"tested_anomalies": [], "search_count": 0, "messages": []})

    assert result["node_retry"] is False
    assert result["current_test_family"] == "parity"

    for index in range(8, 10):
        write_summary(tmp_path, index, f"Parity test {index}", f"Parity statistic number {index}.", declared_family="parity")
    agent, _ = make_planner_agent(tmp_path, [proposal])
    result = agent.planner_node({"tested_anomalies": [], "search_count": 0, "messages": []})

    assert result["node_retry"] is True
    assert "REJECTED FOR FAMILY OVERUSE" in result["messages"][-1].content
    assert (tmp_path / "rejected_proposals.jsonl").exists()


def test_llm_novelty_check_rejects_judged_duplicate(tmp_path):
    write_summary(tmp_path, 1, "Maximum hemispherical variance contrast", "Maximise the variance ratio between opposite hemispheres over a direction grid.", "hemispherical")

    proposal = (
        "TEST_NAME: Fixed ecliptic hemisphere variance ratio\nFAMILY: hemispherical\nSTATISTIC_FORM: ratio\n"
        "DESCRIPTION: Ratio of pixel variance between the north and south ecliptic hemispheres."
    )
    verdict = "VERDICT: DUPLICATE\nDUPLICATE_OF: Maximum hemispherical variance contrast\nREASON: same quantity with a fixed axis."
    agent, llm = make_planner_agent(tmp_path, [proposal, verdict], novelty_mode="llm")
    result = agent.planner_node({"tested_anomalies": [], "search_count": 0, "messages": []})

    assert result["node_retry"] is True
    assert "judged substantively the same" in result["messages"][-1].content
    assert llm.calls == 2


def test_llm_novelty_check_fails_open_on_bad_verdict(tmp_path):
    write_summary(tmp_path, 1, "Prior test", "Some prior statistic.", "variance")
    proposal = "TEST_NAME: New test\nFAMILY: morphology\nSTATISTIC_FORM: count\nDESCRIPTION: Count of cold excursion components."
    agent, _ = make_planner_agent(tmp_path, [proposal, "I am not sure."], novelty_mode="llm")

    result = agent.planner_node({"tested_anomalies": [], "search_count": 0, "messages": []})

    assert result["node_retry"] is False
    assert result["current_test_name"] == "New test"


def test_structured_prompt_has_planner_variables():
    import yaml
    from langchain_core.prompts import PromptTemplate

    template = yaml.safe_load(open(file_paths.planner_structured_dir))["template"]
    assert set(PromptTemplate.from_template(template).input_variables) == {
        "search_instruction", "tested", "prior_catalog_text", "rejected_proposals_text",
        "rotation_guidance", "search_count", "search_results", "planner_feedback",
    }
    assert anomaly_agent.file_paths.novelty_judge_dir.exists()
