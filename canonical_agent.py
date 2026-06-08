"""Canonical CMB anomaly agent.

This module reuses the exploratory AnomalyAgent execution workflow but replaces
the planner with one that researches and specifies a named published anomaly.
"""

import argparse
import json
import re
from pathlib import Path

import file_paths
import yaml
from langgraph.types import RetryPolicy
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate

from anomaly_agent import (
    AnomalyAgent,
    load_runtime_configs,
    normalize_optional_config_value,
)
from utils.string_utils import message_content_to_text, parse_test_metadata, text_to_dict


class CanonicalAnomalyAgent(AnomalyAgent):
    """Agent variant for implementing one specified canonical CMB anomaly."""

    def __init__(
        self,
        canonical_anomaly: str,
        *args,
        canonical_planner_path: str | Path | None = None,
        canonical_review_path: str | Path | None = None,
        **kwargs,
    ):
        anomaly = str(canonical_anomaly).strip()
        if not anomaly:
            raise ValueError("canonical_anomaly must be a non-empty string")

        super().__init__(*args, **kwargs)
        self.canonical_anomaly = anomaly
        self.canonical_planner_path = (
            Path(canonical_planner_path)
            if canonical_planner_path
            else file_paths.canonical_planner_dir
        )
        self.canonical_review_path = (
            Path(canonical_review_path)
            if canonical_review_path
            else file_paths.canonical_review_dir
        )

    def planner_node(self, state: AnomalyAgent.State):
        tested_anomalies = state.get("tested_anomalies", [])
        tested = tested_anomalies if tested_anomalies else "None"
        search_count = state.get("search_count", 0)
        max_searches = self.test_config["max_searches_per_test"]
        last_message_text = (
            message_content_to_text(state["messages"][-1].content)
            if state.get("messages")
            else ""
        )
        planner_feedback = last_message_text if "REJECTED" in last_message_text else ""
        search_results = self.retrieve_state(state, "search_results", max_entries=5)

        if max_searches <= 0 or search_count >= max_searches:
            search_instruction = "- Do NOT call any search tools."
        elif search_count <= 0:
            search_instruction = (
                "- You MUST call web_search or arxiv_search before proposing the "
                "canonical implementation. Make only ONE tool call at a time."
            )
        else:
            search_instruction = (
                "- You MAY call web_search or arxiv_search if more source detail "
                "is needed. Make only ONE tool call at a time. Otherwise provide "
                "the final canonical implementation specification."
            )

        with self.canonical_planner_path.open("r", encoding="utf-8") as stream:
            prompt_config = yaml.safe_load(stream)
            template = prompt_config["template"]

        prompt = PromptTemplate.from_template(template).format_prompt(
            canonical_anomaly=self.canonical_anomaly,
            search_instruction=search_instruction,
            search_count=search_count,
            max_searches=max_searches,
            tested=tested,
            search_results=search_results,
            planner_feedback=planner_feedback,
        )

        print("\n##### PROMPT #####\n")
        print(prompt.to_string())

        msg = self.search_llm.invoke(prompt)

        if getattr(msg, "tool_calls", None):
            if search_count >= max_searches:
                return {
                    "messages": [
                        AIMessage(
                            content=(
                                "REJECTED TOOL CALL:\n"
                                "- The search budget has been exhausted. Provide "
                                "the canonical implementation specification using "
                                "the search results already available."
                            )
                        )
                    ],
                    "node_retry": True,
                }
            return {"search_query": [msg]}

        msg_text = message_content_to_text(msg.content)
        invalid_response = (
            (not msg_text.strip())
            or ("TEST_NAME" not in msg_text)
            or ("DESCRIPTION" not in msg_text)
            or getattr(msg, "invalid_tool_calls", None)
        )
        if invalid_response:
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "REJECTED FORMAT:\n"
                            "- Provide exactly one canonical anomaly statistic "
                            "using the required TEST_NAME and DESCRIPTION fields."
                        )
                    )
                ],
                "node_retry": True,
            }

        test_name, test_description = parse_test_metadata(msg_text)
        return {
            "messages": [msg],
            "current_test_name": test_name,
            "current_test_description": test_description,
            "search_count": 0,
            "node_retry": False,
        }

    def canonical_review_rejection_count(self, state: AnomalyAgent.State) -> int:
        return len(self.canonical_review_feedbacks(state))

    def canonical_review_feedbacks(self, state: AnomalyAgent.State) -> list[str]:
        feedback = []
        for msg in state.get("messages", []):
            text = message_content_to_text(getattr(msg, "content", msg))
            if "CANONICAL REVIEW REJECTED" in text:
                feedback.append(text)
        return feedback

    def canonical_review_node(self, state: AnomalyAgent.State):
        test_name = state["current_test_name"]
        test_description = state["current_test_description"]
        code = self.retrieve_state(state, "code", max_entries=1)
        test_hypothesis = self.retrieve_state(state, "test_hypothesis", max_entries=1)
        test_type = self.retrieve_state(state, "test_type", max_entries=1)
        justification = self.retrieve_state(state, "justification", max_entries=1)

        result_payload = dict(state.get("current_results", {}) or {})
        for key in ("analysis_code", "plot_path", "plot_pdf_path", "output_dir"):
            result_payload.pop(key, None)
        result_payload_text = json.dumps(
            self.to_python_types(result_payload),
            indent=2,
            default=str,
        )

        review_feedbacks = self.canonical_review_feedbacks(state)
        previous_review_feedback = review_feedbacks[-1] if review_feedbacks else ""

        with self.canonical_review_path.open("r", encoding="utf-8") as stream:
            prompt_config = yaml.safe_load(stream)
            template = prompt_config["template"]

        prompt = PromptTemplate.from_template(template).format_prompt(
            canonical_anomaly=self.canonical_anomaly,
            test_name=test_name,
            test_description=test_description,
            test_hypothesis=test_hypothesis,
            test_type=test_type,
            justification=justification,
            analysis_code=code,
            result_payload=result_payload_text,
            previous_review_feedback=previous_review_feedback,
        )

        print("\n##### PROMPT #####\n")
        print(prompt.to_string())

        msg = self.llm.invoke(prompt)
        msg_text = message_content_to_text(msg.content)
        review = text_to_dict(msg_text, ["VERDICT", "REASON", "REVISION_GUIDANCE"])
        verdict = review["VERDICT"].strip().lower()
        reason = review["REASON"].strip()
        guidance = review["REVISION_GUIDANCE"].strip()

        current_results = dict(state.get("current_results", {}) or {})
        review_record = {
            "verdict": verdict or "invalid",
            "reason": reason,
            "revision_guidance": guidance,
        }

        if verdict.startswith("accept"):
            review_record["accepted"] = True
            current_results["canonical_review"] = review_record
            self.python_env["last_error"] = None
            return {
                "messages": [msg],
                "current_results": current_results,
                "node_retry": False,
            }

        if not verdict.startswith("revise"):
            review_record["accepted"] = True
            review_record["warning"] = "Review response did not use ACCEPT or REVISE."
            current_results["canonical_review"] = review_record
            self.python_env["last_error"] = None
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "CANONICAL REVIEW ACCEPTED WITH WARNING:\n"
                            "- Review response did not use ACCEPT or REVISE."
                        )
                    )
                ],
                "current_results": current_results,
                "node_retry": False,
            }

        max_revisions = int(self.test_config.get("max_canonical_review_revisions", 2))
        rejection_count = self.canonical_review_rejection_count(state)
        if rejection_count >= max_revisions:
            review_record["accepted"] = True
            review_record["warning"] = (
                "Accepted after reaching max_canonical_review_revisions."
            )
            current_results["canonical_review"] = review_record
            self.python_env["last_error"] = None
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "CANONICAL REVIEW ACCEPTED WITH WARNING:\n"
                            f"- Reached max_canonical_review_revisions={max_revisions}.\n"
                            f"- Last review reason: {reason}"
                        )
                    )
                ],
                "current_results": current_results,
                "node_retry": False,
            }

        feedback = (
            "CANONICAL REVIEW REJECTED:\n"
            f"- {reason}\n\n"
            "REVISION_GUIDANCE:\n"
            f"{guidance or 'Revise the implementation to better match the canonical specification.'}"
        )
        review_record["accepted"] = False
        current_results["canonical_review"] = review_record
        self.python_env["last_error"] = feedback
        return {
            "messages": [AIMessage(content=feedback)],
            "current_results": current_results,
            "python_env": {"last_error": feedback},
            "node_retry": True,
        }

    def execute_route(self, state: AnomalyAgent.State):
        if state.get("node_retry") == True:
            return "implement"
        else:
            return "canonical_review"

    def execute_route_options(self):
        return {"implement": "implement", "canonical_review": "canonical_review"}

    def canonical_review_route(self, state: AnomalyAgent.State):
        if state.get("node_retry") == True:
            return "implement"
        return "summary"

    def add_extra_workflow_nodes(self, workflow):
        workflow.add_node(
            "canonical_review",
            self.canonical_review_node,
            retry_policy=RetryPolicy(max_attempts=2, initial_interval=5),
        )
        workflow.add_conditional_edges(
            "canonical_review",
            self.canonical_review_route,
            {"implement": "implement", "summary": "summary"},
        )


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "canonical_anomaly"


def main():
    parser = argparse.ArgumentParser(
        description="Run the canonical CMB anomaly agent for a named anomaly."
    )
    parser.add_argument(
        "anomaly",
        help="Named canonical anomaly to implement, e.g. 'cold spot'.",
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
        thread_id = agent_config.get("thread_id", f"canonical_{slugify(args.anomaly)}")
    else:
        thread_id = f"canonical_{slugify(args.anomaly)}"

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

    agent = CanonicalAnomalyAgent(
        canonical_anomaly=args.anomaly,
        model=model,
        thread_id=thread_id,
        base_url=base_url,
        reasoning_effort=reasoning_effort,
        sim_maps_path=sim_maps_path,
        test_config=runtime_configs["test"],
        plot_config=runtime_configs["plot"],
    )
    agent()


if __name__ == "__main__":
    main()
