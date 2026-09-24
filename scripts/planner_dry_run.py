#!/usr/bin/env python3
"""Planner-only dry run: exercise test design without evaluating any map.

Runs the real planner prompt, the real family-rotation and novelty checks,
and the real model through ``AnomalyAgent.planner_node`` against a prior
catalogue seeded from a completed run, under several conditions (prompt
variant, rotation policy, search on/off). No simulation is ever loaded, so a
condition costs a handful of model calls rather than hours of map evaluation.

Example:
    python scripts/planner_dry_run.py --model xiaomi/mimo-v2.6-pro \
        --n-accept 8 --max-calls 32
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

import anomaly_agent  # noqa: E402
import file_paths  # noqa: E402
from anomaly_agent import AnomalyAgent  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from utils.family_novelty import (  # noqa: E402
    extract_test_signature,
    family_rotation_check as real_family_rotation_check,
)
from utils.string_utils import message_content_to_text, text_to_dict  # noqa: E402


# Canonical published Planck anomalies, matched against proposal name+description.
KNOWN_ANOMALIES = {
    "cold spot": r"cold[\s-]?spot|smhw|mexican[\s-]hat",
    "hemispherical asymmetry / dipole modulation": (
        r"hemispher|dipol\w*[\s-]modulat|power asymmetry|dipolar asymmetry"
    ),
    "quadrupole-octopole alignment": (
        r"quadrupole\W+(?:\w+\W+){0,4}oct[ou]pole|oct[ou]pole\W+(?:\w+\W+){0,4}quadrupole"
        r"|angular[\s-]momentum axis|planarity"
    ),
    "low quadrupole / low-l power deficit": (
        r"low[\s-]quadrupole|quadrupole (?:power|amplitude)|power deficit|lack of (?:large|low)"
    ),
    "parity asymmetry": r"\bparity\b|odd[\s-]?even|even[\s-]?odd|\bmirror\b",
    "S_1/2 / large-angle correlation": (
        r"s_?\{?1/2|\bs1/?2\b|large[\s-]angle (?:two[\s-]point|correlation)|two[\s-]point correlation"
    ),
    "ecliptic / solar-system frame": r"ecliptic|kinematic dipole|solar system|supergalactic",
    "variance deficit": r"low variance|variance deficit|deficit of variance|low[\s-]l variance",
}

REJECTION_KINDS = [
    ("REJECTED FOR FAMILY OVERUSE", "blocked"),
    ("REJECTED FOR FAMILY REPETITION", "discouraged"),
    ("REJECTED FOR REPETITION", "novelty"),
    ("REJECTED TOOL CALL", "tool_call_disabled"),
    ("REJECTED", "other_rejection"),
]

EXAMPLES_LINE_PREFIX = "- Prefer cheap single-number summaries such as"


@dataclass
class Condition:
    name: str
    prompt_path: str
    policy: str  # "current" rejects discouraged families; "fixed" only rejects blocked
    search: bool
    seed_run: str
    soft_cap: int = 7
    hard_cap: int = 9
    max_searches: int = 2


# --- thread-local rotation policy so conditions can run concurrently --------

_policy = threading.local()


def patched_family_rotation_check(*args, **kwargs):
    issue = real_family_rotation_check(*args, **kwargs)
    if (
        issue is not None
        and getattr(_policy, "mode", "current") == "fixed"
        and issue.get("severity") == "discouraged"
    ):
        return None
    return issue


anomaly_agent.family_rotation_check = patched_family_rotation_check
anomaly_agent.print = lambda *args, **kwargs: None  # silence prompt dumps


# --- model wrappers ----------------------------------------------------------


class Recorder:
    """Wrap an LLM so every planner call's raw response and usage is captured."""

    def __init__(self, llm):
        self.llm = llm
        self.last = None

    def invoke(self, prompt):
        start = time.time()
        msg = self.llm.invoke(prompt)
        usage = getattr(msg, "usage_metadata", None) or {}
        self.last = {
            "raw_text": message_content_to_text(msg.content),
            "tool_calls": [
                {"name": c.get("name"), "args": c.get("args")} for c in (getattr(msg, "tool_calls", None) or [])
            ],
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "latency_s": round(time.time() - start, 2),
        }
        return msg


class FakePlanner:
    """Deterministic stand-in for smoke-testing the harness mechanics."""

    SCRIPT = [
        "TEST_NAME: Low-l Odd-Even Parity Power Ratio\nDESCRIPTION: Ratio of even to odd multipole power for l=2..30.",
        "TEST_NAME: Preferred direction of low-l variance\nDESCRIPTION: Direction maximising hemispherical variance contrast over a direction grid.",
        "TEST_NAME: Masked skewness of smoothed field\nDESCRIPTION: Skewness of the 5 degree smoothed field over unmasked pixels.",
        "TEST_NAME: Ring-averaged gradient anisotropy\nDESCRIPTION: Ratio of gradient energy along rings of constant latitude to the total gradient energy.",
        "TEST_NAME: Nearest-neighbour rank concordance\nDESCRIPTION: Kendall concordance of pixel ranks with the mean rank of their eight neighbours.",
    ]

    def __init__(self):
        self.cycle = itertools.cycle(self.SCRIPT)

    def invoke(self, prompt):
        return AIMessage(content=next(self.cycle))


def make_llm(model: str, base_url: str, reasoning_effort: str | None, reasoning_max_tokens: int | None = None):
    import os

    from langchain_openai import ChatOpenAI

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set (environment or .env).")
    reasoning = {}
    if reasoning_effort and reasoning_effort.lower() != "none":
        reasoning["effort"] = reasoning_effort
    if reasoning_max_tokens:
        reasoning["max_tokens"] = int(reasoning_max_tokens)
    kwargs = {"reasoning": reasoning} if reasoning else {}
    return ChatOpenAI(model=model, base_url=base_url, api_key=api_key, timeout=1800, **kwargs)


# --- search emulation (same formatting as the agent's tools) -----------------


def run_search(tool_name: str, query: str) -> str:
    try:
        if tool_name == "arxiv_search":
            import arxiv

            client = arxiv.Client(page_size=3, delay_seconds=5.0, num_retries=0)
            results = list(client.results(arxiv.Search(query=query, max_results=3)))
            body = "\n\n".join(
                "\n".join(
                    [
                        "Title: " + r.title,
                        "Published: " + r.published.strftime("%Y/%m/%d"),
                        "Authors: " + ", ".join(a.name for a in r.authors[:5]),
                        "Abstract: " + r.summary,
                    ]
                )
                for r in results
            ) or "No arXiv results returned."
        else:
            from langchain_community.tools import DuckDuckGoSearchResults

            body = DuckDuckGoSearchResults(num_results=5).run(query)
        return f"Query: {query}\n\nResults:\n\n{body}\n"
    except Exception as exc:  # network failures are part of the measurement
        return (
            f"Query: {query}\n\nSearch failed:\n\n{type(exc).__name__}: {exc}\n\n"
            "The agent should proceed without these results.\n"
        )


# --- catalogue seeding --------------------------------------------------------


def seed_catalog(seed_run: Path, work_dir: Path) -> list[str]:
    names = []
    for index, summary_path in enumerate(sorted(seed_run.glob("Test_*/result_summary.json")), start=1):
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        name = str(data.get("test_name") or "").strip()
        if not name:
            continue
        target = work_dir / f"Test_{index:02d}_seed"
        target.mkdir(parents=True, exist_ok=True)
        (target / "result_summary.json").write_text(
            json.dumps(
                {
                    "saved_test_index": index,
                    "test_name": name,
                    "test_description": str(data.get("test_description") or ""),
                    "seeded_from": str(summary_path),
                }
            ),
            encoding="utf-8",
        )
        names.append(name)
    return names


def write_prompt_variant(source: Path, target: Path) -> Path:
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    lines = [line for line in data["template"].splitlines() if not line.strip().startswith(EXAMPLES_LINE_PREFIX)]
    data["template"] = "\n".join(lines) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(data, sort_keys=False, width=10_000), encoding="utf-8")
    return target


# --- one condition -------------------------------------------------------------


def make_agent(cond: Condition, work_dir: Path, planner, search_planner=None) -> AnomalyAgent:
    agent = AnomalyAgent.__new__(AnomalyAgent)
    agent.agent_mode = "planner_dry_run"
    agent.allow_search_tools = cond.search
    agent.planner_prompt_path = Path(cond.prompt_path)
    agent.test_output_dir = work_dir
    agent.test_config = {
        "max_searches_per_test": cond.max_searches if cond.search else 0,
        "family_soft_cap": cond.soft_cap,
        "family_hard_cap": cond.hard_cap,
    }
    agent.prompt_llm = lambda with_search_tools=False: (
        search_planner if (with_search_tools and cond.search and search_planner is not None) else planner
    )
    return agent


def classify_rejection(feedback: str) -> str:
    for marker, kind in REJECTION_KINDS:
        if marker in feedback:
            return kind
    return "format"


def known_anomaly_matches(text: str) -> list[str]:
    lowered = text.lower()
    return [label for label, pattern in KNOWN_ANOMALIES.items() if re.search(pattern, lowered)]


def run_condition(cond: Condition, out_dir: Path, llm_factory, n_accept: int, max_calls: int) -> dict:
    _policy.mode = cond.policy
    work_dir = out_dir / cond.name / "catalog"
    work_dir.mkdir(parents=True, exist_ok=True)
    tested = seed_catalog(Path(cond.seed_run), work_dir)
    seed_count = len(tested)

    planner = llm_factory()
    search_planner = None
    if cond.search and not isinstance(planner.llm, FakePlanner):
        search_planner = Recorder(
            planner.llm.bind_tools([AnomalyAgent.web_search, AnomalyAgent.arxiv_search], parallel_tool_calls=False)
        )
    agent = make_agent(cond, work_dir, planner, search_planner)

    state = {"tested_anomalies": list(tested), "search_count": 0, "messages": [], "search_results": []}
    calls, accepted, searches = [], [], 0
    calls_log = (out_dir / cond.name / "calls.jsonl").open("w", encoding="utf-8")

    while len(accepted) < n_accept and len(calls) < max_calls:
        result = agent.planner_node(state)
        active = search_planner if (search_planner and search_planner.last and (planner.last is None or search_planner.last is not planner.last)) else planner
        record = dict(active.last or {})
        # Prefer whichever recorder was actually invoked this call.
        for candidate in (search_planner, planner):
            if candidate is not None and candidate.last is not None:
                record = dict(candidate.last)
                candidate.last = None
                break
        record.update({"call_index": len(calls) + 1, "condition": cond.name})

        if "search_query" in result:
            msg = result["search_query"][-1]
            tool_call = (getattr(msg, "tool_calls", None) or [{}])[0]
            query = str((tool_call.get("args") or {}).get("query", ""))
            results_text = run_search(tool_call.get("name", "web_search"), query)
            state["search_results"] = list(state.get("search_results", [])) + [HumanMessage(content=results_text)]
            state["search_count"] = state.get("search_count", 0) + 1
            searches += 1
            record.update({"outcome": "search", "search_tool": tool_call.get("name"), "query": query})
        elif result.get("node_retry"):
            feedback = message_content_to_text(result["messages"][-1].content) if result.get("messages") else ""
            kind = classify_rejection(feedback) if feedback else "format"
            if feedback:
                state["messages"] = list(state.get("messages", [])) + [AIMessage(content=feedback)]
            name, description = anomaly_agent.parse_test_metadata(record.get("raw_text", ""))
            record.update(
                {
                    "outcome": "rejected",
                    "rejection": kind,
                    "feedback": feedback,
                    "test_name": name,
                    "families": extract_test_signature(name, description)["families"],
                }
            )
        else:
            name = result["current_test_name"]
            description = result["current_test_description"]
            signature = extract_test_signature(name, description)
            index = seed_count + len(accepted) + 1
            slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
            test_dir = work_dir / f"Test_{index:02d}_{slug}"
            test_dir.mkdir(parents=True, exist_ok=True)
            (test_dir / "result_summary.json").write_text(
                json.dumps({"saved_test_index": index, "test_name": name, "test_description": description}),
                encoding="utf-8",
            )
            proposal = {
                "index": index,
                "test_name": name,
                "test_description": description,
                "families": signature["families"],
                "stat_form": signature["stat_form"],
                "components": signature["components"],
                "tokens": signature["tokens"],
                "known_anomaly_matches": known_anomaly_matches(f"{name} {description}"),
                "calls_to_accept": len(calls) + 1 - sum(1 for c in calls if c.get("outcome") == "accepted") - sum(len(a.get("_calls", [])) for a in accepted),
            }
            accepted.append(proposal)
            state = {
                "tested_anomalies": list(state["tested_anomalies"]) + [name],
                "search_count": 0,
                "messages": [],
                "search_results": [],
            }
            record.update({"outcome": "accepted", "test_name": name, "families": signature["families"]})

        calls.append(record)
        calls_log.write(json.dumps(record, default=str) + "\n")
        calls_log.flush()

    calls_log.close()
    # calls between consecutive acceptances
    boundaries = [i for i, c in enumerate(calls) if c.get("outcome") == "accepted"]
    per_accept = [b - (boundaries[k - 1] if k else -1) for k, b in enumerate(boundaries)]
    for proposal, n in zip(accepted, per_accept):
        proposal["calls_to_accept"] = n

    summary = summarize(cond, calls, accepted, searches, seed_count)
    (out_dir / cond.name / "proposals.json").write_text(json.dumps(accepted, indent=2), encoding="utf-8")
    (out_dir / cond.name / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def pairwise_jaccard(token_sets: list[set]) -> float | None:
    pairs = list(itertools.combinations(token_sets, 2))
    if not pairs:
        return None
    return sum(len(a & b) / (len(a | b) or 1) for a, b in pairs) / len(pairs)


def summarize(cond: Condition, calls: list[dict], accepted: list[dict], searches: int, seed_count: int) -> dict:
    rejections = {}
    for c in calls:
        if c.get("outcome") == "rejected":
            rejections[c["rejection"]] = rejections.get(c["rejection"], 0) + 1
    families = {}
    for p in accepted:
        for f in p["families"]:
            families[f] = families.get(f, 0) + 1
    in_tokens = [c["input_tokens"] for c in calls if c.get("input_tokens")]
    out_tokens = [c["output_tokens"] for c in calls if c.get("output_tokens")]
    latencies = [c["latency_s"] for c in calls if c.get("latency_s") is not None]
    n_acc = len(accepted)
    return {
        "condition": asdict(cond),
        "seed_tests": seed_count,
        "planner_calls": len(calls),
        "searches": searches,
        "accepted": n_acc,
        "rejections": rejections,
        "calls_per_accepted": round(len(calls) / n_acc, 2) if n_acc else None,
        "first_try_acceptance_rate": round(sum(1 for p in accepted if p["calls_to_accept"] == 1) / n_acc, 2) if n_acc else None,
        "families_of_accepted": dict(sorted(families.items(), key=lambda kv: -kv[1])),
        "fraction_other_family": round(sum(1 for p in accepted if p["families"] == ["other"]) / n_acc, 2) if n_acc else None,
        "fraction_matching_known_anomaly": round(sum(1 for p in accepted if p["known_anomaly_matches"]) / n_acc, 2) if n_acc else None,
        "known_anomaly_hits": sorted({m for p in accepted for m in p["known_anomaly_matches"]}),
        "mean_pairwise_jaccard": round(pairwise_jaccard([set(p["tokens"]) for p in accepted]), 3) if n_acc > 1 else None,
        "mean_input_tokens": round(sum(in_tokens) / len(in_tokens)) if in_tokens else None,
        "mean_output_tokens": round(sum(out_tokens) / len(out_tokens)) if out_tokens else None,
        "mean_latency_s": round(sum(latencies) / len(latencies), 1) if latencies else None,
        "accepted_names": [p["test_name"] for p in accepted],
    }


def aggregate_chains(base: Condition, chain_summaries: list[dict], out_dir: Path) -> dict:
    """Merge independent chains of one condition into a single summary."""
    calls, accepted, searches, seed_count = [], [], 0, 0
    for summary in chain_summaries:
        chain_dir = out_dir / summary["condition"]["name"]
        calls.extend(json.loads(line) for line in (chain_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines() if line)
        accepted.extend(json.loads((chain_dir / "proposals.json").read_text(encoding="utf-8")))
        searches += summary["searches"]
        seed_count = summary["seed_tests"]
    merged = summarize(base, calls, accepted, searches, seed_count)
    merged["chains"] = len(chain_summaries)
    (out_dir / base.name).mkdir(parents=True, exist_ok=True)
    (out_dir / base.name / "proposals.json").write_text(json.dumps(accepted, indent=2), encoding="utf-8")
    (out_dir / base.name / "summary.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return merged


# --- optional literature judge ------------------------------------------------

JUDGE_TEMPLATE = """You are an expert on the published literature of CMB temperature anomalies and isotropy tests (Planck 2013/2015/2018 isotropy and statistics papers, WMAP anomaly papers, and follow-ups).

Classify the proposed test statistic below relative to that literature:
- REPEAT: essentially a published statistic (same object, same summary), possibly with trivial parameter changes.
- VARIATION: a recognisable modification of a published statistic or a published family (different scale, region, weighting, or summary).
- NOVEL: no close published analogue that you know of.

Be strict: if you are unsure whether something has been published, say VARIATION, not NOVEL.

PROPOSED TEST:
TEST_NAME: {name}
DESCRIPTION: {description}

Answer in exactly this format and nothing else:
LABEL: <REPEAT, VARIATION, or NOVEL>
CLOSEST: <closest published statistic or family, or None>
REASON: <one sentence>
"""


def judge_proposals(out_dir: Path, summaries: list[dict], judge_llm) -> None:
    for summary in summaries:
        cond_dir = out_dir / summary["condition"]["name"]
        proposals = json.loads((cond_dir / "proposals.json").read_text(encoding="utf-8"))
        labels = {}

        def judge_one(p):
            try:
                msg = judge_llm.invoke(JUDGE_TEMPLATE.format(name=p["test_name"], description=p["test_description"][:3000]))
                parsed = text_to_dict(message_content_to_text(msg.content), ["LABEL", "CLOSEST", "REASON"])
                label = parsed["LABEL"].strip().upper().split()[0] if parsed["LABEL"].strip() else "UNPARSED"
                return {"label": label, "closest": parsed["CLOSEST"].strip(), "reason": parsed["REASON"].strip()}
            except Exception as exc:  # keep the report even if one judge call fails
                return {"label": "ERROR", "closest": "", "reason": f"{type(exc).__name__}: {exc}"[:300]}

        with ThreadPoolExecutor(max_workers=max(1, min(16, len(proposals)))) as pool:
            verdicts = list(pool.map(judge_one, proposals))
        for p, verdict in zip(proposals, verdicts):
            p["judge"] = verdict
            labels[verdict["label"]] = labels.get(verdict["label"], 0) + 1
        summary["judge_labels"] = labels
        (cond_dir / "proposals.json").write_text(json.dumps(proposals, indent=2), encoding="utf-8")
        (cond_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


# --- reporting -------------------------------------------------------------------


def render_table(summaries: list[dict]) -> str:
    cols = [
        ("condition", lambda s: s["condition"]["name"]),
        ("calls", lambda s: s["planner_calls"]),
        ("accepted", lambda s: s["accepted"]),
        ("calls/accept", lambda s: s["calls_per_accepted"]),
        ("rejections", lambda s: ", ".join(f"{k}={v}" for k, v in sorted(s["rejections"].items())) or "none"),
        ("known-anomaly frac", lambda s: s["fraction_matching_known_anomaly"]),
        ("other-family frac", lambda s: s["fraction_other_family"]),
        ("pairwise Jaccard", lambda s: s["mean_pairwise_jaccard"]),
        ("judge", lambda s: ", ".join(f"{k}={v}" for k, v in sorted(s.get("judge_labels", {}).items())) or "-"),
    ]
    header = "| " + " | ".join(c for c, _ in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    rows = ["| " + " | ".join(str(fn(s)) for _, fn in cols) + " |" for s in summaries]
    return "\n".join([header, sep, *rows])


def default_conditions(out_dir: Path, seed_run: str, blind_seed_run: str) -> list[Condition]:
    no_examples = write_prompt_variant(Path(file_paths.planner_dir), out_dir / "prompts" / "planner_no_examples.yaml")
    return [
        Condition("baseline", str(file_paths.planner_dir), "current", False, seed_run),
        Condition("softcap_fix", str(file_paths.planner_dir), "fixed", False, seed_run),
        Condition("no_examples", str(no_examples), "fixed", False, seed_run),
        Condition("search_on", str(file_paths.planner_dir), "fixed", True, seed_run),
        Condition("blind_current", str(file_paths.blind_planner_dir), "current", False, blind_seed_run),
        Condition("blind_fix", str(file_paths.blind_planner_dir), "fixed", False, blind_seed_run),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Planner-only dry run for test-design experiments.")
    parser.add_argument("--model", default="xiaomi/mimo-v2.6-pro")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--reasoning-effort", default=None, help="Passed as reasoning.effort when set.")
    parser.add_argument(
        "--reasoning-max-tokens",
        type=int,
        default=None,
        help="Cap on reasoning tokens per call (OpenRouter reasoning.max_tokens). Strongly recommended for MiMo.",
    )
    parser.add_argument("--seed-run", default="data/output/anomaly_agent/agent_10k_run_001")
    parser.add_argument("--blind-seed-run", default="data/output/anomaly_agent/blind_10k_run_001")
    parser.add_argument("--n-accept", type=int, default=8, help="Accepted proposals to collect per condition.")
    parser.add_argument("--max-calls", type=int, default=32, help="Planner-call budget per condition.")
    parser.add_argument("--conditions", default=None, help="Comma-separated subset of condition names.")
    parser.add_argument("--judge-model", default=None, help="Model for literature novelty labels; 'none' to skip. Defaults to --model.")
    parser.add_argument("--output", default=None, help="Output directory. Defaults to data/output/planner_dry_run/<timestamp>.")
    parser.add_argument("--fake", action="store_true", help="Use a deterministic fake planner to smoke-test the harness.")
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Start every condition from an empty catalogue instead of seeding from completed runs.",
    )
    parser.add_argument("--workers", type=int, default=None, help="Concurrent chains. Defaults to conditions x chains.")
    parser.add_argument(
        "--chains",
        type=int,
        default=1,
        help="Independent planner chains per condition, run concurrently from the same seed and merged.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output or f"data/output/planner_dry_run/{time.strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.from_scratch:
        empty_seed = out_dir / "empty_seed"
        empty_seed.mkdir(parents=True, exist_ok=True)
        args.seed_run = args.blind_seed_run = str(empty_seed)
    conditions = default_conditions(out_dir, args.seed_run, args.blind_seed_run)
    if args.conditions:
        wanted = {c.strip() for c in args.conditions.split(",")}
        conditions = [c for c in conditions if c.name in wanted]
        missing = wanted - {c.name for c in conditions}
        if missing:
            parser.error(f"Unknown condition(s): {', '.join(sorted(missing))}")

    if args.fake:
        llm_factory = lambda: Recorder(FakePlanner())  # noqa: E731
    else:
        shared_llm = make_llm(args.model, args.base_url, args.reasoning_effort, args.reasoning_max_tokens)
        llm_factory = lambda: Recorder(shared_llm)  # noqa: E731

    (out_dir / "run_config.json").write_text(
        json.dumps({**vars(args), "conditions": [asdict(c) for c in conditions]}, indent=2), encoding="utf-8"
    )

    chains = max(1, args.chains)
    jobs = [
        (cond, replace(cond, name=f"{cond.name}/chain{k + 1}") if chains > 1 else cond)
        for cond in conditions
        for k in range(chains)
    ]
    workers = args.workers or len(jobs)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        chain_summaries = list(
            pool.map(lambda job: run_condition(job[1], out_dir, llm_factory, args.n_accept, args.max_calls), jobs)
        )
    if chains > 1:
        summaries = [
            aggregate_chains(cond, [s for (base, _), s in zip(jobs, chain_summaries) if base is cond], out_dir)
            for cond in conditions
        ]
    else:
        summaries = chain_summaries

    judge_model = args.judge_model or args.model
    if not args.fake and judge_model.lower() != "none":
        judge_proposals(
            out_dir, summaries, make_llm(judge_model, args.base_url, args.reasoning_effort, args.reasoning_max_tokens)
        )

    (out_dir / "comparison.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    table = render_table(summaries)
    (out_dir / "comparison.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
