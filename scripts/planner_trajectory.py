#!/usr/bin/env python3
"""Trajectory view of a planner dry run: how a catalogue's diversity evolves as it grows.

Reads a ``scripts/planner_dry_run.py`` output directory and, for every condition,
walks the accepted proposals in order, reporting per-window statistics (calls per
accepted test, family coverage, similarity to the closest earlier test, canonical
anomaly overlap, judge labels) and writing a per-proposal CSV next to the data.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.family_novelty import extract_test_signature, jaccard_overlap  # noqa: E402


def similarity(candidate: dict, prior: dict) -> float:
    """Same score as utils.family_novelty.novelty_check, without the threshold."""
    tokens_c, tokens_p = set(candidate["tokens"]), set(prior["tokens"])
    score = len(tokens_c & tokens_p) / (len(tokens_c | tokens_p) or 1)
    score += 0.20 * jaccard_overlap(candidate["families"], prior["families"])
    if candidate["stat_form"] == prior["stat_form"]:
        score += 0.15
    score += 0.25 * jaccard_overlap(candidate["components"], prior["components"])
    return score


def load_condition(cond_dir: Path) -> tuple[list[dict], list[dict]]:
    calls = [json.loads(line) for line in (cond_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    proposals = json.loads((cond_dir / "proposals.json").read_text(encoding="utf-8"))
    return calls, proposals


def trajectory(calls: list[dict], proposals: list[dict]) -> list[dict]:
    rows, priors, since = [], [], 0
    accepted_iter = iter(proposals)
    for call in calls:
        since += 1
        if call.get("outcome") != "accepted":
            continue
        proposal = next(accepted_iter)
        declared = {
            "family": proposal.get("declared_family", ""),
            "statistic_form": proposal.get("declared_stat_form", ""),
        }
        signature = extract_test_signature(proposal["test_name"], proposal["test_description"], declared=declared)
        closest = max(((similarity(signature, p), p["name"]) for p in priors), default=(0.0, ""))
        rows.append(
            {
                "n": len(rows) + 1,
                "test_name": proposal["test_name"],
                "calls_to_accept": since,
                "families": "+".join(signature["families"]),
                "stat_form": signature["stat_form"],
                "closest_prior_score": round(closest[0], 3),
                "closest_prior": closest[1],
                "known_anomaly": "; ".join(proposal.get("known_anomaly_matches") or []),
                "judge": (proposal.get("judge") or {}).get("label", ""),
                "duplicate": (proposal.get("dup_judge") or {}).get("verdict", "") == "DUPLICATE",
                "duplicate_of": (proposal.get("dup_judge") or {}).get("duplicate_of", ""),
                "output_tokens": call.get("output_tokens"),
            }
        )
        priors.append({**signature, "name": proposal["test_name"]})
        since = 0
    return rows


def window_summary(rows: list[dict], proposals: list[dict], width: int) -> list[dict]:
    out = []
    for start in range(0, len(rows), width):
        chunk = rows[start : start + width]
        sigs = [
            extract_test_signature(
                p["test_name"],
                p["test_description"],
                declared={"family": p.get("declared_family", ""), "statistic_form": p.get("declared_stat_form", "")},
            )
            for p in proposals[start : start + width]
        ]
        pairs = list(itertools.combinations([set(s["tokens"]) for s in sigs], 2))
        families = {f for s in sigs for f in s["families"]}
        out.append(
            {
                "tests": f"{start + 1}-{start + len(chunk)}",
                "calls_per_accept": round(sum(r["calls_to_accept"] for r in chunk) / len(chunk), 2),
                "distinct_families": len(families - {"other"}),
                "other_frac": round(sum(1 for s in sigs if s["families"] == ["other"]) / len(sigs), 2),
                "mean_closest_score": round(sum(r["closest_prior_score"] for r in chunk) / len(chunk), 3),
                "max_closest_score": round(max(r["closest_prior_score"] for r in chunk), 3),
                "known_anomaly_frac": round(sum(1 for r in chunk if r["known_anomaly"]) / len(chunk), 2),
                "duplicate_frac": round(sum(1 for r in chunk if r["duplicate"]) / len(chunk), 2),
                "pairwise_jaccard": round(sum(len(a & b) / (len(a | b) or 1) for a, b in pairs) / len(pairs), 3) if pairs else None,
                "judge": ", ".join(f"{k}={v}" for k, v in sorted(
                    {lab: sum(1 for r in chunk if r["judge"] == lab) for lab in {r["judge"] for r in chunk} if lab}.items()
                )) or "-",
            }
        )
    return out


def render(rows: list[dict], cols: list[str]) -> str:
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def main() -> None:
    parser = argparse.ArgumentParser(description="Trajectory analysis of a planner dry run.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--names", action="store_true", help="Also print every accepted test name in order.")
    args = parser.parse_args()

    for cond_dir in sorted(p for p in args.run_dir.iterdir() if (p / "proposals.json").exists()):
        calls, proposals = load_condition(cond_dir)
        rows = trajectory(calls, proposals)
        if not rows:
            continue
        with (cond_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        rejections = sum(1 for c in calls if c.get("outcome") == "rejected")
        print(f"\n## {cond_dir.name}: {len(rows)} accepted in {len(calls)} calls ({rejections} rejections)\n")
        print(render(window_summary(rows, proposals, args.window), [
            "tests", "calls_per_accept", "distinct_families", "other_frac", "mean_closest_score",
            "max_closest_score", "known_anomaly_frac", "duplicate_frac", "pairwise_jaccard", "judge",
        ]))
        if args.names:
            print()
            print(render(rows, ["n", "test_name", "families", "stat_form", "closest_prior_score", "known_anomaly", "judge", "duplicate_of"]))


if __name__ == "__main__":
    main()
