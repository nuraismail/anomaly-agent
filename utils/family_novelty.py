import file_paths
import yaml
import json
import re
from pathlib import Path


def normalise_family(family: str) -> str:
    if family == "correlation":
        return "cross correlation"
    return family


def normalise_stat_form(stat_form: str) -> str:
    aliases = {
        "contrast": "difference",
        "coefficent": "coefficient",
    }
    return aliases.get(stat_form, stat_form)


def jaccard_overlap(left, right) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 0.0
    return len(left_set & right_set) / (len(left_set | right_set) or 1)


def family_labels(entry: dict) -> list[str]:
    families = entry.get("families")
    if families:
        return list(families)

    family = entry.get("family", "other")
    return [family] if family else ["other"]


def extract_test_signature(name: str, description: str):
    text = f"{name} {description}".lower()
    matched_families = []
    stat_form = "other"

    with open(file_paths.cmb_dict_dir) as stream:
        cmb_dict = yaml.safe_load(stream)
        families = cmb_dict["families"]

        for f in families:
            if re.search(rf"\b{re.escape(f)}\b", text):
                matched_families.append(normalise_family(f))

        stat_forms = list(cmb_dict["stat_forms"])
        if "coefficent" not in stat_forms:
            stat_forms.append("coefficent")

        for s in stat_forms:
            if re.search(rf"\b{re.escape(s)}\b", text):
                stat_form = normalise_stat_form(s)
                break

        comps = cmb_dict["comps"]
        components = []

        for c in comps:
            if re.search(rf"\b{re.escape(c)}\b", text):
                components.append(c)

    stopwords = cmb_dict["stopwords"]

    parts = re.findall(r"[a-z0-9]+", text.lower())
    tokens = [part for part in parts if part not in stopwords and len(part) > 2]
    families = matched_families or ["other"]
    
    return {
        "family": families[0],
        "families": families,
        "stat_form": stat_form,
        "components": components,
        "tokens": tokens,
    }


def build_prior_catalog(tests_dir):
    catalog = []

    for out_dir in sorted(Path(tests_dir).glob("*/result_summary.json")):
        try:
            with open(out_dir, 'r', encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError):
            continue

        if not isinstance(data, dict):
            continue

        name = str(data.get("test_name", "")).strip()
        description = str(data.get("test_description", "")).strip()

        if not name:
            summary = data.get("test_summary", {})
            if isinstance(summary, dict):
                name = str(summary.get("Test name", "")).strip()
                description = description or str(summary.get("Description", "")).strip()

        if not name:
            continue

        signature = extract_test_signature(name, description)

        catalog.append({
            "name": name,
            "family": signature["family"],
            "families": signature["families"],
            "stat_form": signature["stat_form"],
            "components": signature["components"],
            "tokens": signature["tokens"],
        })

    return catalog 


def compact_catalog_text(tests_dir, max_items: int = 12):
    catalog = build_prior_catalog(tests_dir)

    if not catalog:
        return "None"

    family_counts = {}

    for entry in catalog:
        for family in family_labels(entry):
            family_counts[family] = family_counts.get(family, 0) + 1

    lines = ["Family counts:"]

    for family, count in sorted(family_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- {family}: {count}")

    lines.append("")
    lines.append("Recent tests:")
    
    for entry in catalog[-max_items:]:
        comp = ", ".join(entry["components"]) if entry["components"] else "none"
        families = ", ".join(family_labels(entry))
        lines.append(
            f"- {entry['name']} | families={families} | form={entry['stat_form']} | components={comp}"
        )

    return "\n".join(lines)


def build_rejected_proposal_catalog(tests_dir):
    rejected_path = Path(tests_dir) / "rejected_proposals.jsonl"
    if not rejected_path.exists():
        return []

    catalog = []
    with rejected_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                catalog.append(data)

    return catalog


def compact_rejected_proposals_text(tests_dir, max_items: int = 5):
    catalog = build_rejected_proposal_catalog(tests_dir)
    if not catalog:
        return "None"

    lines = []
    for entry in catalog[-max_items:]:
        reasons = []
        rotation_issue = entry.get("rotation_issue") or {}
        novelty_issue = entry.get("novelty_issue") or {}

        if rotation_issue:
            reasons.append(f"rotation={rotation_issue.get('severity', 'yes')}")
        if novelty_issue:
            reasons.append(
                f"similar_to={novelty_issue.get('prior_name', 'prior')} "
                f"score={float(novelty_issue.get('score') or 0.0):.2f}"
            )

        reason_text = "; ".join(reasons) if reasons else "rejected"
        lines.append(f"- {entry.get('test_name', 'unknown')} | {reason_text}")

    return "\n".join(lines)


def blocked_families(tests_dir, test_config:dict):
    counts = {}

    for entry in build_prior_catalog(tests_dir):
        for family in family_labels(entry):
            counts[family] = counts.get(family, 0) + 1

    blocked = sorted([family for family, count in counts.items() if count >= test_config["family_hard_cap"]])
    discouraged = sorted([family for family, count in counts.items() if test_config["family_soft_cap"] <= count < test_config["family_hard_cap"]])
    
    return blocked, discouraged


def rotation_guidance_text(tests_dir, test_config:dict):
    blocked, discouraged = blocked_families(tests_dir, test_config)
    blocked = [family for family in blocked if family != "other"]
    discouraged = [family for family in discouraged if family != "other"]

    parts = []
    
    if blocked:
        parts.append(
            "Temporarily blocked families because they are overused: "
            + ", ".join(blocked)
            + ". Do not propose tests from these families unless no other scientifically valid family remains."
        )

    if discouraged:
        parts.append(
            "Families to avoid unless the proposal is clearly distinct: "
            + ", ".join(discouraged)
            + "."
        )

    if not parts:
        return "No family blocks are active."
    else:
        return "\n".join(parts)

def novelty_check(tests_dir, candidate_name: str, candidate_description: str):
    candidate = extract_test_signature(candidate_name, candidate_description)
    prior_catalog = build_prior_catalog(tests_dir)

    if not prior_catalog:
        return None

    candidate_tokens = set(candidate["tokens"])
    candidate_families = set(candidate["families"])
    candidate_components = set(candidate["components"])
    best = None
    best_score = -1.0

    for prior in prior_catalog:
        prior_tokens = set(prior["tokens"])
        overlap = len(candidate_tokens & prior_tokens)
        union = len(candidate_tokens | prior_tokens) or 1
        jaccard = overlap / union

        prior_families = set(family_labels(prior))
        prior_components = set(prior["components"])
        family_overlap = jaccard_overlap(candidate_families, prior_families)
        component_overlap = jaccard_overlap(candidate_components, prior_components)
        same_family = family_overlap > 0.0
        same_form = candidate["stat_form"] == prior["stat_form"]
        same_components = component_overlap == 1.0 and bool(candidate_components)

        score = jaccard

        score += 0.20 * family_overlap

        if same_form:
            score += 0.15

        score += 0.25 * component_overlap

        if score > best_score:
            best_score = score
            best = {
                "prior_name": prior["name"],
                "score": score,
                "jaccard": jaccard,
                "family_overlap": family_overlap,
                "component_overlap": component_overlap,
                "shared_families": sorted(candidate_families & prior_families),
                "same_family": same_family,
                "same_form": same_form,
                "same_components": same_components,
                "family": prior["family"],
                "families": family_labels(prior),
            }

    if best and (best["score"] >= 0.75 or (best["same_family"] and best["jaccard"] >= 0.45)):
        return best
    else:
        return None

def family_rotation_check(tests_dir, test_config:dict, candidate_name: str, candidate_description: str):
    signature = extract_test_signature(candidate_name, candidate_description)
    blocked, discouraged = blocked_families(tests_dir, test_config)
    candidate_families = set(signature["families"]) - {"other"}
    blocked_families_for_candidate = sorted(candidate_families & set(blocked))
    discouraged_families_for_candidate = sorted(candidate_families & set(discouraged))

    if blocked_families_for_candidate:
        blocked = [family for family in blocked if family != "other"]
        return {
            "family": signature["family"],
            "families": signature["families"],
            "matched_families": blocked_families_for_candidate,
            "severity": "blocked",
            "message": f"The following anomaly families have reached the hard cap and are temporarily blocked: {str(blocked)}"
        }
    elif discouraged_families_for_candidate:
        discouraged = [family for family in discouraged if family != "other"]
        return {
            "family": signature["family"],
            "families": signature["families"],
            "matched_families": discouraged_families_for_candidate,
            "severity": "discouraged",
            "message": f"The following anomaly families are already heavily used and should be avoided unless the test is genuinely distinct: {str(discouraged)}"
        }
    else:
        return None
