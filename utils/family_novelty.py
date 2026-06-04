import file_paths
import yaml
import json
import re
from pathlib import Path

def extract_test_signature(name: str, description: str):
    text = f"{name} {description}".lower()
    family = "other"
    stat_form = "other"

    with open(file_paths.cmb_dict_dir) as stream:
        cmb_dict = yaml.safe_load(stream)
        families = cmb_dict["families"]

        for f in families:
            if f in text:
                if f == "correlation":
                    family = "cross correlation"
                else:
                    family = f

        stat_forms = cmb_dict["stat_forms"]

        for s in stat_forms:
            if s in text:
                if s == "contrast":
                    stat_form = "difference"
                else:
                    stat_form = s

        comps = cmb_dict["comps"]
        components = []

        for c in comps:
            if c in text:
                components.append(c)

    stopwords = cmb_dict["stopwords"]

    parts = re.findall(r"[a-z0-9]+", text.lower())
    tokens = [part for part in parts if part not in stopwords and len(part) > 2]
    
    return {
        "family": family,
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
        family_counts[entry["family"]] = family_counts.get(entry["family"], 0) + 1

    lines = ["Family counts:"]

    for family, count in sorted(family_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- {family}: {count}")

    lines.append("")
    lines.append("Recent tests:")
    
    for entry in catalog[-max_items:]:
        comp = ", ".join(entry["components"]) if entry["components"] else "none"
        lines.append(
            f"- {entry['name']} | family={entry['family']} | form={entry['stat_form']} | components={comp}"
        )

    return "\n".join(lines)

def blocked_families(tests_dir, test_config:dict):
    counts = {}

    for entry in build_prior_catalog(tests_dir):
        family = entry["family"]
        counts[family] = counts.get(family, 0) + 1

    blocked = sorted([family for family, count in counts.items() if count >= test_config["family_hard_cap"]])
    discouraged = sorted([family for family, count in counts.items() if test_config["family_soft_cap"] <= count < test_config["family_hard_cap"]])
    
    return blocked, discouraged

def rotation_guidance_text(tests_dir, test_config:dict):
    blocked, discouraged = blocked_families(tests_dir, test_config)
    blocked.remove('other') if 'other' in blocked else blocked
    discouraged.remove('other') if 'other' in discouraged else discouraged

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
    best = None
    best_score = -1.0

    for prior in prior_catalog:
        prior_tokens = set(prior["tokens"])
        overlap = len(candidate_tokens & prior_tokens)
        union = len(candidate_tokens | prior_tokens) or 1
        jaccard = overlap / union

        same_family = candidate["family"] == prior["family"]
        same_form = candidate["stat_form"] == prior["stat_form"]
        same_components = set(candidate["components"]) == set(prior["components"]) and bool(candidate["components"])

        score = jaccard

        if same_family:
            score += 0.20

        if same_form:
            score += 0.15

        if same_components:
            score += 0.25

        if score > best_score:
            best_score = score
            best = {
                "prior_name": prior["name"],
                "score": score,
                "jaccard": jaccard,
                "same_family": same_family,
                "same_form": same_form,
                "same_components": same_components,
                "family": prior["family"],
            }

    if best and (best["score"] >= 0.75 or (best["same_family"] and best["jaccard"] >= 0.45)):
        return best
    else:
        return None

def family_rotation_check(tests_dir, test_config:dict, candidate_name: str, candidate_description: str):
    signature = extract_test_signature(candidate_name, candidate_description)
    blocked, discouraged = blocked_families(tests_dir, test_config)

    if signature["family"] in blocked and signature["family"] != 'other':
        blocked.remove('other') if 'other' in blocked else blocked         
        return {
            "family": signature["family"],
            "severity": "blocked",
            "message": f"The following anomaly families have reached the hard cap and are temporarily blocked: {str(blocked)}"
        }
    elif signature["family"] in discouraged and signature["family"] != 'other':
        discouraged.remove('other') if 'other' in discouraged else discouraged
        return {
            "family": signature["family"],
            "severity": "discouraged",
            "message": f"The following anomaly families are already heavily used and should be avoided unless the test is genuinely distinct: {str(discouraged)}"
        }
    else:
        return None
