from pathlib import Path
import re

def cleanup_output_dir(path: Path):
    import shutil
    try:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
    except Exception as exc:
        print(f"Output cleanup warning: {exc}")

def parse_test_metadata(test_text: str) -> tuple[str, str]:
    name_match = re.search(r"TEST_NAME\s*:\s*(.+)", test_text)
    desc_match = re.search(r"DESCRIPTION\s*:\s*(.+)", test_text, re.DOTALL)
    name = name_match.group(1).strip() if name_match else "Unnamed anomaly test"
    raw_description = desc_match.group(1).strip() if desc_match else test_text.strip()
    description = re.sub(r"\n{3,}", "\n\n", raw_description)
    return name, description

REASONING_BLOCK_TYPES = {
    "reasoning",
    "reasoning_text",
    "reasoning_content",
    "reasoning_summary",
    "thinking",
    "redacted_thinking",
}


DECLARED_FIELDS = ("FAMILY", "STATISTIC_FORM")


def parse_declared_fields(test_text: str) -> dict:
    """Return planner-declared FAMILY / STATISTIC_FORM labels, normalised, if present.

    Labels are lower-cased with underscores and hyphens turned into spaces so
    that "power_spectrum", "Power-Spectrum" and "power spectrum" count as one
    family. Missing or placeholder values are omitted.
    """
    declared = {}
    for field in DECLARED_FIELDS:
        match = re.search(rf"(?im)^\s*(?:\*\*)?{field}(?:\*\*)?\s*:\s*(.+)$", test_text)
        if not match:
            continue
        value = match.group(1).strip().strip("*`").strip().lower()
        value = re.sub(r"[_\-]+", " ", value)
        value = re.sub(r"\s+", " ", value).strip(" .")
        if value and value not in {"none", "n/a", "null"}:
            declared[field.lower()] = value
    return declared


def message_content_to_text(content) -> str:
    """Flatten provider content blocks to visible text, dropping reasoning blocks.

    Responses-API style reasoning blocks carry the chain of thought under
    ``content`` (for example MiMo) or ``summary`` (OpenAI); neither is part of
    the model's answer and must not reach downstream parsers.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                item_type = str(item.get("type") or "")
                if item_type in REASONING_BLOCK_TYPES:
                    continue
                if "text" in item:
                    parts.append(str(item.get("text", "")))
                elif "content" in item:
                    parts.append(message_content_to_text(item.get("content")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)

def text_to_dict(text: str, stop_markers: list) -> dict:
    cleaned_text = message_content_to_text(text).strip()
    parsed = {marker: "" for marker in stop_markers}

    if not cleaned_text or not stop_markers:
        return parsed

    marker_pattern = "|".join(re.escape(marker) for marker in stop_markers)
    pattern = re.compile(
        rf"(?im)^\s*(?:\*\*)?\s*(?P<marker>{marker_pattern})\s*(?:\*\*)?\s*:\s*(?:\*\*)?\s*"
    )
    marker_lookup = {marker.lower(): marker for marker in stop_markers}
    matches = list(pattern.finditer(cleaned_text))

    for index, match in enumerate(matches):
        marker = marker_lookup.get(match.group("marker").lower())
        if marker is None:
            continue
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(cleaned_text)
        parsed[marker] = cleaned_text[start:end].strip()

    return parsed
