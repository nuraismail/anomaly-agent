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

def message_content_to_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item.get("text", "")))
                elif "content" in item:
                    parts.append(str(item.get("content", "")))
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
