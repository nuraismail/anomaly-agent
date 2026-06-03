from pathlib import Path

def cleanup_output_dir(path: Path):
    import shutil
    try:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
    except Exception as exc:
        print(f"Output cleanup warning: {exc}")

def parse_test_metadata(test_text: str) -> tuple[str, str]:
    import re
    name_match = re.search(r"TEST_NAME\s*:\s*(.+)", test_text)
    desc_match = re.search(r"DESCRIPTION\s*:\s*(.+)", test_text, re.DOTALL)
    name = name_match.group(1).strip() if name_match else "Unnamed anomaly test"
    raw_description = desc_match.group(1).strip() if desc_match else test_text.strip()
    description = re.sub(r"\n{3,}", "\n\n", raw_description)
    return name, description

def text_to_dict(text: str, stop_markers: list) -> dict:
    cleaned_text = str(text).rstrip()
    test_dict = {}
    for i in range(len(stop_markers)):
        if stop_markers[i] in cleaned_text:
            string = str(cleaned_text).split(str(stop_markers[i]) + ':', 1)[1].split('\n', 1)[0].strip()
        test_dict[stop_markers[i]] = string
    print(test_dict)
    return test_dict

