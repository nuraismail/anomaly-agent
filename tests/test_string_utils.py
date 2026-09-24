from utils.string_utils import message_content_to_text, parse_test_metadata, text_to_dict


def test_parse_test_metadata_extracts_name_and_multiline_description():
    text = (
        "TEST_NAME: Quadrupole alignment\n"
        "DESCRIPTION: First line.\n\n\n"
        "Second line."
    )

    name, description = parse_test_metadata(text)

    assert name == "Quadrupole alignment"
    assert description == "First line.\n\nSecond line."


def test_message_content_to_text_handles_provider_content_blocks():
    content = [
        {"type": "text", "text": "first"},
        {"content": "second"},
        "third",
        {"type": "reasoning", "summary": "ignored"},
    ]

    assert message_content_to_text(content) == "first\nsecond\nthird"


def test_text_to_dict_parses_markdown_markers_case_insensitively():
    text = (
        "**HYPOTHESIS:** unusually high value\n"
        "with continuation\n\n"
        "test_type: one-tailed upper\n"
        "JUSTIFICATION: empirical tail test"
    )

    parsed = text_to_dict(text, ["HYPOTHESIS", "TEST_TYPE", "JUSTIFICATION"])

    assert parsed["HYPOTHESIS"] == "unusually high value\nwith continuation"
    assert parsed["TEST_TYPE"] == "one-tailed upper"
    assert parsed["JUSTIFICATION"] == "empirical tail test"


def test_message_content_to_text_drops_responses_api_reasoning_blocks():
    content = [
        {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": "TEST_NAME: draft thought\nDESCRIPTION: not the answer"}],
        },
        {"type": "text", "text": "TEST_NAME: Final\nDESCRIPTION: the answer", "annotations": []},
    ]

    text = message_content_to_text(content)

    assert text == "TEST_NAME: Final\nDESCRIPTION: the answer"
    assert "draft thought" not in text


def test_message_content_to_text_recurses_into_nested_content_lists():
    content = [{"type": "message", "content": [{"type": "output_text", "text": "nested"}]}]

    assert message_content_to_text(content) == "nested"
