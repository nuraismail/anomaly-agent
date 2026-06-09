import json

from utils.family_novelty import (
    extract_test_signature,
    family_rotation_check,
    novelty_check,
)


def write_summary(run_dir, index, name, description=""):
    test_dir = run_dir / f"Test_{index:02d}"
    test_dir.mkdir()
    with (test_dir / "result_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"test_name": name, "test_description": description}, handle)


def test_extract_test_signature_finds_family_form_and_components():
    signature = extract_test_signature(
        "Quadrupole octupole alignment angle",
        "Alignment angle between the quadrupole and octupole planes.",
    )

    assert signature["family"] == "alignment"
    assert signature["stat_form"] == "angle"
    assert signature["components"] == ["quadrupole", "octupole"]


def test_novelty_check_rejects_near_duplicate_prior_test(tmp_path):
    write_summary(
        tmp_path,
        1,
        "Quadrupole octupole alignment angle",
        "Alignment angle between quadrupole and octupole planes.",
    )

    issue = novelty_check(
        tmp_path,
        "Quadrupole octupole alignment angle",
        "Alignment angle between quadrupole and octupole planes.",
    )

    assert issue is not None
    assert issue["prior_name"] == "Quadrupole octupole alignment angle"
    assert issue["same_family"] is True
    assert issue["same_form"] is True


def test_family_rotation_check_flags_soft_and_hard_caps(tmp_path):
    config = {"family_soft_cap": 2, "family_hard_cap": 3}
    write_summary(tmp_path, 1, "Parity quadrupole ratio")
    write_summary(tmp_path, 2, "Parity octupole ratio")

    discouraged = family_rotation_check(
        tmp_path,
        config,
        "Parity dipole ratio",
        "Parity ratio for the dipole.",
    )

    assert discouraged["severity"] == "discouraged"
    assert discouraged["matched_families"] == ["parity"]

    write_summary(tmp_path, 3, "Parity hemisphere ratio")
    blocked = family_rotation_check(
        tmp_path,
        config,
        "Parity galactic ratio",
        "Parity ratio for galactic components.",
    )

    assert blocked["severity"] == "blocked"
    assert blocked["matched_families"] == ["parity"]
