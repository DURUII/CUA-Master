from __future__ import annotations

from pathlib import Path

from cuamaster.seeding.common.core import sanitize, target_path
from cuamaster.seeding.common.utils import filter_records, label_matches
from cuamaster.seeding.scoring import score
from cuamaster.seeding.transcription import transcribe


def test_sanitize_is_shared_for_path_segments() -> None:
    assert sanitize("Business & Administration") == "Business_and_Administration"
    assert sanitize("Monday.com") == "Monday_com"
    assert sanitize("7-Zip") == "7_Zip"
    assert sanitize("Notepad++") == "Notepad"


def test_filter_records_matches_raw_and_sanitized_labels() -> None:
    records = [
        {
            "video_id": "abc123def45",
            "category": "Business & Administration",
            "subcategory": "Project & Operations",
            "application": "Monday.com",
        }
    ]

    assert label_matches("Monday_com", "Monday.com")
    assert filter_records(records, category="Business_and_Administration", app="Monday_com") == records


def test_target_path_uses_shared_sanitize_rule(tmp_path: Path) -> None:
    record = {
        "video_id": "abc123def45",
        "category": "Business & Administration",
        "subcategory": "Project & Operations",
        "application": "Monday.com",
    }

    assert target_path(record, tmp_path).relative_to(tmp_path) == Path(
        "Business_and_Administration/Project_and_Operations/Monday_com_abc123def45.mp4"
    )


def test_scan_filters_match_sanitized_file_application(tmp_path: Path) -> None:
    video = (
        tmp_path
        / "Business_and_Administration"
        / "Project_and_Operations"
        / "Monday_com_abc123def45.mp4"
    )
    video.parent.mkdir(parents=True)
    video.write_bytes(b"fake")

    transcribe_tasks = transcribe._scan_tasks(
        tmp_path,
        category="Business & Administration",
        subcategory="Project & Operations",
        app="Monday.com",
        limit=None,
    )
    score_tasks = score._scan_tasks(
        tmp_path,
        category="Business & Administration",
        subcategory="Project & Operations",
        app="Monday.com",
        limit=None,
    )

    assert [task["record"]["video_id"] for task in transcribe_tasks] == ["abc123def45"]
    assert [task["record"]["video_id"] for task in score_tasks] == ["abc123def45"]
