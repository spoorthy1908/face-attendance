from datetime import date

import pytest

import attendance


@pytest.fixture(autouse=True)
def temporary_database(tmp_path, monkeypatch):
    """Use a fresh, empty database for every test instead of the real one."""
    monkeypatch.setattr(attendance, "ATTENDANCE_DB", tmp_path / "test.db")


def test_marks_person_present():
    assert attendance.mark_present("Ammu") is True
    records = attendance.get_attendance(date.today().isoformat())
    assert [name for name, _ in records] == ["Ammu"]


def test_only_marks_once_per_day():
    attendance.mark_present("Ammu")
    assert attendance.mark_present("Ammu") is False
    assert len(attendance.get_attendance(date.today().isoformat())) == 1


def test_empty_day_has_no_records():
    assert attendance.get_attendance("2000-01-01") == []
