import numpy as np
import pytest

import app as dashboard
import attendance


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A test browser with two registered people and an empty database."""
    embeddings = tmp_path / "embeddings"
    embeddings.mkdir()
    for name in ["Ammu", "Ravi_K"]:
        np.save(embeddings / f"{name}.npy", np.zeros((5, 128)))
    monkeypatch.setattr(dashboard, "EMBEDDINGS_DIR", embeddings)
    monkeypatch.setattr(attendance, "ATTENDANCE_DB", tmp_path / "test.db")
    return dashboard.app.test_client()


def test_dashboard_shows_present_and_absent(client):
    attendance.mark_present("Ammu")
    page = client.get("/").get_data(as_text=True)
    assert "1 of 2 present" in page
    assert "Ravi K" in page


def test_csv_export(client):
    attendance.mark_present("Ammu")
    csv_text = client.get("/export.csv").get_data(as_text=True)
    assert "Ammu" in csv_text and "present" in csv_text
    assert "Ravi K" in csv_text and "absent" in csv_text


def test_bad_date_falls_back_to_today(client):
    assert client.get("/?date=not-a-date").status_code == 200
