import pytest

from poppler import jobs


@pytest.fixture(autouse=True)
def poppler_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("POPPLER_HOME", str(home))
    yield home
    for job in jobs.all_jobs():
        if job.current_state() in (*jobs.ACTIVE, "lost"):
            jobs.cancel(job.id)
