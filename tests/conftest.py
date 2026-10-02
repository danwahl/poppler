import pytest

from poppler import jobs


@pytest.fixture(autouse=True)
def poppler_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    config = tmp_path / "config.toml"
    config.write_text("KillWait = 5\n")
    monkeypatch.setenv("POPPLER_HOME", str(home))
    monkeypatch.setenv("POPPLER_CONFIG", str(config))
    yield home
    for job in jobs.all_jobs():
        if job.current_state() in (*jobs.ACTIVE, jobs.LOST):
            jobs.cancel(job.id)
