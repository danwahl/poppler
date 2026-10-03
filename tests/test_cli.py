import json
import signal
import subprocess
import sys
import time

from poppler import jobs


def poppler(*args, cwd=None):
    return subprocess.run(
        [sys.executable, "-m", "poppler", *args],
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=30,
    )


def test_run_streams_output_and_passes_exit_code(tmp_path):
    result = poppler("run", "--", "sh", "-c", "echo hi; exit 4", cwd=tmp_path)
    assert result.stdout == "hi\n"
    assert result.returncode == 4


def test_run_accepts_a_single_shell_string(tmp_path):
    result = poppler("run", "--", "echo a | tr a b", cwd=tmp_path)
    assert result.stdout == "b\n"
    assert result.returncode == 0


def test_ctrl_c_cancels_the_job(tmp_path):
    proc = subprocess.Popen(
        [sys.executable, "-m", "poppler", "run", "--", "echo started; sleep 30"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=tmp_path,
    )
    assert proc.stdout.readline() == "started\n"
    proc.send_signal(signal.SIGINT)
    proc.wait(timeout=15)
    assert proc.returncode == 128 + signal.SIGTERM
    assert jobs.load(1).current_state() == "CANCELLED"


def test_log_follow_prints_until_the_job_ends(tmp_path):
    job_id = int(poppler("run", "-d", "--", "echo one; sleep 0.5; echo two", cwd=tmp_path).stdout)
    start = time.monotonic()
    assert poppler("log", "-f", str(job_id)).stdout == "one\ntwo\n"
    assert time.monotonic() - start >= 0.3


def test_detach_prints_job_id(tmp_path):
    result = poppler("run", "-d", "-J", "quick", "--", "true", cwd=tmp_path)
    job = jobs.wait(int(result.stdout), timeout=5)
    assert job.name == "quick"
    assert job.current_state() == "COMPLETED"


def test_time_limit_takes_slurm_format(tmp_path):
    result = poppler("run", "-t", "0:01", "--", "sleep 30", cwd=tmp_path)
    assert result.returncode == 128 + signal.SIGTERM
    assert "TIMEOUT" in result.stderr
    assert "invalid time limit" in poppler("run", "-t", "1h", "--", "true").stderr


def test_status_and_list(tmp_path):
    poppler("run", "--", "true", cwd=tmp_path)
    status = json.loads(poppler("status", "--json").stdout)
    assert status["gpu_busy"] is False
    assert status["running"] == []
    listed = poppler("list").stdout
    assert "COMPLETED" in listed


def test_unknown_job_is_an_error():
    result = poppler("log", "999")
    assert result.returncode == 1
    assert "no job 999" in result.stderr
