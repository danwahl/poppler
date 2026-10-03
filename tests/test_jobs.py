import os
import signal
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from poppler import jobs


@pytest.fixture
def submit(tmp_path):
    def submit(command, **kwargs):
        return jobs.submit(command, cwd=str(tmp_path), **kwargs)

    return submit


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.05)


def wait_running(job_id):
    wait_for(lambda: jobs.load(job_id).current_state() == "RUNNING")
    return jobs.load(job_id)


def alive(pid):
    return jobs.process_start(pid) is not None


def test_success_records_exit_code_and_output(submit):
    job = jobs.wait(submit('echo "hello from $POPPLER_JOB_ID"').id, timeout=5)
    assert job.current_state() == "COMPLETED"
    assert job.exit_code == 0
    assert jobs.read_log(job.id) == f"hello from {job.id}\n"


def test_failure_records_exit_code(submit):
    job = jobs.wait(submit("exit 3").id, timeout=5)
    assert job.current_state() == "FAILED"
    assert job.exit_code == 3


def test_runs_in_requested_directory(submit, tmp_path):
    job = jobs.wait(submit("pwd").id, timeout=5)
    assert jobs.read_log(job.id).strip() == str(tmp_path)


def test_jobs_take_turns(submit, tmp_path):
    ids = [
        submit(f"echo start {n} >> trace; sleep 0.2; echo end {n} >> trace").id for n in range(3)
    ]
    for job_id in ids:
        assert jobs.wait(job_id, timeout=10).current_state() == "COMPLETED"
    lines = (tmp_path / "trace").read_text().split("\n")[:-1]
    pairs = [(lines[i], lines[i + 1]) for i in range(0, len(lines), 2)]
    assert all(start.split()[1] == end.split()[1] for start, end in pairs), lines


def test_instant_failures_do_not_stall_the_queue(submit):
    holder = submit("sleep 0.3")
    crashes = [submit("exit 1") for _ in range(3)]
    last = submit("echo ok")
    assert jobs.wait(last.id, timeout=5).current_state() == "COMPLETED"
    assert all(jobs.load(c.id).current_state() == "FAILED" for c in crashes)
    assert jobs.wait(holder.id, timeout=5).current_state() == "COMPLETED"


def test_killed_job_releases_gpu(submit):
    victim = wait_running(submit("sleep 30").id)
    queued = submit("true")
    os.kill(victim.runner_pid, signal.SIGKILL)
    os.killpg(victim.child_pid, signal.SIGKILL)
    assert jobs.wait(queued.id, timeout=5).current_state() == "COMPLETED"
    assert jobs.load(victim.id).current_state() == "LOST"


def test_command_keeps_gpu_after_its_runner_dies(submit):
    job = wait_running(submit("sleep 30").id)
    os.kill(job.runner_pid, signal.SIGKILL)
    wait_for(lambda: jobs.load(job.id).current_state() == "LOST")
    assert jobs.gpu_busy()
    assert jobs.cancel(job.id).current_state() == "CANCELLED"
    wait_for(lambda: not jobs.gpu_busy())


def test_cancel_kills_the_whole_process_tree(submit, tmp_path):
    job = wait_running(submit("sleep 60 & echo $! > bg.pid; wait").id)
    wait_for(lambda: (tmp_path / "bg.pid").exists())
    background = int((tmp_path / "bg.pid").read_text())
    assert alive(background)
    assert jobs.cancel(job.id).current_state() == "CANCELLED"
    wait_for(lambda: not alive(background))
    assert not jobs.gpu_busy()


def test_leftover_background_processes_are_killed(submit, tmp_path):
    job = jobs.wait(submit("sleep 60 & echo $! > bg.pid").id, timeout=5)
    assert job.current_state() == "COMPLETED"
    wait_for(lambda: not alive(int((tmp_path / "bg.pid").read_text())))
    assert not jobs.gpu_busy()


def test_cancel_waiting_job(submit):
    holder = wait_running(submit("sleep 30").id)
    queued = submit("touch should-not-exist")
    assert queued.current_state() == "PENDING"
    cancelled = jobs.cancel(queued.id)
    assert cancelled.current_state() == "CANCELLED"
    assert cancelled.started_at is None
    assert jobs.load(holder.id).current_state() == "RUNNING"


def test_timeout_stops_the_job(submit):
    job = jobs.wait(submit("sleep 30", time_limit=0.3).id, timeout=5)
    assert job.current_state() == "TIMEOUT"
    assert job.exit_code == -signal.SIGTERM
    assert not jobs.gpu_busy()


def test_sigterm_reaches_the_command(submit, tmp_path):
    job = wait_running(
        submit("trap 'echo checkpoint; exit 0' TERM; touch ready; sleep 30 & wait").id
    )
    wait_for(lambda: (tmp_path / "ready").exists())
    assert jobs.cancel(job.id).current_state() == "CANCELLED"
    assert "checkpoint" in jobs.read_log(job.id)


def test_kill_wait_covers_processes_under_the_shell(submit, tmp_path):
    # dash does not exec a lone command, so python runs as a child of sh,
    # which dies on SIGTERM at once.
    (tmp_path / "train.py").write_text(
        textwrap.dedent("""
            import pathlib, signal, sys, time

            def checkpoint(*_):
                time.sleep(0.5)
                print("saved", flush=True)
                sys.exit(0)

            signal.signal(signal.SIGTERM, checkpoint)
            pathlib.Path("ready").touch()
            time.sleep(30)
        """)
    )
    job = wait_running(submit(f"{sys.executable} train.py").id)
    wait_for(lambda: (tmp_path / "ready").exists())
    assert jobs.cancel(job.id).current_state() == "CANCELLED"
    assert "saved" in jobs.read_log(job.id)


def test_sigkill_after_kill_wait(submit, tmp_path):
    (tmp_path / "config.toml").write_text("KillWait = 0.3\n")
    job = wait_running(submit("trap '' TERM; touch ready; sleep 30").id)
    wait_for(lambda: (tmp_path / "ready").exists())
    start = time.monotonic()
    cancelled = jobs.cancel(job.id)
    assert time.monotonic() - start < 3
    assert cancelled.current_state() == "CANCELLED"
    assert cancelled.exit_code == -signal.SIGKILL


def test_cancel_lost_job_kills_what_is_left_of_its_group(submit, tmp_path):
    job = wait_running(submit("sleep 60 & echo $! > bg.pid; sleep 0.5").id)
    os.kill(job.runner_pid, signal.SIGKILL)
    wait_for(lambda: not alive(job.child_pid))
    background = int((tmp_path / "bg.pid").read_text())
    assert jobs.gpu_busy()
    assert background in jobs.status()["lock_holders"]
    assert jobs.cancel(job.id).current_state() == "CANCELLED"
    wait_for(lambda: not jobs.gpu_busy())


def test_missing_directory_fails_cleanly(tmp_path):
    job = jobs.wait(jobs.submit("true", cwd=str(tmp_path / "missing")).id, timeout=5)
    assert job.current_state() == "FAILED"
    assert "could not start job" in jobs.read_log(job.id)
    assert not jobs.gpu_busy()


def test_concurrent_submits_get_distinct_ids(submit):
    with ThreadPoolExecutor(8) as pool:
        ids = list(pool.map(lambda _: submit("true").id, range(8)))
    assert sorted(ids) == list(range(1, 9))
    for job_id in ids:
        assert jobs.wait(job_id, timeout=10).current_state() == "COMPLETED"


def test_gpu_memory_tolerates_odd_nvidia_smi_output(monkeypatch):
    for stdout in ("", "1024, 24564, [N/A]\n"):
        monkeypatch.setattr(
            jobs.subprocess, "run", lambda *a, out=stdout, **k: SimpleNamespace(stdout=out)
        )
        assert jobs.gpu_memory() is None


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("90", 5400),
        ("1:30", 90),
        ("1:00:05", 3605),
        ("2-0", 172800),
        ("1-2:03", 93780),
        ("1-0:00:07", 86407),
        ("UNLIMITED", None),
    ],
)
def test_parse_time(text, seconds):
    assert jobs.parse_time(text) == seconds


@pytest.mark.parametrize("text", ["", "1:2:3:4", "1-2:3:4:5", "-5", "1.5", "a"])
def test_parse_time_rejects_other_formats(text):
    with pytest.raises(ValueError):
        jobs.parse_time(text)


def test_config_rejects_unknown_settings(tmp_path):
    (tmp_path / "config.toml").write_text("Killwait = 1\n")
    with pytest.raises(ValueError, match="Killwait"):
        jobs.config()


@pytest.mark.parametrize("value", ["-1", "inf", "nan", "true", '"30"'])
def test_config_rejects_bad_kill_wait(tmp_path, value):
    (tmp_path / "config.toml").write_text(f"KillWait = {value}\n")
    with pytest.raises(ValueError, match="number of seconds"):
        jobs.config()


def test_kill_wait_is_fixed_at_submission(submit, tmp_path):
    (tmp_path / "config.toml").write_text("KillWait = 0.3\n")
    job = wait_running(submit("trap '' TERM; touch ready; sleep 30").id)
    (tmp_path / "config.toml").write_text("KillWait = 60\n")
    wait_for(lambda: (tmp_path / "ready").exists())
    assert job.kill_wait == 0.3
    assert jobs.cancel(job.id).exit_code == -signal.SIGKILL


def test_time_limit_of_zero_means_no_limit(submit):
    job = submit("true", time_limit=0)
    assert job.time_limit is None
    assert jobs.wait(job.id, timeout=5).current_state() == "COMPLETED"


def test_unknown_job():
    with pytest.raises(KeyError):
        jobs.load(999)


def test_pending_jobs_start_in_submission_order(submit, tmp_path):
    holder = wait_running(submit("sleep 0.5").id)
    ids = [submit(f"echo {n} >> order").id for n in range(5)]
    for job_id in ids:
        assert jobs.wait(job_id, timeout=10).current_state() == "COMPLETED"
    assert (tmp_path / "order").read_text().split() == [str(n) for n in range(5)]
    assert jobs.load(holder.id).current_state() == "COMPLETED"


def test_lost_pending_job_does_not_block_the_queue(submit):
    holder = wait_running(submit("sleep 0.5").id)
    lost = submit("true")
    os.kill(lost.runner_pid, signal.SIGKILL)
    after = submit("true")
    assert jobs.wait(after.id, timeout=5).current_state() == "COMPLETED"
    assert jobs.load(lost.id).current_state() == "LOST"
    assert jobs.wait(holder.id, timeout=5).current_state() == "COMPLETED"


def test_finished_jobs_leave_the_active_list(submit):
    job = jobs.wait(submit("true").id, timeout=5)
    assert job.current_state() == "COMPLETED"
    assert jobs.active_jobs() == []
    assert [j.id for j in jobs.all_jobs()] == [job.id]


def test_higher_qos_starts_first(submit, tmp_path):
    holder = wait_running(submit("sleep 0.5").id)
    ids = [submit(f"echo {qos} >> order", qos=qos).id for qos in ("scavenger", "normal", "high")]
    for job_id in ids:
        assert jobs.wait(job_id, timeout=10).current_state() == "COMPLETED"
    assert (tmp_path / "order").read_text().split() == ["high", "normal", "scavenger"]
    assert jobs.load(holder.id).current_state() == "COMPLETED"


RESUMABLE = (
    'echo "run $POPPLER_RESTART_COUNT" >> trace; '
    '[ "$POPPLER_RESTART_COUNT" -gt 0 ] && exit 0; '
    "trap 'echo checkpoint >> trace; exit 0' TERM; touch ready; sleep 30 & wait"
)


def test_preempted_scavenger_job_is_requeued(submit, tmp_path):
    scavenger = wait_running(submit(RESUMABLE, qos="scavenger").id)
    wait_for(lambda: (tmp_path / "ready").exists())
    urgent = submit("echo urgent >> trace")
    assert jobs.wait(urgent.id, timeout=10).current_state() == "COMPLETED"
    finished = jobs.wait(scavenger.id, timeout=10)
    assert finished.current_state() == "COMPLETED"
    assert finished.restart_count == 1
    assert (tmp_path / "trace").read_text().split("\n")[:-1] == [
        "run 0",
        "checkpoint",
        "urgent",
        "run 1",
    ]
    assert "preempted and requeued" in jobs.read_log(scavenger.id)


def test_preempted_job_without_requeue_ends(submit):
    scavenger = wait_running(submit("sleep 30", qos="scavenger", requeue=False).id)
    urgent = submit("true", qos="high")
    assert jobs.wait(urgent.id, timeout=10).current_state() == "COMPLETED"
    finished = jobs.load(scavenger.id)
    assert finished.current_state() == "PREEMPTED"
    assert finished.exit_code == -signal.SIGTERM


def test_only_scavenger_jobs_are_preempted(submit):
    holder = wait_running(submit("sleep 1").id)
    urgent = submit("true", qos="high")
    time.sleep(0.5)
    assert jobs.load(holder.id).current_state() == "RUNNING"
    assert jobs.load(urgent.id).current_state() == "PENDING"
    assert jobs.wait(urgent.id, timeout=5).current_state() == "COMPLETED"
    assert jobs.load(holder.id).current_state() == "COMPLETED"


def test_cancel_requeued_job(submit, tmp_path):
    scavenger = wait_running(submit("sleep 30", qos="scavenger").id)
    holder = submit("sleep 30")
    wait_for(lambda: jobs.load(scavenger.id).current_state() == "PENDING")
    assert jobs.load(scavenger.id).restart_count == 1
    assert jobs.cancel(scavenger.id).current_state() == "CANCELLED"
    jobs.cancel(holder.id)


def test_unknown_qos(submit):
    with pytest.raises(ValueError, match="unknown QOS"):
        submit("true", qos="urgent")


def test_lost_scavenger_job_is_preempted(submit, tmp_path):
    scavenger = wait_running(submit("sleep 60 & echo $! > bg.pid; wait", qos="scavenger").id)
    wait_for(lambda: (tmp_path / "bg.pid").exists())
    os.kill(scavenger.runner_pid, signal.SIGKILL)
    wait_for(lambda: jobs.load(scavenger.id).current_state() == "LOST")
    urgent = submit("true")
    assert jobs.wait(urgent.id, timeout=5).current_state() == "COMPLETED"
    assert jobs.load(scavenger.id).current_state() == "PREEMPTED"
    wait_for(lambda: not alive(int((tmp_path / "bg.pid").read_text())))


def test_status_lists_pending_jobs_in_start_order(submit):
    holder = wait_running(submit("sleep 30").id)
    ids = [submit("true", qos=qos).id for qos in ("scavenger", "normal", "high")]
    assert [job["id"] for job in jobs.status()["pending"]] == ids[::-1]
    jobs.cancel(holder.id)


def test_afterok_waits_for_the_job_to_complete(submit, tmp_path):
    holder = wait_running(submit("sleep 0.5").id)
    first = submit("echo first >> trace")
    second = submit("echo second >> trace", qos="high", dependency=f"afterok:{first.id}")
    assert jobs.wait(second.id, timeout=10).current_state() == "COMPLETED"
    assert (tmp_path / "trace").read_text().split() == ["first", "second"]
    assert jobs.load(holder.id).current_state() == "COMPLETED"


def test_afterok_on_a_failed_job_cancels_the_dependent(submit):
    failed = jobs.wait(submit("exit 1").id, timeout=5)
    dependent = jobs.wait(submit("true", dependency=f"afterok:{failed.id}").id, timeout=5)
    assert dependent.current_state() == "CANCELLED"
    assert dependent.started_at is None
    assert "can never be satisfied" in jobs.read_log(dependent.id)


def test_a_job_waiting_on_a_dependency_neither_preempts_nor_blocks(submit):
    scavenger = wait_running(submit("sleep 30", qos="scavenger").id)
    dependent = submit("true", qos="high", dependency=f"afterany:{scavenger.id}")
    time.sleep(0.5)
    assert jobs.load(scavenger.id).current_state() == "RUNNING"
    assert jobs.status()["pending"][0]["reason"] == "Dependency"
    jobs.cancel(scavenger.id)
    assert jobs.wait(dependent.id, timeout=5).current_state() == "COMPLETED"


def test_pending_reasons(submit):
    holder = wait_running(submit("sleep 30").id)
    ids = [submit("true").id for _ in range(2)]
    reasons = {job["id"]: job["reason"] for job in jobs.status()["pending"]}
    assert reasons == {ids[0]: "Resources", ids[1]: "Priority"}
    jobs.cancel(holder.id)


@pytest.mark.parametrize("text", ["after:1", "afterok", "afterok:x", "afterok:1?afterany:1"])
def test_invalid_dependency(submit, text):
    with pytest.raises(ValueError, match="invalid dependency"):
        submit("true", dependency=text)


def test_dependency_on_a_missing_job(submit):
    with pytest.raises(KeyError, match="no job 99"):
        submit("true", dependency="afterany:99")


def test_env_adds_to_the_environment(submit):
    job = jobs.wait(submit('echo "$GREETING $HOME"', env={"GREETING": "hi"}).id, timeout=5)
    assert jobs.read_log(job.id) == f"hi {os.environ['HOME']}\n"


def test_select_filters_by_owner_and_state(submit):
    done = jobs.wait(submit("true", owner="a").id, timeout=5)
    jobs.wait(submit("exit 1", owner="a").id, timeout=5)
    jobs.wait(submit("true", owner="b").id, timeout=5)
    assert [job.id for job in jobs.select("a", ["COMPLETED"])] == [done.id]
    assert len(jobs.select(states=["FAILED", "COMPLETED"])) == 3


def test_a_missing_dependency_record_cancels_only_its_dependent(submit):
    holder = wait_running(submit("sleep 30").id)
    target = submit("true")
    dependent = submit("true", dependency=f"afterok:{target.id}")
    other = submit("true")
    jobs.cancel(target.id)
    (jobs.jobs_dir() / f"{target.id}.json").unlink()
    assert jobs.wait(dependent.id, timeout=5).current_state() == "CANCELLED"
    jobs.cancel(holder.id)
    assert jobs.wait(other.id, timeout=10).current_state() == "COMPLETED"


def test_env_value_with_a_null_byte_is_rejected(submit):
    with pytest.raises(ValueError, match="null byte"):
        submit("true", env={"A": "x\0y"})


def test_records_are_private(submit):
    job = submit("true")
    assert job.path.stat().st_mode & 0o777 == 0o600
