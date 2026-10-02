"""Run one job: wait for the GPU lock, run the command, record the outcome.

Started by jobs.submit as `python -m poppler.runner <id>`. It forks into its own
session so it outlives whoever submitted the job.

The command inherits the lock's file descriptor, so the lock is held until the
runner and every job process that inherited the descriptor have exited. When
the command's main process exits, anything it left running in its process group
is killed.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import signal
import subprocess
import sys
import time
import traceback

from poppler import jobs


class _Cancelled(Exception):
    pass


def _killpg(pgid: int, sig: signal.Signals) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, sig)


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def _stop(child: subprocess.Popen[bytes], kill_wait: float) -> int:
    """SIGTERM the job's group, then SIGKILL whatever is left after kill_wait seconds.

    The whole group gets the time to exit, not just its leader: /bin/sh often
    runs the real command as a child and dies on SIGTERM straight away.
    """
    _killpg(child.pid, signal.SIGTERM)
    deadline = time.monotonic() + kill_wait
    while time.monotonic() < deadline:
        if child.poll() is not None and not _group_alive(child.pid):
            break
        time.sleep(jobs.POLL)
    _killpg(child.pid, signal.SIGKILL)
    return child.wait()


def _finish(job: jobs.Job, state: str, exit_code: int | None = None) -> None:
    job.state = state
    job.exit_code = exit_code
    job.finished_at = time.time()
    job.save()


def run(job_id: int) -> None:
    job = jobs.load(job_id)
    stop: str | None = None
    waiting = True

    def on_signal(signum: int, frame: object) -> None:
        nonlocal stop
        stop = jobs.CANCELLED
        if waiting:
            raise _Cancelled  # interrupts the blocking flock below

    with open(job.log_path, "ab") as log, open(jobs.lock_path(), "a") as lock:
        try:
            signal.signal(signal.SIGTERM, on_signal)
            signal.signal(signal.SIGINT, on_signal)
            # cancel finds the runner through this pid, so no signal can come earlier
            job.runner_pid = os.getpid()
            job.runner_start = jobs.process_start(job.runner_pid)
            job.save()
            fcntl.flock(lock, fcntl.LOCK_EX)
            waiting = False
        except _Cancelled:
            _finish(job, jobs.CANCELLED)
            return

        job.state = jobs.RUNNING
        job.started_at = time.time()
        try:
            child = subprocess.Popen(
                ["/bin/sh", "-c", job.command],
                cwd=job.cwd or None,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                pass_fds=(lock.fileno(),),
                start_new_session=True,
                env={**os.environ, "POPPLER_JOB_ID": str(job.id)},
            )
        except OSError as e:
            log.write(f"poppler: could not start job: {e}\n".encode())
            _finish(job, jobs.FAILED)
            return
        job.child_pid = child.pid
        job.child_start = jobs.process_start(child.pid)
        job.save()

        deadline = None if job.time_limit is None else job.started_at + job.time_limit
        while True:
            try:
                exit_code = child.wait(timeout=jobs.POLL)
                break
            except subprocess.TimeoutExpired:
                pass
            if stop is None and deadline is not None and time.time() >= deadline:
                stop = jobs.TIMEOUT
            if stop is not None:
                exit_code = _stop(child, jobs.config()["KillWait"])
                break
        _killpg(child.pid, signal.SIGKILL)

    if stop is not None:
        _finish(job, stop, exit_code)
    else:
        _finish(job, jobs.COMPLETED if exit_code == 0 else jobs.FAILED, exit_code)


def main() -> None:
    job_id = int(sys.argv[1])
    if os.fork():
        os._exit(0)
    os.setsid()
    try:
        run(job_id)
    except Exception:
        traceback.print_exc()
        job = jobs.load(job_id)
        if job.state in jobs.ACTIVE:
            _finish(job, jobs.FAILED)
        sys.exit(1)


if __name__ == "__main__":
    main()
