"""Run one job: wait for its turn on the GPU, run the command, record the outcome.

Started by jobs.submit as `python -m poppler.runner <id>`. It forks into its own
session so it outlives whoever submitted the job.

A pending runner polls the queue and tries the lock only when its job is first,
so jobs start in order. The command inherits the lock's file descriptor, so the
lock is held until the runner and every job process that inherited the
descriptor have exited. When the command's main process exits, anything it
left running in its process group is killed.
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
from typing import BinaryIO, TextIO

from poppler import jobs


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


class Runner:
    """Runs one job. Signal handlers set flags that the polling loops check."""

    def __init__(self, job: jobs.Job) -> None:
        self.job = job
        self.cancelled = False

    def on_cancel(self, signum: int, frame: object) -> None:
        self.cancelled = True

    def wait_for_gpu(self) -> TextIO | None:
        """Return the lock file, locked, once this job is first in the queue.

        Returns None if the job is cancelled first.
        """
        while not self.cancelled:
            pending = jobs.queue()
            if pending and pending[0].id == self.job.id:
                lock = open(jobs.lock_path(), "a")  # noqa: SIM115 (returned to the caller)
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return lock
                except BlockingIOError:
                    lock.close()
            time.sleep(jobs.POLL)
        return None

    def execute(self, lock: TextIO, log: BinaryIO) -> tuple[str, int | None]:
        """Run the command while holding the lock; return its final state and exit code."""
        job = self.job
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
            return jobs.FAILED, None
        job.child_pid = child.pid
        job.child_start = jobs.process_start(child.pid)
        job.save()

        deadline = None if job.time_limit is None else job.started_at + job.time_limit
        stop = None
        try:
            while True:
                try:
                    exit_code = child.wait(timeout=jobs.POLL)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if self.cancelled:
                    stop = jobs.CANCELLED
                elif deadline is not None and time.time() >= deadline:
                    stop = jobs.TIMEOUT
                if stop is not None:
                    exit_code = _stop(child, jobs.config()["KillWait"])
                    break
        finally:
            _killpg(child.pid, signal.SIGKILL)
        if stop is None:
            stop = jobs.COMPLETED if exit_code == 0 else jobs.FAILED
        return stop, exit_code


def run(job_id: int) -> None:
    runner = Runner(jobs.load(job_id))
    job = runner.job
    signal.signal(signal.SIGTERM, runner.on_cancel)
    signal.signal(signal.SIGINT, runner.on_cancel)
    # cancel finds the runner through this pid, so no signal can come earlier
    job.runner_pid = os.getpid()
    job.runner_start = jobs.process_start(job.runner_pid)
    job.save()
    with open(job.log_path, "ab") as log:
        lock = runner.wait_for_gpu()
        if lock is None:
            jobs.finish(job, jobs.CANCELLED)
            return
        with lock:
            state, exit_code = runner.execute(lock, log)
    jobs.finish(job, state, exit_code)


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
            jobs.finish(job, jobs.FAILED)
        sys.exit(1)


if __name__ == "__main__":
    main()
