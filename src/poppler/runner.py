"""Run one job: wait for its turn on the GPU, run the command, record the outcome.

Started by jobs.submit as `python -m poppler.runner <id>`. It forks into its own
session so it outlives whoever submitted the job.

A pending runner polls the queue and tries the lock only when its job is first,
so jobs start in order. If the lock is busy, the first job preempts the running
job when its QOS allows; a preempted job that allows requeueing goes back to
pending under the same id and runs again from the start.

The command inherits the lock's file descriptor, so the lock is held until the
runner and every job process that inherited the descriptor have exited. When
the command's main process exits, anything it left running in its process
group is killed.
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
        self.preempted = False
        self.running = False

    def on_cancel(self, signum: int, frame: object) -> None:
        self.cancelled = True

    def on_preempt(self, signum: int, frame: object) -> None:
        # A preemptor that read a stale record may signal a job that is pending again.
        if self.running:
            self.preempted = True

    def wait_for_gpu(self, log: BinaryIO) -> TextIO | None:
        """Return the lock file, locked, once this job is first in the queue.

        While first, signal any running job this job's QOS may preempt, once per
        poll until that job stops. Returns None if the job is cancelled first,
        or if its dependencies can never be met, as Slurm does with
        kill_invalid_depend.
        """
        while not self.cancelled:
            deps = jobs.dependency_state(self.job)
            if deps == "never":
                log.write(b"poppler: cancelled because its dependency can never be satisfied\n")
                return None
            if deps == "waiting":
                time.sleep(jobs.POLL)
                continue
            pending = jobs.queue()
            if pending and pending[0].id == self.job.id:
                lock = open(jobs.lock_path(), "a")  # noqa: SIM115 (returned to the caller)
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return lock
                except BlockingIOError:
                    lock.close()
                for other in jobs.active_jobs():
                    # A lost job that was running may still hold the lock.
                    if other.state == jobs.RUNNING and jobs.may_preempt(self.job, other):
                        jobs.preempt(jobs.load_settled(other.id))
            time.sleep(jobs.POLL)
        return None

    def execute(self, lock: TextIO, log: BinaryIO) -> tuple[str, int | None]:
        """Run the command while holding the lock; return its final state and exit code."""
        job = self.job
        self.running = True  # before the record says RUNNING, so no preemption is missed
        job.state = jobs.RUNNING
        job.started_at = time.time()
        if self.cancelled:
            self.running = False
            return jobs.CANCELLED, None
        try:
            child = subprocess.Popen(
                ["/bin/sh", "-c", job.command],
                cwd=job.cwd or None,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                pass_fds=(lock.fileno(),),
                start_new_session=True,
                env={
                    **os.environ,
                    **job.env,
                    "POPPLER_JOB_ID": str(job.id),
                    "POPPLER_RESTART_COUNT": str(job.restart_count),
                },
            )
        except OSError as e:
            log.write(f"poppler: could not start job: {e}\n".encode())
            self.running = False
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
                elif self.preempted:
                    stop = jobs.PREEMPTED
                elif deadline is not None and time.time() >= deadline:
                    stop = jobs.TIMEOUT
                if stop is not None:
                    exit_code = _stop(child, job.kill_wait)
                    break
        finally:
            _killpg(child.pid, signal.SIGKILL)
            self.running = False
        if stop is None:
            stop = jobs.COMPLETED if exit_code == 0 else jobs.FAILED
        return stop, exit_code

    def requeue(self, log: BinaryIO) -> None:
        job = self.job
        log.write(f"poppler: job {job.id} preempted and requeued\n".encode())
        log.flush()
        job.state = jobs.PENDING
        job.restart_count += 1
        job.started_at = job.exit_code = job.child_pid = job.child_start = None
        job.save()
        self.preempted = False


def run(job_id: int) -> None:
    runner = Runner(jobs.load(job_id))
    job = runner.job
    signal.signal(signal.SIGTERM, runner.on_cancel)
    signal.signal(signal.SIGINT, runner.on_cancel)
    signal.signal(signal.SIGUSR1, runner.on_preempt)
    # cancel finds the runner through this pid, so no signal can come earlier
    job.runner_pid = os.getpid()
    job.runner_start = jobs.process_start(job.runner_pid)
    job.save()
    with open(job.log_path, "ab") as log:
        while True:
            lock = runner.wait_for_gpu(log)
            if lock is None:
                jobs.finish(job, jobs.CANCELLED)
                return
            with lock:
                state, exit_code = runner.execute(lock, log)
            if state == jobs.PREEMPTED and runner.cancelled:
                state = jobs.CANCELLED  # cancelled while stopping for a preemption
            if state != jobs.PREEMPTED or not job.requeue:
                break
            runner.requeue(log)
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
