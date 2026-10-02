"""Job records, the GPU lock, and the operations the CLI and MCP server share.

Each job is a JSON file in the jobs directory. submit creates it, the job's
runner (see runner.py) updates it, and cancel finishes it if the runner has
died. A record whose runner is gone reads as "lost", so a crashed runner never
leaves a job that looks active.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ACTIVE = ("waiting", "running")
POLL = 0.1


def home() -> Path:
    if path := os.environ.get("POPPLER_HOME"):
        return Path(path)
    state = os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state"
    return Path(state) / "poppler"


def jobs_dir() -> Path:
    path = home() / "jobs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def lock_path() -> Path:
    jobs_dir()
    return home() / "gpu.lock"


def process_start(pid: int) -> int | None:
    """Return the start time of a live, non-zombie process, or None.

    Comparing start times guards against a recorded pid being reused.
    """
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    fields = stat.rsplit(")", 1)[1].split()
    if fields[0] == "Z":
        return None
    return int(fields[19])


@dataclass
class Job:
    id: int
    command: str
    name: str = ""
    owner: str = ""
    cwd: str = ""
    timeout: float | None = None
    grace: float = 10.0
    state: str = "waiting"
    submitted_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    exit_code: int | None = None
    runner_pid: int | None = None
    runner_start: int | None = None
    child_pid: int | None = None
    child_start: int | None = None

    @property
    def path(self) -> Path:
        return jobs_dir() / f"{self.id}.json"

    @property
    def log_path(self) -> Path:
        return jobs_dir() / f"{self.id}.log"

    def save(self) -> None:
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(asdict(self)))
        tmp.replace(self.path)

    def runner_alive(self) -> bool:
        return self.runner_pid is not None and process_start(self.runner_pid) == self.runner_start

    def current_state(self) -> str:
        if self.state in ACTIVE and not self.runner_alive():
            return "lost"
        return self.state

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["state"] = self.current_state()
        data["log"] = str(self.log_path)
        return data


def load(job_id: int) -> Job:
    try:
        text = (jobs_dir() / f"{job_id}.json").read_text()
    except FileNotFoundError:
        raise KeyError(f"no job {job_id}") from None
    return Job(**json.loads(text))


def all_jobs() -> list[Job]:
    found = []
    for path in jobs_dir().glob("*.json"):
        try:
            found.append(Job(**json.loads(path.read_text())))
        except (OSError, ValueError, TypeError):
            continue  # a record reserved but not yet written
    return sorted(found, key=lambda job: job.id)


def _create(**fields: Any) -> Job:
    directory = jobs_dir()
    while True:
        ids = (int(p.stem) for p in directory.glob("*.json") if p.stem.isdigit())
        next_id = max(ids, default=0) + 1
        try:
            os.close(os.open(directory / f"{next_id}.json", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            continue
        job = Job(id=next_id, **fields)
        job.save()
        return job


def submit(
    command: str,
    *,
    name: str = "",
    owner: str = "",
    cwd: str | None = None,
    timeout: float | None = None,
    grace: float = 10.0,
) -> Job:
    """Record a job and start its runner, which waits for the GPU in the background."""
    cwd = str(Path(cwd or os.getcwd()).resolve())
    job = _create(command=command, name=name, owner=owner, cwd=cwd, timeout=timeout, grace=grace)
    with open(job.log_path, "wb") as log:
        subprocess.run(
            [sys.executable, "-m", "poppler.runner", str(job.id)],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            check=True,
        )
    # The runner records its pid on startup; until then the job would read as lost.
    deadline = time.monotonic() + 10
    while (job := load(job.id)).runner_pid is None and job.state in ACTIVE:
        if time.monotonic() > deadline:
            raise RuntimeError(f"runner for job {job.id} did not start; see {job.log_path}")
        time.sleep(POLL / 10)
    return job


def wait(job_id: int, timeout: float | None = None) -> Job:
    """Block until the job leaves the active states, or until timeout seconds pass."""
    deadline = None if timeout is None else time.monotonic() + timeout
    while (job := load(job_id)).current_state() in ACTIVE:
        if deadline is not None and time.monotonic() >= deadline:
            break
        time.sleep(POLL)
    return job


def _signal_runner(job: Job, sig: signal.Signals) -> None:
    """Signal the job's runner, unless it has exited and its pid has been reused."""
    # The start time check leaves a gap of microseconds before the kill, far too
    # short for the pid counter to wrap around and hand the pid to someone else.
    assert job.runner_pid is not None
    if job.runner_alive():
        with contextlib.suppress(ProcessLookupError):
            os.kill(job.runner_pid, sig)


def cancel(job_id: int) -> Job:
    """Stop a job.

    A live runner gets SIGTERM and stops the command itself. A lost job's
    process group gets SIGKILL.
    """
    job = load(job_id)
    state = job.current_state()
    if state in ACTIVE:
        _signal_runner(job, signal.SIGTERM)
        return wait(job_id, timeout=job.grace + 5)
    if state == "lost":
        # The runner died without cleaning up, so whatever is left of the command's
        # group may still hold the GPU. A pgid stays reserved while any member
        # lives, so unless the leader's pid now belongs to a new process, this
        # signal can only reach the job's own processes.
        if job.child_pid is not None and process_start(job.child_pid) in (None, job.child_start):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(job.child_pid, signal.SIGKILL)
        job.state = "cancelled"
        job.finished_at = time.time()
        job.save()
    return job


def gpu_busy() -> bool:
    """Return whether a job holds the lock, by trying to take it without blocking."""
    with open(lock_path(), "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(lock, fcntl.LOCK_UN)
        return False


def gpu_memory() -> dict[str, int] | None:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        used, total, util = (int(x) for x in out.splitlines()[0].split(","))
    except (IndexError, ValueError):
        return None
    return {"memory_used_mib": used, "memory_total_mib": total, "utilization_pct": util}


def lock_holders() -> list[int]:
    """Return the pids of processes with the lock file open, waiting runners included."""
    target = str(lock_path())
    holders = []
    for fd_dir in Path("/proc").glob("[0-9]*/fd"):
        try:
            if any(os.readlink(fd) == target for fd in fd_dir.iterdir()):
                holders.append(int(fd_dir.parent.name))
        except OSError:
            continue  # exited, or not ours to inspect
    return sorted(holders)


def status() -> dict[str, Any]:
    by_state: dict[str, list[dict[str, Any]]] = {"running": [], "waiting": [], "lost": []}
    for job in all_jobs():
        if (state := job.current_state()) in by_state:
            by_state[state].append(job.to_dict())
    busy = gpu_busy()
    return {
        "gpu_busy": busy,
        **by_state,
        "lock_holders": lock_holders() if busy else [],
        "gpu": gpu_memory(),
    }


def read_log(job_id: int, tail_lines: int | None = None) -> str:
    text = load(job_id).log_path.read_bytes().decode(errors="replace")
    if tail_lines is not None:
        lines = text.splitlines(keepends=True)
        text = "".join(lines[len(lines) - tail_lines :]) if tail_lines > 0 else ""
    return text
