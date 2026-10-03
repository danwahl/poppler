"""Job records, the GPU lock, and the operations the CLI and MCP server share.

Each job is a JSON file in the jobs directory. submit creates it, the job's
runner (see runner.py) updates it, and cancel finishes it if the runner has
died. A record whose runner is gone reads as LOST, so a crashed runner never
leaves a job that looks active.

Each unfinished job also has an empty marker file in the active directory, so
the runners polling the queue read only those records.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import math
import os
import signal
import subprocess
import sys
import time
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args

# Slurm's job state names, plus LOST for a job whose runner died.
PENDING = "PENDING"
RUNNING = "RUNNING"
COMPLETED = "COMPLETED"
FAILED = "FAILED"
CANCELLED = "CANCELLED"
TIMEOUT = "TIMEOUT"
PREEMPTED = "PREEMPTED"
LOST = "LOST"
ACTIVE = (PENDING, RUNNING)
State = Literal[
    "PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "PREEMPTED", "LOST"
]
STATES: tuple[str, ...] = get_args(State)
POLL = 0.1
STARTUP = 10  # seconds a new runner has to record its pid

# Fixed QOS levels, as an admin might define them with sacctmgr. Priority
# orders the pending queue, and a pending job that is first in line preempts
# a running job whose QOS is in its preempt list.
QOS = {
    "high": {"priority": 2, "preempt": ("scavenger",)},
    "normal": {"priority": 1, "preempt": ("scavenger",)},
    "scavenger": {"priority": 0, "preempt": ()},
}

DEFAULT_CONFIG = {"KillWait": 30.0}


def home() -> Path:
    if path := os.environ.get("POPPLER_HOME"):
        return Path(path)
    state = os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state"
    return Path(state) / "poppler"


def jobs_dir() -> Path:
    path = home() / "jobs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def active_dir() -> Path:
    path = home() / "active"
    path.mkdir(parents=True, exist_ok=True)
    return path


def lock_path() -> Path:
    jobs_dir()
    return home() / "gpu.lock"


def config() -> dict[str, float]:
    """Read settings from $POPPLER_CONFIG, or config.toml in ~/.config/poppler.

    Settings take their names from slurm.conf. KillWait is the number of
    seconds between SIGTERM and SIGKILL when a job is stopped.
    """
    if path := os.environ.get("POPPLER_CONFIG"):
        file = Path(path)
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
        file = Path(base) / "poppler" / "config.toml"
    try:
        settings = tomllib.loads(file.read_text())
    except FileNotFoundError:
        settings = {}
    if unknown := settings.keys() - DEFAULT_CONFIG.keys():
        raise ValueError(f"unknown settings in {file}: {', '.join(sorted(unknown))}")
    values = {key: settings.get(key, value) for key, value in DEFAULT_CONFIG.items()}
    for key, value in values.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not 0 <= value < math.inf
        ):
            raise ValueError(f"{key} in {file} must be a number of seconds, not {value!r}")
    return {key: float(value) for key, value in values.items()}


def parse_time(text: str) -> float | None:
    """Parse a time limit in Slurm's format into seconds, or None for no limit.

    The formats are "minutes", "minutes:seconds", "hours:minutes:seconds",
    "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds".
    """
    if text.upper() in ("UNLIMITED", "INFINITE"):
        return None
    days, dash, rest = text.partition("-")
    parts = rest.split(":") if dash else days.split(":")
    if dash:
        units = (3600, 60, 1)[: len(parts)] if len(parts) <= 3 else ()
    else:
        units = {1: (60,), 2: (60, 1), 3: (3600, 60, 1)}.get(len(parts), ())
        days = "0"
    if not units or not all(part.isdigit() for part in (days, *parts)):
        raise ValueError(f"invalid time limit {text!r}")
    seconds = int(days) * 86400
    return float(seconds + sum(int(part) * unit for part, unit in zip(parts, units, strict=True)))


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
    time_limit: float | None = None
    kill_wait: float = DEFAULT_CONFIG["KillWait"]
    qos: str = "normal"
    requeue: bool = True
    dependency: str = ""
    env: dict[str, str] = field(default_factory=dict)
    restart_count: int = 0
    state: str = PENDING
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
        # Records can hold secrets passed in env, so only the owner may read them.
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with open(fd, "w") as f:
            f.write(json.dumps(asdict(self)))
        tmp.replace(self.path)

    def runner_alive(self) -> bool:
        return self.runner_pid is not None and process_start(self.runner_pid) == self.runner_start

    def current_state(self) -> str:
        if self.state in ACTIVE and not self.runner_alive():
            if self.runner_pid is None and time.time() - self.submitted_at < STARTUP:
                return self.state  # the runner is still starting
            return LOST
        return self.state

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["state"] = self.current_state()
        data["log"] = str(self.log_path)
        return data

    def summary(self) -> dict[str, Any]:
        """Return the fields worth showing in a list of jobs."""
        return {
            "id": self.id,
            "name": self.name,
            "owner": self.owner,
            "qos": self.qos,
            "state": self.current_state(),
            "exit_code": self.exit_code,
            "dependency": self.dependency,
            "time_limit": self.time_limit,
            "requeue": self.requeue,
            "restart_count": self.restart_count,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "command": self.command,
        }


def load(job_id: int) -> Job:
    try:
        text = (jobs_dir() / f"{job_id}.json").read_text()
    except FileNotFoundError:
        raise KeyError(f"no job {job_id}") from None
    return Job(**json.loads(text))


def load_settled(job_id: int) -> Job:
    """Load a job, reading it again if it looks lost.

    A runner that records an outcome and exits between the read and the
    liveness check would otherwise make a finished job look lost.
    """
    job = load(job_id)
    return load(job_id) if job.current_state() == LOST else job


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
        (active_dir() / str(job.id)).touch()
        return job


def active_jobs() -> list[Job]:
    """Return the pending, running and lost jobs."""
    found = []
    for marker in active_dir().iterdir():
        try:
            found.append(load(int(marker.name)))
        except (KeyError, OSError, ValueError, TypeError):
            continue  # finished since the directory was listed
    return sorted(found, key=lambda job: job.id)


def _priority(job: Job) -> tuple[int, int]:
    return -QOS[job.qos]["priority"], job.id


def select(owner: str | None = None, states: list[str] | None = None) -> list[Job]:
    """Return every job, oldest first, optionally only one owner's or those in some states."""
    return [
        job
        for job in all_jobs()
        if (owner is None or job.owner == owner) and (not states or job.current_state() in states)
    ]


def parse_dependency(text: str) -> list[tuple[str, list[int]]]:
    """Parse a Slurm dependency list such as "afterok:16:17,afterany:18".

    Every condition must hold. Slurm's other dependency types, and "?" for
    any one condition, are not supported.
    """
    conditions = []
    for part in filter(None, text.split(",")):
        kind, *ids = part.split(":")
        if kind not in ("afterok", "afterany") or not ids or not all(i.isdecimal() for i in ids):
            raise ValueError(
                f"invalid dependency {part!r}; use afterok:ID[:ID...] or afterany:ID[:ID...]"
            )
        conditions.append((kind, [int(i) for i in ids]))
    return conditions


# Ended jobs never change state, so each process remembers them.
_ended: dict[int, str] = {}


def _dependency_target_state(job_id: int) -> str | None:
    """Return a dependency target's state, or None if its record is gone."""
    if job_id in _ended:
        return _ended[job_id]
    try:
        state = load_settled(job_id).current_state()
    except KeyError:
        return None
    except (OSError, ValueError, TypeError):
        return PENDING  # reserved but not yet written
    if state not in (*ACTIVE, LOST):
        _ended[job_id] = state
    return state


def dependency_state(job: Job) -> str:
    """Return "met", "waiting" or "never" for the job's dependencies.

    afterok needs each job to end COMPLETED; afterany needs each job to end in
    any state. A LOST job has not ended, since it may still be cancelled. A
    job whose record is gone can never satisfy a dependency.
    """
    try:
        conditions = parse_dependency(job.dependency)
    except ValueError:
        return "never"
    met = True
    for kind, ids in conditions:
        for dep_id in ids:
            state = _dependency_target_state(dep_id)
            if state is None:
                return "never"
            if state in (*ACTIVE, LOST):
                met = False
            elif kind == "afterok" and state != COMPLETED:
                return "never"
    return "met" if met else "waiting"


def queue() -> list[Job]:
    """Return the pending jobs that may start, in the order they will start.

    Jobs whose dependencies are not met are left out. The rest are ordered by
    QOS priority, then id.
    """
    pending = (job for job in active_jobs() if job.current_state() == PENDING)
    return sorted((job for job in pending if dependency_state(job) == "met"), key=_priority)


def may_preempt(pending: Job, running: Job) -> bool:
    return running.qos in QOS[pending.qos]["preempt"]


def finish(job: Job, state: str, exit_code: int | None = None) -> None:
    """Record a job's outcome and remove it from the active jobs."""
    job.state = state
    job.exit_code = exit_code
    job.finished_at = time.time()
    job.save()
    (active_dir() / str(job.id)).unlink(missing_ok=True)


def submit(
    command: str,
    *,
    name: str = "",
    owner: str = "",
    cwd: str | None = None,
    time_limit: float | None = None,
    qos: str = "normal",
    requeue: bool = True,
    dependency: str = "",
    env: dict[str, str] | None = None,
) -> Job:
    """Record a job and start its runner, which waits for the GPU in the background.

    A time limit of 0 means no limit, as in Slurm. env adds to the environment
    the job inherits from the caller.
    """
    if qos not in QOS:
        raise ValueError(f"unknown QOS {qos!r}; choose from {', '.join(QOS)}")
    conditions = parse_dependency(dependency)
    for _, ids in conditions:
        for dep_id in ids:
            load(dep_id)  # a KeyError names a missing job
    dependency = ",".join(f"{kind}:{':'.join(map(str, ids))}" for kind, ids in conditions)
    env = env or {}
    if bad := [key for key in env if not key or "=" in key or "\0" in key]:
        raise ValueError(f"invalid environment variable name {bad[0]!r}")
    if bad := [key for key, value in env.items() if "\0" in value]:
        raise ValueError(f"environment variable {bad[0]} contains a null byte")
    cwd = str(Path(cwd or os.getcwd()).resolve())
    job = _create(
        command=command,
        name=name,
        owner=owner,
        cwd=cwd,
        time_limit=time_limit or None,
        kill_wait=config()["KillWait"],
        qos=qos,
        requeue=requeue,
        dependency=dependency,
        env=env,
    )
    with open(job.log_path, "wb") as log:
        subprocess.run(
            [sys.executable, "-m", "poppler.runner", str(job.id)],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            check=True,
        )
    job = _started(job.id)
    if job.runner_pid is None:
        raise RuntimeError(f"runner for job {job.id} did not start; see {job.log_path}")
    return job


def _started(job_id: int) -> Job:
    """Wait for the job's runner to record its pid, for up to STARTUP seconds after submission."""
    while (job := load(job_id)).current_state() in ACTIVE and job.runner_pid is None:
        time.sleep(POLL / 10)
    return job


def wait(job_id: int, timeout: float | None = None) -> Job:
    """Block until the job leaves the active states, or until timeout seconds pass."""
    deadline = None if timeout is None else time.monotonic() + timeout
    while (job := load_settled(job_id)).current_state() in ACTIVE:
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


def _kill_lost(job: Job, state: str) -> None:
    """Kill what is left of a lost job's processes and record the state."""
    # The runner died without cleaning up, so whatever is left of the command's
    # group may still hold the GPU. A pgid stays reserved while any member
    # lives, so unless the leader's pid now belongs to a new process, this
    # signal can only reach the job's own processes.
    if job.child_pid is not None and process_start(job.child_pid) in (None, job.child_start):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(job.child_pid, signal.SIGKILL)
    finish(job, state)


def preempt(job: Job) -> None:
    """Stop a running job so a pending one can take the GPU.

    A live runner stops the job and requeues it, or ends it as PREEMPTED if it
    was submitted without requeue. A lost job's processes get SIGKILL, and it
    ends as PREEMPTED.
    """
    if job.current_state() == LOST:
        _kill_lost(job, PREEMPTED)
    else:
        _signal_runner(job, signal.SIGUSR1)


def cancel(job_id: int) -> Job:
    """Stop a job.

    A live runner gets SIGTERM and stops the command itself. A lost job's
    process group gets SIGKILL.
    """
    _started(job_id)
    job = load_settled(job_id)
    state = job.current_state()
    if state in ACTIVE:
        _signal_runner(job, signal.SIGTERM)
        return wait(job_id, timeout=job.kill_wait + 5)
    if state == LOST:
        _kill_lost(job, CANCELLED)
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
    """Return the pids of processes with the lock file open."""
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
    """Return the GPU's state and summaries of the active jobs.

    Pending jobs are in start order, each with Slurm's reason for waiting.
    Jobs waiting on dependencies come last.
    """
    by_state: dict[str, list[Job]] = {RUNNING: [], PENDING: [], LOST: []}
    for job in active_jobs():
        if (state := job.current_state()) in by_state:
            by_state[state].append(job)
    deps = {job.id: dependency_state(job) for job in by_state[PENDING]}
    by_state[PENDING].sort(key=lambda job: (deps[job.id] != "met", *_priority(job)))
    reasons = {"waiting": "Dependency", "never": "DependencyNeverSatisfied"}
    pending = []
    for job in by_state[PENDING]:
        reason = reasons.get(deps[job.id]) or ("Priority" if pending else "Resources")
        pending.append({**job.summary(), "reason": reason})
    busy = gpu_busy()
    return {
        "gpu_busy": busy,
        "running": [job.summary() for job in by_state[RUNNING]],
        "pending": pending,
        "lost": [job.summary() for job in by_state[LOST]],
        "lock_holders": lock_holders() if busy else [],
        "gpu": gpu_memory(),
    }


def read_log(job_id: int, tail_lines: int | None = None) -> str:
    text = load(job_id).log_path.read_bytes().decode(errors="replace")
    if tail_lines is not None:
        lines = text.splitlines(keepends=True)
        text = "".join(lines[len(lines) - tail_lines :]) if tail_lines > 0 else ""
    return text
