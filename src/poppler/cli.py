"""Command-line interface: `poppler run|status|list|log|cancel|mcp`."""

from __future__ import annotations

import argparse
import getpass
import json
import shlex
import signal
import sys
import time
from typing import BinaryIO

from poppler import jobs


def _exit_status(job: jobs.Job) -> int:
    if job.current_state() == jobs.COMPLETED:
        return 0
    if job.exit_code is not None and job.exit_code > 0:
        return job.exit_code
    if job.exit_code is not None and job.exit_code < 0:
        return 128 - job.exit_code
    return 1


def _follow(job_id: int, out: BinaryIO) -> jobs.Job:
    """Copy the job's log to out until the job finishes."""
    job = jobs.load(job_id)
    with open(job.log_path, "rb") as log:
        while True:
            job = jobs.load_settled(job_id)
            done = job.current_state() not in jobs.ACTIVE
            if chunk := log.read():
                out.write(chunk)
                out.flush()
            if done:
                return job
            time.sleep(jobs.POLL)


def _age(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


def _table(rows: list[jobs.Job]) -> str:
    now = time.time()
    lines = [f"{'ID':>4}  {'STATE':<9}  {'QOS':<9}  {'OWNER':<12}  {'AGE':>6}  COMMAND"]
    for job in rows:
        label = f"[{job.name}] " if job.name else ""
        lines.append(
            f"{job.id:>4}  {job.current_state():<9}  {job.qos:<9}  {job.owner[:12]:<12}  "
            f"{_age(now - job.submitted_at):>6}  {label}{job.command}"[:160]
        )
    return "\n".join(lines)


def cmd_run(args: argparse.Namespace) -> int:
    if not args.command:
        sys.exit("poppler run: no command given")
    command = args.command[0] if len(args.command) == 1 else shlex.join(args.command)
    # Submitting takes a moment; an interrupt then would orphan a job we can't name.
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        job = jobs.submit(
            command,
            name=args.job_name,
            owner=args.owner or getpass.getuser(),
            time_limit=args.time,
            qos=args.qos,
            requeue=args.requeue,
        )
    finally:
        signal.signal(signal.SIGINT, previous)
    if args.detach:
        print(job.id)
        return 0
    print(f"poppler: job {job.id} submitted", file=sys.stderr)
    try:
        job = _follow(job.id, sys.stdout.buffer)
    except KeyboardInterrupt:
        print(f"\npoppler: cancelling job {job.id}", file=sys.stderr)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        job = jobs.cancel(job.id)
    print(f"poppler: job {job.id} {job.current_state()}", file=sys.stderr)
    return _exit_status(job)


def cmd_status(args: argparse.Namespace) -> int:
    info = jobs.status()
    if args.json:
        print(json.dumps(info, indent=2))
        return 0
    print(f"GPU: {'busy' if info['gpu_busy'] else 'free'}", end="")
    if gpu := info["gpu"]:
        print(
            f", {gpu['memory_used_mib']}/{gpu['memory_total_mib']} MiB used, "
            f"{gpu['utilization_pct']}% utilization",
            end="",
        )
    print()
    if info["gpu_busy"] and not info["running"]:
        holders = " ".join(map(str, info["lock_holders"]))
        print(f"The lock is held outside any running job, by pids: {holders}")
    active = [jobs.load(j["id"]) for state in ("running", "pending", "lost") for j in info[state]]
    if active:
        print(_table(active))
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    rows = jobs.all_jobs()[-args.limit :] if args.limit else jobs.all_jobs()
    if args.json:
        print(json.dumps([job.to_dict() for job in rows], indent=2))
    else:
        print(_table(rows))
    return 0


def cmd_log(args: argparse.Namespace) -> int:
    if args.follow:
        _follow(args.id, sys.stdout.buffer)
    else:
        sys.stdout.write(jobs.read_log(args.id, args.tail))
    return 0


def cmd_cancel(args: argparse.Namespace) -> int:
    job = jobs.cancel(args.id)
    print(f"job {job.id} {job.current_state()}")
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    from poppler.server import mcp

    mcp.run()
    return 0


def _time(text: str) -> float | None:
    try:
        return jobs.parse_time(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def _count(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return value


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="poppler", description="Take turns on a single GPU.")
    sub = p.add_subparsers(required=True, metavar="COMMAND")

    run = sub.add_parser("run", help="run a command once the GPU is free")
    run.add_argument("-J", "--job-name", default="", help="short label for the job")
    run.add_argument("--owner", help="who is submitting (default: your username)")
    run.add_argument(
        "-t",
        "--time",
        type=_time,
        help="time limit, as minutes, [hours:]minutes:seconds or days-hours[:minutes[:seconds]]",
    )
    run.add_argument(
        "-q",
        "--qos",
        choices=list(jobs.QOS),
        default="normal",
        help="start order high, normal, scavenger; high and normal jobs preempt "
        "running scavenger jobs (default normal)",
    )
    run.add_argument(
        "--requeue",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="when preempted, go back to pending instead of ending as PREEMPTED "
        "(default --requeue)",
    )
    run.add_argument("-d", "--detach", action="store_true", help="print the job id and return")
    run.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="command to run, after --; a single argument runs as a shell string",
    )
    run.set_defaults(func=cmd_run)

    status = sub.add_parser("status", help="show GPU state and active jobs")
    status.add_argument("--json", action="store_true", help="print JSON")
    status.set_defaults(func=cmd_status)

    lst = sub.add_parser("list", help="list recent jobs")
    lst.add_argument(
        "-n", "--limit", type=_count, default=20, help="jobs to show (default 20, 0 for all)"
    )
    lst.add_argument("--json", action="store_true", help="print JSON")
    lst.set_defaults(func=cmd_list)

    log = sub.add_parser("log", help="print a job's output")
    log.add_argument("id", type=int)
    log.add_argument("-f", "--follow", action="store_true", help="keep printing until it ends")
    log.add_argument("-n", "--tail", type=_count, help="only the last N lines")
    log.set_defaults(func=cmd_log)

    cancel = sub.add_parser("cancel", help="stop a pending, running or lost job")
    cancel.add_argument("id", type=int)
    cancel.set_defaults(func=cmd_cancel)

    mcp = sub.add_parser("mcp", help="serve the MCP tools over stdio")
    mcp.set_defaults(func=cmd_mcp)
    return p


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if getattr(args, "command", None) and args.command[0] == "--":
        args.command = args.command[1:]
    try:
        sys.exit(args.func(args))
    except (KeyError, ValueError) as e:
        sys.exit(f"poppler: {e.args[0]}")
