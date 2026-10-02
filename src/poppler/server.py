"""MCP server exposing poppler's job queue to agents over stdio."""

from __future__ import annotations

from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations
from pydantic import Field

from poppler import jobs

INSTRUCTIONS = """\
This machine has one GPU, shared by several agents and people. poppler gives \
it to one job at a time: a submitted job is PENDING until the GPU is free, \
then RUNNING, and releases the GPU when it exits.

- Run every command that uses the GPU through `submit`, not directly. That \
includes training, evaluation and inference: anything that allocates GPU \
memory. CPU-only work does not need poppler.
- Pass `cwd` as an absolute path. The default is the server's working \
directory, which may not be your project.
- Set `owner` to a name that identifies you, so others can see whose job is \
running.
- Pending jobs start in submission order.
- A job holds the GPU until it exits, so do not submit long-lived servers \
unless the user asks for one.
- When a job is cancelled or reaches its time limit, its processes get \
SIGTERM, then SIGKILL after the machine's KillWait setting (30 seconds unless \
configured). Long jobs should save a checkpoint on SIGTERM.
- Submitting jobs, waiting on them and reading logs need no confirmation. \
Cancel only your own jobs unless the user asks otherwise.
"""

mcp = MCPServer("poppler", instructions=INSTRUCTIONS)

READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False)

JobId = Annotated[int, Field(description="Job id, as returned by submit or list_jobs.")]


@mcp.tool(annotations=READ_ONLY)
def gpu_status() -> dict[str, Any]:
    """Show whether the GPU is busy, GPU memory and utilization, and active jobs.

    LOST jobs had their runner die without recording an outcome;
    their command may still hold the GPU until cancelled. lock_holders lists
    the pids with the lock open whenever the GPU is busy.
    """
    return jobs.status()


@mcp.tool(annotations=ToolAnnotations(destructiveHint=False, openWorldHint=False))
def submit(
    command: Annotated[
        str,
        Field(description="Shell command to run, e.g. 'uv run python train.py --epochs 3'."),
    ],
    owner: Annotated[
        str, Field(description="Who is submitting: your agent or session name.")
    ] = "agent",
    name: Annotated[str, Field(description="Short label for the job.")] = "",
    cwd: Annotated[
        str | None,
        Field(
            description="Working directory, preferably absolute. "
            "Defaults to the MCP server's working directory."
        ),
    ] = None,
    time_limit_s: Annotated[
        float | None,
        Field(
            ge=0, description="Stop the job after this many seconds of running. No limit if unset."
        ),
    ] = None,
) -> dict[str, Any]:
    """Queue a command for the GPU and return its job record without waiting.

    The job starts once the GPU is free. Output goes to the job's log. Use
    wait_job to block until it finishes and job_log to read its output.
    """
    job = jobs.submit(command, name=name, owner=owner, cwd=cwd, time_limit=time_limit_s)
    return job.to_dict()


@mcp.tool(annotations=READ_ONLY)
def wait_job(
    job_id: JobId,
    timeout_s: Annotated[
        float,
        Field(ge=0, description="Return after this many seconds even if the job is still active."),
    ] = 60.0,
) -> dict[str, Any]:
    """Wait for a job to finish and return its record.

    If the job is still pending or running when timeout_s passes, the record
    shows that state; call again to keep waiting.
    """
    return jobs.wait(job_id, timeout_s).to_dict()


@mcp.tool(annotations=READ_ONLY)
def job_log(
    job_id: JobId,
    tail_lines: Annotated[
        int | None,
        Field(ge=1, description="Return only the last N lines. Pass null for the whole log."),
    ] = 100,
) -> str:
    """Return a job's combined stdout and stderr."""
    return jobs.read_log(job_id, tail_lines)


@mcp.tool(annotations=READ_ONLY)
def list_jobs(
    limit: Annotated[
        int, Field(ge=1, description="How many of the most recent jobs to return.")
    ] = 20,
) -> list[dict[str, Any]]:
    """List recent jobs, oldest first, in every state."""
    return [job.to_dict() for job in jobs.all_jobs()[-limit:]]


@mcp.tool(annotations=ToolAnnotations(destructiveHint=True, openWorldHint=False))
def cancel_job(job_id: JobId) -> dict[str, Any]:
    """Cancel a pending, running or lost job and return its record.

    A running job gets SIGTERM, then SIGKILL after KillWait seconds; a lost
    job's processes get SIGKILL. A job that is slow to stop may still show as
    running; check it again with wait_job.
    """
    return jobs.cancel(job_id).to_dict()
