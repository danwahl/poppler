# poppler

poppler lets the people and agents sharing a single-GPU machine take turns on it. A submitted job waits until the GPU is free, runs, and releases it when it exits. The same queue is available as a CLI and as an MCP server, so agents can submit work, wait on it and read its output without a wrapper.

## How it works

The GPU is guarded by one `flock` on `gpu.lock` in `$POPPLER_HOME`, or `~/.local/state/poppler` by default. Each job gets a small runner process that waits until its job is first in the queue and the lock is free, runs the command with `/bin/sh -c` in its own process group, and records the outcome.

The command inherits the lock's file descriptor, so the kernel releases the lock when the last job process exits, even if the job crashes on startup or is killed with SIGKILL. No daemon is involved.

When a job's main process exits, anything it left running in its process group is killed. Cancelling a job or reaching its time limit sends SIGTERM to the group, then SIGKILL to anything still running after `KillWait` seconds, so a job that handles SIGTERM has time to save a checkpoint.

Job records and logs live in the `jobs/` directory next to the lock. Jobs move through Slurm's states: `PENDING`, `RUNNING`, then `COMPLETED`, `FAILED`, `CANCELLED` or `TIMEOUT`. A job whose runner died without recording an outcome shows as `LOST`. Its processes may still hold the GPU, and `poppler cancel` kills them. When the GPU is busy but no job is running, `poppler status` lists the pids holding the lock.

## Install

```sh
uv tool install git+https://github.com/danwahl/poppler
```

## CLI

```sh
poppler run -- uv run python train.py --epochs 3   # wait, run, stream output
poppler run -d -J sweep -t 1:00:00 -- ./sweep.sh   # print the job id and return
poppler status          # GPU state and active jobs
poppler list            # recent jobs
poppler log 12 -f       # follow a job's output
poppler cancel 12
```

`poppler run` exits with the job's exit code, or 128+N if signal N ended it. A job cancelled before it started, or lost, exits with 1. Pressing Ctrl-C while it waits or streams output cancels the job. A command given as a single argument runs as a shell string, so `poppler run -- "python a.py | tee out.txt"` works.

`-t` takes a time limit in Slurm's formats: `minutes`, `minutes:seconds`, `hours:minutes:seconds`, `days-hours`, `days-hours:minutes` or `days-hours:minutes:seconds`.

Jobs see `POPPLER_JOB_ID` in their environment.

## Configuration

Settings go in `~/.config/poppler/config.toml`, or the file named by `$POPPLER_CONFIG`, and take their names from `slurm.conf`:

```toml
KillWait = 30   # seconds between SIGTERM and SIGKILL when a job is stopped
```

## MCP

```sh
claude mcp add --scope user poppler -- poppler mcp
```

The server's tools are `gpu_status`, `submit`, `wait_job`, `job_log`, `list_jobs` and `cancel_job`. Its instructions tell agents to run GPU work through `submit`, identify themselves with `owner`, checkpoint on SIGTERM and cancel only their own jobs. A job's working directory defaults to the server's, so agents should pass `cwd`.

## Limits

- One Unix account: the lock lives in that user's state directory.
- Only cooperating users are covered. A process that uses the GPU without going through poppler is invisible to it.
- One job holds the GPU at a time. Jobs cannot share it, even when their combined memory would fit.
- Linux only: liveness checks read `/proc`.
- Old job records and logs are never deleted. Pending runners read only the unfinished jobs, so old records do not slow the queue.

## Development

```sh
uv sync
uv run pytest
uv run ruff check && uv run ruff format --check
```
