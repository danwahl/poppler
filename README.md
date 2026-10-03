# poppler

poppler lets the people and agents sharing a single-GPU machine take turns on it. A submitted job waits until the GPU is free, runs, and releases it when it exits. The same queue is available as a CLI and as an MCP server, so agents can submit work, wait on it and read its output without a wrapper.

## How it works

The GPU is guarded by one `flock` on `gpu.lock` in `$POPPLER_HOME`, or `~/.local/state/poppler` by default. Each job gets a small runner process that waits until its job is first in the queue and the lock is free, runs the command with `/bin/sh -c` in its own process group, and records the outcome.

The command inherits the lock's file descriptor, so the kernel releases the lock when the last job process exits, even if the job crashes on startup or is killed with SIGKILL. No daemon is involved.

When a job's main process exits, anything it left running in its process group is killed. Cancelling or preempting a job, or reaching its time limit, sends SIGTERM to the group, then SIGKILL to anything still running after `KillWait` seconds, so a job that handles SIGTERM has time to save a checkpoint.

Job records and logs live in the `jobs/` directory next to the lock. Jobs move through Slurm's states: `PENDING`, `RUNNING`, then `COMPLETED`, `FAILED`, `CANCELLED`, `TIMEOUT` or `PREEMPTED`. A job whose runner died without recording an outcome shows as `LOST`. Its processes may still hold the GPU, and `poppler cancel` kills them. When the GPU is busy but no job is running, `poppler status` lists the pids holding the lock.

## Install

```sh
uv tool install git+https://github.com/danwahl/poppler
```

## CLI

```sh
poppler run -- uv run python train.py --epochs 3   # wait, run, stream output
poppler run --detach -J sweep -t 1:00:00 -- ./sweep.sh   # print the job id and return
poppler run --detach -d afterok:12 -- ./report.sh        # start once job 12 completes
poppler status          # GPU state, active jobs, and why pending jobs wait
poppler list -u dan -t FAILED   # recent jobs, filtered by owner and state
poppler log 12 -f       # follow a job's output
poppler cancel 12
```

`poppler run` exits with the job's exit code, or 128+N if signal N ended it. Any other job that does not complete, such as one cancelled before it started or lost, exits with 1. Pressing Ctrl-C while it waits or streams output cancels the job. A command given as a single argument runs as a shell string, so `poppler run -- "python a.py | tee out.txt"` works.

`-t` takes a time limit in Slurm's formats: `minutes`, `minutes:seconds`, `hours:minutes:seconds`, `days-hours`, `days-hours:minutes`, `days-hours:minutes:seconds`. `0` or `UNLIMITED` means no limit.

`-d`/`--dependency` takes Slurm's syntax. `afterok:ID[:ID...]` waits for those jobs to complete, and `afterany:ID[:ID...]` waits for them to end in any state. Join conditions with commas; all of them must hold. If a job named by `afterok` ends in any state other than `COMPLETED`, the dependent job is cancelled, as Slurm does with `kill_invalid_depend`. A job waiting on a dependency does not hold up the jobs behind it, and does not preempt anything.

`poppler status` gives each pending job Slurm's reason for waiting: `Resources` for the next job to start once the GPU is free, `Priority` for jobs behind it, `Dependency` for jobs waiting on others, and `DependencyNeverSatisfied` for a job about to be cancelled.

Jobs run in the current directory, or the one given by `-D`/`--chdir`. They inherit the environment of whoever submitted them, plus any variables given with `--export=ALL,NAME=value[,...]`. They also see `POPPLER_JOB_ID` and `POPPLER_RESTART_COUNT`.

## QOS and preemption

Every job has one of three fixed QOS levels:

| QOS | Starts | Preempts |
|---|---|---|
| `high` | before `normal` and `scavenger` jobs | `scavenger` |
| `normal` (default) | before `scavenger` jobs | `scavenger` |
| `scavenger` | when nothing else is pending | nothing |

Within a level, pending jobs start in submission order. When the job at the front of the queue may preempt the running job, the running job gets SIGTERM, then SIGKILL after `KillWait` seconds. A preempted job is requeued by default: it goes back to `PENDING` under the same id and runs again from the start, with `POPPLER_RESTART_COUNT` one higher. With `--no-requeue` it ends as `PREEMPTED` instead. A `LOST` job has no runner to requeue it, so a preemptible job that is `LOST` while its processes still hold the GPU gets SIGKILL at once and ends as `PREEMPTED`.

`scavenger` suits long, resumable work that should use whatever GPU time is left over, such as a training run that saves a checkpoint on SIGTERM and resumes from it:

```sh
poppler run --detach -q scavenger -J train -- uv run python train.py --resume
poppler run -- uv run python eval.py   # preempts train, which reruns when eval ends
poppler run -q high -- ./smoke.sh      # jumps ahead of pending normal jobs
```

Running `normal` and `high` jobs are never preempted, so a `high` job waits for them to finish.

## Configuration

Settings go in `~/.config/poppler/config.toml`, or the file named by `$POPPLER_CONFIG`, and take their names from `slurm.conf`:

```toml
KillWait = 30   # seconds between SIGTERM and SIGKILL when a job is stopped
```

A job keeps the `KillWait` in effect when it was submitted.

## MCP

```sh
claude mcp add --scope user poppler -- poppler mcp
```

The server's tools are `gpu_status`, `submit`, `wait_job`, `job_log`, `list_jobs`, `job_info` and `cancel_job`. `gpu_status` and `list_jobs` return short summaries, and `list_jobs` filters by owner and state. The server's instructions tell agents to run GPU work through `submit`, identify themselves with `owner`, use the `normal` QOS unless asked otherwise, keep commands that cannot resume out of `scavenger`, checkpoint on SIGTERM and cancel only their own jobs. A job's working directory defaults to the server's, so agents should pass `cwd`.

## Slurm equivalents

poppler borrows Slurm's names where it has the same feature, and leaves out the rest.

| poppler | Slurm |
|---|---|
| `poppler run` | `srun` |
| `poppler run --detach` | `sbatch` |
| `poppler status`, `poppler list` | `squeue`, `sacct` |
| `poppler cancel` | `scancel` |
| `-J`/`--job-name`, `-t`/`--time`, `-q`/`--qos`, `--requeue`/`--no-requeue`, `-D`/`--chdir` | the same options |
| `-d`/`--dependency` with `afterok` and `afterany` | the same option, with `SchedulerParameters=kill_invalid_depend` |
| `--export=ALL,NAME=value` | the same option; other forms are not supported |
| `poppler list -u`/`--owner`, `-t`/`--states` | `squeue -u`/`--user`, `-t`/`--states` |
| pending reasons `Resources`, `Priority`, `Dependency`, `DependencyNeverSatisfied` | the same reasons |
| job states other than `LOST` | the same states |
| `KillWait` | `KillWait` in `slurm.conf` |
| the three QOS levels | QOS `Priority` and `Preempt` set with `sacctmgr`, with `PriorityWeightQOS`, `PreemptType=preempt/qos` and `PreemptMode=REQUEUE` |
| `POPPLER_JOB_ID`, `POPPLER_RESTART_COUNT` | `SLURM_JOB_ID`, `SLURM_RESTART_COUNT` |

There are no partitions, accounts, fair-share, `--nice`, job arrays, other dependency types, `GraceTime` or suspend-based preemption. The QOS levels cannot be configured.

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

## License

MIT
