from __future__ import annotations

import argparse
import os
import signal
import subprocess
from pathlib import Path
from typing import Any

import mcp_bridge
from job_status import update_status


class RunnerCancelled(KeyboardInterrupt):
    def __init__(self, signal_value: int | None = None) -> None:
        super().__init__()
        self.signal_value = signal_value


def run_job(
    *,
    job_id: str,
    status_path: str | Path,
    runtime_config_path: str | Path,
    timeout_sec: int | None = None,
) -> dict[str, Any]:
    resolved_status_path = Path(status_path)
    log_path = _log_path_for_job(resolved_status_path, job_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = mcp_bridge.build_command(runtime_config_path)
    child: subprocess.Popen[Any] | None = None
    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)

    def _handle_sigterm(signum: int, _frame: object) -> None:
        raise RunnerCancelled(signum)

    try:
        signal.signal(signal.SIGTERM, _handle_sigterm)
        with log_path.open("a", encoding="utf-8") as log_handle:
            try:
                child = subprocess.Popen(
                    command,
                    cwd=str(mcp_bridge.PROJECT_ROOT),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                update_status(
                    resolved_status_path,
                    status="running",
                    pid=child.pid,
                    runner_pid=os.getpid(),
                    log_path=str(log_path),
                )
                exit_code = child.wait(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                raise
            except (KeyboardInterrupt, RunnerCancelled):
                raise
            except Exception:
                exit_code = _stop_child(child)
                _best_effort_failed_status(
                    resolved_status_path,
                    exit_code=exit_code,
                    log_path=log_path,
                    error={
                        "code": "runner_setup_failed",
                        "message": f"Runner setup failed for job {job_id}",
                    },
                )
                raise
    except (KeyboardInterrupt, RunnerCancelled) as exc:
        exit_code = _stop_child(child)
        signal_value = exc.signal_value if isinstance(exc, RunnerCancelled) else None
        return update_status(
            resolved_status_path,
            status="cancelled",
            exit_code=exit_code,
            log_path=str(log_path),
            **({"signal": signal_value} if signal_value is not None else {}),
        )
    except subprocess.TimeoutExpired:
        if child is not None:
            child.kill()
            exit_code = child.wait()
        else:
            exit_code = None
        return update_status(
            resolved_status_path,
            status="failed",
            exit_code=exit_code,
            log_path=str(log_path),
            error={
                "code": "timeout",
                "message": f"Job {job_id} exceeded timeout of {timeout_sec} seconds",
            },
        )
    except OSError as exc:
        return update_status(
            resolved_status_path,
            status="failed",
            exit_code=None,
            log_path=str(log_path),
            error={"code": "spawn_failed", "message": str(exc)},
        )
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm_handler)

    final_status = "succeeded" if exit_code == 0 else "failed"
    return update_status(
        resolved_status_path,
        status=final_status,
        exit_code=exit_code,
        log_path=str(log_path),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a tracked MCP job")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--status-path", required=True)
    parser.add_argument("--runtime-config-path", required=True)
    parser.add_argument("--timeout-sec", type=int)
    args = parser.parse_args()

    result = run_job(
        job_id=args.job_id,
        status_path=args.status_path,
        runtime_config_path=args.runtime_config_path,
        timeout_sec=args.timeout_sec,
    )
    exit_code = result.get("exit_code")
    if isinstance(exit_code, int):
        return exit_code
    return 0 if result.get("status") == "succeeded" else 1


def _log_path_for_job(status_path: Path, job_id: str) -> Path:
    parent = status_path.parent
    root = parent.parent if parent.name == "jobs" else parent
    return root / "logs" / f"{job_id}.log"


def _stop_child(child: subprocess.Popen[Any] | None) -> int | None:
    if child is None:
        return None

    process_group_id: int | None = None
    getpgid = getattr(os, "getpgid", None)
    killpg = getattr(os, "killpg", None)
    if callable(getpgid) and callable(killpg):
        try:
            process_group_id = getpgid(child.pid)
        except OSError:
            process_group_id = None

    if process_group_id is not None:
        killpg(process_group_id, signal.SIGTERM)
    else:
        child.terminate()

    try:
        return child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        child.kill()
        return child.wait()


def _best_effort_failed_status(
    status_path: Path,
    *,
    exit_code: int | None,
    log_path: Path,
    error: dict[str, Any],
) -> None:
    try:
        update_status(
            status_path,
            status="failed",
            exit_code=exit_code,
            log_path=str(log_path),
            error=error,
        )
    except Exception:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
