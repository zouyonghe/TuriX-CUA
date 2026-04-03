from __future__ import annotations

import copy
import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from config_env import resolve_env_placeholders
from job_status import read_status, update_status, write_status


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_CANDIDATES = ("examples/config.json",)
EXAMPLE_CONFIG_CANDIDATES = ("examples/config.example.json",)
DEFAULT_TEMP_DIR = PROJECT_ROOT / ".turix_tmp" / "mcp"
DEFAULT_OUTPUT_LIMIT = 4000


class BridgeInputError(ValueError):
    """Raised when the bridge receives invalid user input."""


def get_example_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = _resolve_example_config_path(config_path)
    return _load_config(path, resolve_env=True)


def build_runtime_config(
    base_config: dict[str, Any],
    *,
    task: str | None,
    use_plan: bool | None,
    use_skills: bool | None,
    resume: bool | None,
    agent_id: str | None,
    max_steps: int | None,
) -> dict[str, Any]:
    runtime_config = copy.deepcopy(base_config)
    agent_cfg = runtime_config.setdefault("agent", {})
    normalized_agent_id = agent_id.strip() if agent_id is not None else None

    if task is not None:
        if not task.strip():
            raise BridgeInputError("task must not be empty")
        agent_cfg["task"] = task

    if use_plan is not None:
        agent_cfg["use_plan"] = bool(use_plan)

    if use_skills is not None:
        agent_cfg["use_skills"] = bool(use_skills)

    if resume is not None:
        if resume and not normalized_agent_id:
            raise BridgeInputError("agent_id is required when resume is true")
        agent_cfg["resume"] = bool(resume)

    if agent_id is not None:
        if not normalized_agent_id:
            raise BridgeInputError("agent_id must not be empty")
        agent_cfg["agent_id"] = normalized_agent_id

    if max_steps is not None:
        agent_cfg["max_steps"] = max_steps

    return runtime_config


def write_runtime_config(
    config: dict[str, Any], output_dir: Path | None = None
) -> Path:
    target_dir = Path(output_dir or DEFAULT_TEMP_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        prefix="turix-mcp-",
        dir=target_dir,
        delete=False,
    ) as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        return Path(handle.name)


def build_command(config_path: str | Path) -> list[str]:
    return [
        sys.executable,
        str(_main_entrypoint_path()),
        "-c",
        str(Path(config_path)),
    ]


def build_runner_command(
    *,
    job_id: str,
    status_path: str | Path,
    runtime_config_path: str | Path,
    timeout_sec: int | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(_runner_entrypoint_path()),
        "--job-id",
        job_id,
        "--status-path",
        str(Path(status_path)),
        "--runtime-config-path",
        str(Path(runtime_config_path)),
    ]
    if timeout_sec is not None:
        command.extend(["--timeout-sec", str(timeout_sec)])
    return command


def run_task_bridge(
    *,
    task: str,
    config_path: str | Path | None = None,
    use_plan: bool | None = None,
    use_skills: bool | None = None,
    resume: bool | None = False,
    agent_id: str | None = None,
    max_steps: int | None = None,
    dry_run: bool = False,
    timeout_sec: int | None = None,
) -> dict[str, Any]:
    if not task or not task.strip():
        raise BridgeInputError("task must not be empty")
    if agent_id is not None and not agent_id.strip():
        raise BridgeInputError("agent_id must not be empty")

    base_config = _load_config(_resolve_config_path(config_path), resolve_env=False)
    runtime_config = build_runtime_config(
        base_config,
        task=task,
        use_plan=use_plan,
        use_skills=use_skills,
        resume=resume,
        agent_id=agent_id,
        max_steps=max_steps,
    )

    if dry_run:
        runtime_config_path = write_runtime_config(runtime_config)
        command = build_command(runtime_config_path)
        return {
            "status": "dry_run",
            "command": command,
            "config_path": str(_resolve_config_path(config_path)),
            "runtime_config_path": str(runtime_config_path),
            "exit_code": None,
            "stdout": "",
            "stderr": "",
        }

    startup_error = _launch_startup_error()
    if startup_error:
        runtime_config_path = write_runtime_config(runtime_config)
        command = build_command(runtime_config_path)
        return {
            "status": "error",
            "command": command,
            "config_path": str(_resolve_config_path(config_path)),
            "runtime_config_path": str(runtime_config_path),
            "exit_code": None,
            "stdout": "",
            "stderr": startup_error,
        }

    job_id = str(uuid.uuid4())
    status_path = _job_status_path(job_id)
    runtime_config["job_status_path"] = str(status_path)
    runtime_config_path = write_runtime_config(runtime_config)
    command = build_command(runtime_config_path)
    status_payload = {
        "job_id": job_id,
        "status": "queued",
        "status_path": str(status_path),
        "config_path": str(_resolve_config_path(config_path)),
        "runtime_config_path": str(runtime_config_path),
        "command": command,
    }
    write_status(status_path, status_payload)

    try:
        process = subprocess.Popen(
            build_runner_command(
                job_id=job_id,
                status_path=status_path,
                runtime_config_path=runtime_config_path,
                timeout_sec=timeout_sec,
            ),
            cwd=str(PROJECT_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        update_status(
            status_path,
            status="error",
            error={"code": "spawn_failed", "message": str(exc)},
        )
        return {
            "status": "error",
            "job_id": job_id,
            "status_path": str(status_path),
            "command": command,
            "config_path": str(_resolve_config_path(config_path)),
            "runtime_config_path": str(runtime_config_path),
            "exit_code": None,
            "stdout": "",
            "stderr": str(exc),
        }

    _record_runner_pid(status_path, process.pid)
    return {
        "status": "queued",
        "job_id": job_id,
        "status_path": str(status_path),
        "command": command,
        "config_path": str(_resolve_config_path(config_path)),
        "runtime_config_path": str(runtime_config_path),
        "pid": process.pid,
        "exit_code": None,
        "stdout": "",
        "stderr": "",
    }


def resume_task_bridge(
    *,
    agent_id: str,
    task: str | None = None,
    config_path: str | Path | None = None,
    use_plan: bool | None = None,
    use_skills: bool | None = None,
    max_steps: int | None = None,
    dry_run: bool = False,
    timeout_sec: int | None = None,
) -> dict[str, Any]:
    if not agent_id.strip():
        raise BridgeInputError("agent_id must not be empty")

    resolved_task = task or get_example_config(config_path).get("agent", {}).get("task")
    if not resolved_task:
        raise BridgeInputError("task must not be empty")

    return run_task_bridge(
        task=resolved_task,
        config_path=config_path,
        use_plan=use_plan,
        use_skills=use_skills,
        resume=True,
        agent_id=agent_id,
        max_steps=max_steps,
        dry_run=dry_run,
        timeout_sec=timeout_sec,
    )


def health_check(config_path: str | Path | None = None) -> dict[str, Any]:
    resolved_config = _resolve_config_path(config_path)
    main_path = _main_entrypoint_path()
    runner_path = _runner_entrypoint_path()
    config_info = _inspect_config_path(resolved_config)
    return {
        "status": (
            "ok"
            if config_info["config_loadable"]
            and main_path.exists()
            and main_path.is_file()
            and runner_path.exists()
            and runner_path.is_file()
            else "error"
        ),
        "python": sys.executable,
        "config_path": str(resolved_config),
        "config_exists": config_info["config_exists"],
        "config_is_file": config_info["config_is_file"],
        "config_loadable": config_info["config_loadable"],
        "main_path": str(main_path),
        "main_exists": main_path.exists(),
        "runner_path": str(runner_path),
        "runner_exists": runner_path.exists(),
        **(
            {"config_error": config_info["config_error"]}
            if config_info["config_error"]
            else {}
        ),
    }


def get_task_status_bridge(*, job_id: str) -> dict[str, Any]:
    return read_status(_job_status_path(job_id))


def cancel_task_bridge(*, job_id: str) -> dict[str, Any]:
    status = get_task_status_bridge(job_id=job_id)
    pid = status.get("pid")
    cancel_pid = status.get("runner_pid") or pid
    status_path = _job_status_path(job_id)

    if status.get("status") == "missing":
        return status

    if cancel_pid is None:
        return _cancel_result(
            status,
            code="missing_pid",
            message=f"No pid recorded for job {job_id}",
        )

    if not _pid_exists(cancel_pid):
        return _cancel_result(
            status,
            code="stale_pid",
            message=f"Process {cancel_pid} is no longer running for job {job_id}",
        )

    if not _pid_matches_job(cancel_pid, status):
        return _cancel_result(
            status,
            code="pid_mismatch",
            message=f"Process {cancel_pid} does not match job {job_id}",
        )

    result_status = status
    if status.get("status") == "queued":
        update_status(status_path, status="cancelled")
        result_status = read_status(status_path)

    try:
        os.kill(cancel_pid, signal.SIGTERM)
    except ProcessLookupError:
        if result_status.get("status") == "cancelled":
            return result_status
        return _cancel_result(
            result_status,
            code="stale_pid",
            message=f"Process {cancel_pid} exited before it could be cancelled for job {job_id}",
        )
    except PermissionError:
        return _cancel_result(
            result_status,
            code="permission_denied",
            message=f"Permission denied when cancelling process {cancel_pid} for job {job_id}",
        )
    return _cancel_result(result_status, signal_value=signal.SIGTERM)


def _resolve_config_path(config_path: str | Path | None) -> Path:
    if config_path is None:
        return _resolve_existing_project_path(
            DEFAULT_CONFIG_CANDIDATES,
            fallback=PROJECT_ROOT / DEFAULT_CONFIG_CANDIDATES[0],
        )

    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def _resolve_example_config_path(config_path: str | Path | None) -> Path:
    if config_path is not None:
        return _resolve_config_path(config_path)

    return _resolve_existing_project_path(
        EXAMPLE_CONFIG_CANDIDATES,
        fallback=PROJECT_ROOT / EXAMPLE_CONFIG_CANDIDATES[0],
    )


def _resolve_existing_project_path(
    candidates: tuple[str, ...], *, fallback: Path
) -> Path:
    for name in candidates:
        candidate = (PROJECT_ROOT / name).resolve()
        if candidate.exists():
            return candidate
    return fallback


def _main_entrypoint_path() -> Path:
    return PROJECT_ROOT / "examples" / "main.py"


def _runner_entrypoint_path() -> Path:
    return PROJECT_ROOT / "mcp_job_runner.py"


def _launch_startup_error() -> str:
    for label, path in (
        ("main entrypoint", _main_entrypoint_path()),
        ("runner entrypoint", _runner_entrypoint_path()),
    ):
        if not path.exists():
            return f"Missing {label}: {path}"
        if not path.is_file():
            return f"Invalid {label}: {path}"
    return ""


def _job_status_path(job_id: str) -> Path:
    if not job_id or not job_id.strip():
        raise BridgeInputError("job_id must not be empty")
    return DEFAULT_TEMP_DIR / "jobs" / f"{job_id}.json"


def _record_runner_pid(status_path: Path, runner_pid: int) -> None:
    current = read_status(status_path)
    if current.get("status") == "missing":
        return

    updates: dict[str, Any] = {}
    if current.get("runner_pid") is None:
        updates["runner_pid"] = runner_pid
    if current.get("status") == "queued" and current.get("pid") is None:
        updates["pid"] = runner_pid

    if updates:
        update_status(status_path, **updates)


def _cancel_result(
    status: dict[str, Any],
    *,
    code: str | None = None,
    message: str | None = None,
    signal_value: int | None = None,
) -> dict[str, Any]:
    result = dict(status)
    if code and message:
        result["error"] = {"code": code, "message": message}
    if signal_value is not None:
        result["signal"] = signal_value
    return result


def _inspect_config_path(path: Path) -> dict[str, Any]:
    info = {
        "config_exists": path.exists(),
        "config_is_file": path.is_file(),
        "config_loadable": False,
        "config_error": "",
    }
    if not info["config_exists"]:
        info["config_error"] = f"Missing config file: {path}"
        return info
    if not info["config_is_file"]:
        info["config_error"] = f"Config path is not a file: {path}"
        return info

    try:
        _load_config(path, resolve_env=False)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        info["config_error"] = str(exc)
        return info

    info["config_loadable"] = True
    return info


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _pid_matches_job(pid: int, status: dict[str, Any]) -> bool:
    completed = subprocess.run(
        ["ps", "-o", "command=", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return False

    command = completed.stdout.strip()
    if not command:
        return False

    try:
        argv = shlex.split(command)
    except ValueError:
        return False

    if str(_runner_entrypoint_path()) not in argv:
        return False

    markers: dict[str, str] = {}
    for index, token in enumerate(argv[:-1]):
        if token in {"--job-id", "--status-path"}:
            markers[token] = argv[index + 1]

    job_id = status.get("job_id")
    if job_id and markers.get("--job-id") == str(job_id):
        return True

    status_path = status.get("status_path")
    if status_path and markers.get("--status-path") == str(status_path):
        return True

    return False


def _load_config(path: Path, *, resolve_env: bool) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if resolve_env:
        return resolve_env_placeholders(config)
    return config


def _truncate_output(output: str | None, limit: int = DEFAULT_OUTPUT_LIMIT) -> str:
    if not output:
        return ""
    if len(output) <= limit:
        return output
    clipped = len(output) - limit
    return f"{output[:limit]}\n...[truncated {clipped} chars]"
