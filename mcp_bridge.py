from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "examples" / "config.json"
DEFAULT_TEMP_DIR = PROJECT_ROOT / ".turix_tmp" / "mcp"
DEFAULT_OUTPUT_LIMIT = 4000


class BridgeInputError(ValueError):
    """Raised when the bridge receives invalid user input."""


def get_example_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = _resolve_config_path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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

    if task is not None:
        if not task.strip():
            raise BridgeInputError("task must not be empty")
        agent_cfg["task"] = task

    if use_plan is not None:
        agent_cfg["use_plan"] = bool(use_plan)

    if use_skills is not None:
        agent_cfg["use_skills"] = bool(use_skills)

    if resume is not None:
        if resume and not agent_id:
            raise BridgeInputError("agent_id is required when resume is true")
        agent_cfg["resume"] = bool(resume)

    if agent_id is not None:
        agent_cfg["agent_id"] = agent_id

    if max_steps is not None:
        agent_cfg["max_steps"] = max_steps

    return runtime_config


def write_runtime_config(config: dict[str, Any], output_dir: Path | None = None) -> Path:
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
        str(PROJECT_ROOT / "examples" / "main.py"),
        "-c",
        str(Path(config_path)),
    ]


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

    base_config = get_example_config(config_path)
    runtime_config = build_runtime_config(
        base_config,
        task=task,
        use_plan=use_plan,
        use_skills=use_skills,
        resume=resume,
        agent_id=agent_id,
        max_steps=max_steps,
    )
    runtime_config_path = write_runtime_config(runtime_config)
    command = build_command(runtime_config_path)

    if dry_run:
        return {
            "status": "dry_run",
            "command": command,
            "config_path": str(_resolve_config_path(config_path)),
            "runtime_config_path": str(runtime_config_path),
            "exit_code": None,
            "stdout": "",
            "stderr": "",
        }

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            cwd=str(PROJECT_ROOT),
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "command": command,
            "config_path": str(_resolve_config_path(config_path)),
            "runtime_config_path": str(runtime_config_path),
            "exit_code": None,
            "stdout": _truncate_output(exc.stdout),
            "stderr": _truncate_output(exc.stderr),
        }
    except OSError as exc:
        return {
            "status": "error",
            "command": command,
            "config_path": str(_resolve_config_path(config_path)),
            "runtime_config_path": str(runtime_config_path),
            "exit_code": None,
            "stdout": "",
            "stderr": str(exc),
        }

    return {
        "status": "ok" if completed.returncode == 0 else "error",
        "command": command,
        "config_path": str(_resolve_config_path(config_path)),
        "runtime_config_path": str(runtime_config_path),
        "exit_code": completed.returncode,
        "stdout": _truncate_output(completed.stdout),
        "stderr": _truncate_output(completed.stderr),
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
    main_path = PROJECT_ROOT / "examples" / "main.py"
    return {
        "status": "ok" if resolved_config.exists() and main_path.exists() else "error",
        "python": sys.executable,
        "config_path": str(resolved_config),
        "config_exists": resolved_config.exists(),
        "main_path": str(main_path),
        "main_exists": main_path.exists(),
    }


def _resolve_config_path(config_path: str | Path | None) -> Path:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def _truncate_output(output: str | None, limit: int = DEFAULT_OUTPUT_LIMIT) -> str:
    if not output:
        return ""
    if len(output) <= limit:
        return output
    clipped = len(output) - limit
    return f"{output[:limit]}\n...[truncated {clipped} chars]"
