from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class JobStatusError(ValueError):
    """Raised when a persisted job status payload is missing required structure."""


def write_status(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(target):
        _atomic_write_json(target, payload)
    return target


def update_status(path: str | Path, **changes: Any) -> dict[str, Any]:
    target = Path(path)

    with _exclusive_lock(target):
        if not target.exists():
            raise FileNotFoundError(f"Job status not found: {target}")

        current = _load_status(target)
        current.update(changes)
        _atomic_write_json(target, current)
        return current


def update_status_if_current(
    path: str | Path,
    *,
    expected_status: str,
    **changes: Any,
) -> dict[str, Any]:
    target = Path(path)

    with _exclusive_lock(target):
        if not target.exists():
            return {
                "status": "missing",
                "job_id": target.stem,
                "path": str(target),
                "error": {
                    "code": "job_not_found",
                    "message": f"Job status not found: {target}",
                },
            }

        current = _load_status(target)
        if current.get("status") != expected_status:
            return current

        current.update(changes)
        _atomic_write_json(target, current)
        return current


def build_progress_update(
    *,
    agent: Any,
    model_output: Any | None,
    step: int,
) -> dict[str, Any]:
    actions = _serialize_actions(getattr(model_output, "action", None))
    current_state = getattr(agent, "current_state", None)
    step_evaluate = None
    if isinstance(current_state, dict):
        step_evaluate = current_state.get("step_evaluate")

    history = getattr(getattr(agent, "history", None), "history", [])
    memory_root = getattr(agent, "save_temp_file_path", None)
    memory_path = str(Path(memory_root) / "memory.jsonl") if memory_root else None

    return {
        "current_step": step,
        "next_goal": getattr(agent, "next_goal", ""),
        "last_actions": actions,
        "last_step_evaluate": step_evaluate,
        "wait_this_step": not actions or _is_wait_action(actions[0]),
        "history_length": len(history) if isinstance(history, list) else 0,
        "memory_path": memory_path,
    }


def read_status(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {
            "status": "missing",
            "job_id": target.stem,
            "path": str(target),
            "error": {
                "code": "job_not_found",
                "message": f"Job status not found: {target}",
            },
        }

    return _load_status(target)


def _load_status(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise JobStatusError(f"Invalid job status in {path}: {exc.msg}") from exc

    if not isinstance(payload, dict):
        raise JobStatusError(f"Invalid job status in {path}: expected JSON object")

    return payload


@contextmanager
def _exclusive_lock(path: Path):
    lock_path = path.with_name(f"{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=path.suffix or ".json",
        prefix=f"{path.stem}.",
        dir=path.parent,
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temp_path = handle.name

    os.replace(temp_path, path)


def _serialize_actions(actions: Any) -> list[dict[str, Any]]:
    if not isinstance(actions, list):
        return []

    serialized: list[dict[str, Any]] = []
    for action in actions:
        if hasattr(action, "model_dump"):
            payload = action.model_dump(exclude_unset=True)
        else:
            payload = action
        if isinstance(payload, dict):
            serialized.append(payload)
    return serialized


def _is_wait_action(action: dict[str, Any]) -> bool:
    return list(action.keys())[:1] == ["wait"]
