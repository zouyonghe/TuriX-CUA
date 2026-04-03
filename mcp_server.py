from __future__ import annotations

from typing import Any

from mcp_bridge import (
    BridgeInputError,
    cancel_task_bridge,
    get_example_config,
    get_task_status_bridge,
    health_check,
    resume_task_bridge,
    run_task_bridge,
)

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - exercised in local startup checks
    FastMCP = None
    MCP_IMPORT_ERROR = exc
else:
    MCP_IMPORT_ERROR = None


def _error_result(exc: Exception) -> dict[str, Any]:
    return {"status": "error", "error": str(exc)}


def run_task(
    task: str,
    config_path: str | None = None,
    use_plan: bool | None = None,
    use_skills: bool | None = None,
    resume: bool | None = False,
    agent_id: str | None = None,
    max_steps: int | None = None,
    dry_run: bool = False,
    timeout_sec: int | None = None,
) -> dict[str, Any]:
    """Run a new TuriX task through the examples/main.py entrypoint."""
    try:
        return run_task_bridge(
            task=task,
            config_path=config_path,
            use_plan=use_plan,
            use_skills=use_skills,
            resume=resume,
            agent_id=agent_id,
            max_steps=max_steps,
            dry_run=dry_run,
            timeout_sec=timeout_sec,
        )
    except BridgeInputError as exc:
        return _error_result(exc)


def resume_task(
    agent_id: str,
    task: str | None = None,
    config_path: str | None = None,
    use_plan: bool | None = None,
    use_skills: bool | None = None,
    max_steps: int | None = None,
    dry_run: bool = False,
    timeout_sec: int | None = None,
) -> dict[str, Any]:
    """Resume a previous TuriX task using a stable agent_id."""
    try:
        return resume_task_bridge(
            agent_id=agent_id,
            task=task,
            config_path=config_path,
            use_plan=use_plan,
            use_skills=use_skills,
            max_steps=max_steps,
            dry_run=dry_run,
            timeout_sec=timeout_sec,
        )
    except BridgeInputError as exc:
        return _error_result(exc)


def get_task_status(job_id: str) -> dict[str, Any]:
    """Return the persisted status for a tracked TuriX MCP job."""
    try:
        return get_task_status_bridge(job_id=job_id)
    except BridgeInputError as exc:
        return _error_result(exc)


def cancel_task(job_id: str) -> dict[str, Any]:
    """Request cancellation for a tracked TuriX MCP job."""
    try:
        return cancel_task_bridge(job_id=job_id)
    except BridgeInputError as exc:
        return _error_result(exc)


def get_example_config_tool(config_path: str | None = None) -> dict[str, Any]:
    """Return the current example config so Codex can inspect the local setup."""
    try:
        return get_example_config(config_path)
    except (BridgeInputError, FileNotFoundError, OSError, ValueError) as exc:
        return _error_result(exc)


def health_check_tool(config_path: str | None = None) -> dict[str, Any]:
    """Report whether the local repository looks ready to launch TuriX."""
    try:
        return health_check(config_path)
    except (BridgeInputError, FileNotFoundError, OSError, ValueError) as exc:
        return _error_result(exc)


if FastMCP is not None:
    mcp = FastMCP(
        "TuriX",
        instructions=(
            "Run and resume local TuriX desktop-automation tasks. "
            "Prefer dry_run first if the setup is uncertain."
        ),
        json_response=True,
    )
    mcp.tool()(run_task)
    mcp.tool()(resume_task)
    mcp.tool()(get_task_status)
    mcp.tool()(cancel_task)
    mcp.tool(name="get_example_config")(get_example_config_tool)
    mcp.tool(name="health_check")(health_check_tool)
else:  # pragma: no cover - exercised in local startup checks
    mcp = None


def main() -> None:
    """Start the local stdio MCP server."""
    if mcp is None:
        raise RuntimeError(
            "The 'mcp' package is not installed. Install dependencies from "
            "requirements.txt before starting the TuriX MCP server."
        ) from MCP_IMPORT_ERROR
    mcp.run()


if __name__ == "__main__":
    main()
