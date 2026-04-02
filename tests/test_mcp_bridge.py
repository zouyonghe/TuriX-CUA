import importlib
import json
import os
import os as stdlib_os
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mcp_bridge
from job_status import read_status, update_status, write_status
from mcp_bridge import (
    BridgeInputError,
    build_runtime_config,
    get_example_config,
    health_check,
    run_task_bridge,
    write_runtime_config,
)
from config_env import resolve_env_placeholders


def _create_launchable_project_root(base_dir: Path) -> Path:
    examples_dir = base_dir / "examples"
    examples_dir.mkdir()
    (examples_dir / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (base_dir / "mcp_job_runner.py").write_text("print('runner')\n", encoding="utf-8")
    config_path = base_dir / "config.json"
    config_path.write_text(
        json.dumps({"agent": {"task": "original task"}}), encoding="utf-8"
    )
    return config_path


def _create_runner_job_fixture(
    base_dir: Path, job_id: str = "job-123"
) -> tuple[Path, Path]:
    status_path = base_dir / ".turix_tmp" / "mcp" / "jobs" / f"{job_id}.json"
    runtime_config_path = base_dir / ".turix_tmp" / "mcp" / f"{job_id}-runtime.json"
    runtime_config_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_config_path.write_text(
        json.dumps({"agent": {"task": "runner test"}}), encoding="utf-8"
    )
    write_status(
        status_path,
        {
            "job_id": job_id,
            "status": "queued",
            "status_path": str(status_path),
            "runtime_config_path": str(runtime_config_path),
        },
    )
    return status_path, runtime_config_path


class _FakeRunnerChildProcess:
    def __init__(
        self,
        *,
        pid: int,
        returncode: int,
        wait_side_effect: object | None = None,
    ) -> None:
        self.pid = pid
        self.returncode = returncode
        self.wait_timeout = None
        self.wait_calls = 0
        self.wait_side_effect = wait_side_effect
        self.terminate_calls = 0
        self.kill_calls = 0

    def wait(self, timeout: int | None = None) -> int:
        self.wait_timeout = timeout
        self.wait_calls += 1
        effect = self.wait_side_effect
        if isinstance(effect, list):
            effect = effect.pop(0) if effect else None
        if isinstance(effect, BaseException):
            raise effect
        if callable(effect):
            return effect(timeout)
        if effect is not None:
            return effect
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1

    def kill(self) -> None:
        self.kill_calls += 1


class BuildRuntimeConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base_config = {
            "brain_llm": {"provider": "turix", "model_name": "turix-brain"},
            "actor_llm": {"provider": "turix", "model_name": "turix-actor"},
            "memory_llm": {"provider": "turix", "model_name": "turix-brain"},
            "planner_llm": {"provider": "turix", "model_name": "turix-brain"},
            "agent": {
                "task": "original task",
                "use_plan": True,
                "use_skills": True,
                "resume": False,
                "agent_id": None,
                "max_steps": 100,
            },
        }

    def test_build_runtime_config_overrides_task_and_feature_flags(self) -> None:
        cfg = build_runtime_config(
            self.base_config,
            task="open Safari",
            use_plan=False,
            use_skills=False,
            resume=False,
            agent_id=None,
            max_steps=12,
        )

        self.assertEqual(cfg["agent"]["task"], "open Safari")
        self.assertFalse(cfg["agent"]["use_plan"])
        self.assertFalse(cfg["agent"]["use_skills"])
        self.assertFalse(cfg["agent"]["resume"])
        self.assertIsNone(cfg["agent"]["agent_id"])
        self.assertEqual(cfg["agent"]["max_steps"], 12)

    def test_build_runtime_config_requires_agent_id_for_resume(self) -> None:
        with self.assertRaises(BridgeInputError):
            build_runtime_config(
                self.base_config,
                task="continue work",
                use_plan=None,
                use_skills=None,
                resume=True,
                agent_id=None,
                max_steps=None,
            )

    def test_build_runtime_config_sets_resume_fields(self) -> None:
        cfg = build_runtime_config(
            self.base_config,
            task="continue work",
            use_plan=None,
            use_skills=None,
            resume=True,
            agent_id="agent-123",
            max_steps=None,
        )

        self.assertTrue(cfg["agent"]["resume"])
        self.assertEqual(cfg["agent"]["agent_id"], "agent-123")
        self.assertEqual(cfg["agent"]["task"], "continue work")


class RuntimeConfigFileTests(unittest.TestCase):
    def test_write_runtime_config_persists_json(self) -> None:
        config = {"agent": {"task": "hello"}}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_runtime_config(config, Path(tmpdir))
            data = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(data["agent"]["task"], "hello")
        self.assertEqual(path.suffix, ".json")


class RunTaskBridgeTests(unittest.TestCase):
    def test_run_task_bridge_rejects_missing_task(self) -> None:
        with self.assertRaises(BridgeInputError):
            run_task_bridge(task="")

    def test_run_task_bridge_rejects_whitespace_agent_id_for_resume(self) -> None:
        with self.assertRaises(BridgeInputError):
            run_task_bridge(
                task="continue work", resume=True, agent_id="   ", dry_run=True
            )

    @patch.dict(os.environ, {}, clear=True)
    def test_run_task_bridge_dry_run_preserves_env_placeholders_in_runtime_config(
        self,
    ) -> None:
        config = {
            "brain_llm": {
                "provider": "gpt",
                "model_name": "gpt-5.4",
                "api_key": "$API_KEY",
                "base_url": "$BASE_URL",
            },
            "actor_llm": {
                "provider": "gpt",
                "model_name": "gpt-5.4",
                "api_key": "$API_KEY",
                "base_url": "$BASE_URL",
            },
            "memory_llm": {
                "provider": "gpt",
                "model_name": "gpt-5.4",
                "api_key": "$API_KEY",
                "base_url": "$BASE_URL",
            },
            "planner_llm": {
                "provider": "gpt",
                "model_name": "gpt-5.4",
                "api_key": "$API_KEY",
                "base_url": "$BASE_URL",
            },
            "agent": {
                "task": "original task",
                "use_plan": True,
                "use_skills": True,
                "resume": False,
                "agent_id": None,
                "max_steps": 100,
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            with patch("mcp_bridge.DEFAULT_TEMP_DIR", Path(tmpdir)):
                result = run_task_bridge(
                    task="open Chrome",
                    config_path=config_path,
                    dry_run=True,
                )

            runtime_config = json.loads(
                Path(result["runtime_config_path"]).read_text(encoding="utf-8")
            )

        self.assertEqual(runtime_config["brain_llm"]["api_key"], "$API_KEY")
        self.assertEqual(runtime_config["brain_llm"]["base_url"], "$BASE_URL")

    @patch("mcp_bridge.subprocess.run")
    def test_run_task_bridge_dry_run_skips_subprocess(self, mock_run) -> None:
        result = run_task_bridge(task="open Chrome", dry_run=True)

        self.assertEqual(result["status"], "dry_run")
        self.assertIn("examples/main.py", " ".join(result["command"]))
        mock_run.assert_not_called()

    @patch("mcp_bridge.subprocess.run")
    @patch("mcp_bridge.subprocess.Popen")
    def test_run_task_bridge_returns_job_id_and_status_path_immediately(
        self, mock_popen, mock_run
    ) -> None:
        mock_popen.return_value.pid = 4321
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "ok"
        mock_run.return_value.stderr = ""

        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            config_path = _create_launchable_project_root(project_root)

            with (
                patch("mcp_bridge.PROJECT_ROOT", project_root),
                patch(
                    "mcp_bridge.DEFAULT_TEMP_DIR", project_root / ".turix_tmp" / "mcp"
                ),
            ):
                result = run_task_bridge(
                    task="open Chrome", dry_run=False, config_path=config_path
                )

            self.assertIn("job_id", result)
            self.assertIn("status_path", result)
            status_path = Path(result["status_path"])
            stored = json.loads(status_path.read_text(encoding="utf-8"))

        self.assertEqual(result["status"], "queued")
        self.assertTrue(result["job_id"])
        self.assertEqual(
            status_path.parent, Path(tmpdir) / ".turix_tmp" / "mcp" / "jobs"
        )
        self.assertEqual(status_path.suffix, ".json")
        self.assertEqual(stored["job_id"], result["job_id"])
        self.assertEqual(stored["status"], "queued")

    @patch("mcp_bridge.subprocess.run")
    @patch("mcp_bridge.subprocess.Popen")
    def test_run_task_bridge_writes_job_status_path_into_runtime_config(
        self, mock_popen, mock_run
    ) -> None:
        mock_popen.return_value.pid = 4321
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "ok"
        mock_run.return_value.stderr = ""

        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            config_path = _create_launchable_project_root(project_root)

            with (
                patch("mcp_bridge.PROJECT_ROOT", project_root),
                patch(
                    "mcp_bridge.DEFAULT_TEMP_DIR", project_root / ".turix_tmp" / "mcp"
                ),
            ):
                result = run_task_bridge(
                    task="open Chrome", dry_run=False, config_path=config_path
                )

            runtime_config = json.loads(
                Path(result["runtime_config_path"]).read_text(encoding="utf-8")
            )

        self.assertEqual(runtime_config["job_status_path"], result["status_path"])

    @patch("mcp_bridge.subprocess.run")
    @patch("mcp_bridge.subprocess.Popen")
    def test_run_task_bridge_uses_popen_instead_of_run(
        self, mock_popen, mock_run
    ) -> None:
        mock_popen.return_value.pid = 4321
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "ok"
        mock_run.return_value.stderr = ""

        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            config_path = _create_launchable_project_root(project_root)

            with (
                patch("mcp_bridge.PROJECT_ROOT", project_root),
                patch(
                    "mcp_bridge.DEFAULT_TEMP_DIR", project_root / ".turix_tmp" / "mcp"
                ),
            ):
                run_task_bridge(
                    task="open Chrome", dry_run=False, config_path=config_path
                )

        mock_popen.assert_called_once()
        mock_run.assert_not_called()

    @patch("mcp_bridge.subprocess.run")
    @patch("mcp_bridge.subprocess.Popen")
    def test_run_task_bridge_detaches_runner_stdio_from_caller(
        self, mock_popen, mock_run
    ) -> None:
        mock_popen.return_value.pid = 4321
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "ok"
        mock_run.return_value.stderr = ""

        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            config_path = _create_launchable_project_root(project_root)

            with (
                patch("mcp_bridge.PROJECT_ROOT", project_root),
                patch(
                    "mcp_bridge.DEFAULT_TEMP_DIR", project_root / ".turix_tmp" / "mcp"
                ),
            ):
                run_task_bridge(
                    task="open Chrome", dry_run=False, config_path=config_path
                )

        kwargs = mock_popen.call_args.kwargs
        self.assertTrue(kwargs["start_new_session"])
        self.assertIs(kwargs["stdin"], subprocess.DEVNULL)
        self.assertIs(kwargs["stdout"], subprocess.DEVNULL)
        self.assertIs(kwargs["stderr"], subprocess.DEVNULL)

    @patch("mcp_bridge.subprocess.run")
    @patch("mcp_bridge.subprocess.Popen")
    def test_run_task_bridge_preserves_existing_child_pid_when_runner_updates_first(
        self, mock_popen, mock_run
    ) -> None:
        mock_popen.return_value.pid = 4321
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "ok"
        mock_run.return_value.stderr = ""

        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            config_path = _create_launchable_project_root(project_root)
            status_path = project_root / ".turix_tmp" / "mcp" / "jobs" / "job-123.json"

            def popen_side_effect(*args: object, **kwargs: object) -> object:
                update_status(
                    status_path,
                    status="running",
                    pid=9876,
                    runner_pid=4321,
                )
                return mock_popen.return_value

            mock_popen.side_effect = popen_side_effect

            with (
                patch("mcp_bridge.PROJECT_ROOT", project_root),
                patch(
                    "mcp_bridge.DEFAULT_TEMP_DIR", project_root / ".turix_tmp" / "mcp"
                ),
                patch("mcp_bridge.uuid.uuid4", return_value="job-123"),
            ):
                result = run_task_bridge(
                    task="open Chrome", dry_run=False, config_path=config_path
                )

            stored = read_status(status_path)

        self.assertEqual(result["pid"], 4321)
        self.assertEqual(stored["pid"], 9876)
        self.assertEqual(stored["runner_pid"], 4321)

    @patch("mcp_bridge.subprocess.Popen")
    def test_run_task_bridge_fails_fast_when_runner_entrypoint_missing(
        self, mock_popen
    ) -> None:
        mock_popen.return_value.pid = 4321
        config = {"agent": {"task": "original task"}}

        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            config_path = project_root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            examples_dir = project_root / "examples"
            examples_dir.mkdir()
            (examples_dir / "main.py").write_text("print('ok')\n", encoding="utf-8")

            with (
                patch("mcp_bridge.PROJECT_ROOT", project_root),
                patch(
                    "mcp_bridge.DEFAULT_TEMP_DIR", project_root / ".turix_tmp" / "mcp"
                ),
            ):
                result = run_task_bridge(task="open Chrome", config_path=config_path)

        self.assertEqual(result["status"], "error")
        self.assertIn("runner", result["stderr"].lower())
        self.assertNotIn("job_id", result)
        mock_popen.assert_not_called()

    def test_get_task_status_bridge_returns_saved_json_for_known_job(self) -> None:
        bridge = getattr(mcp_bridge, "get_task_status_bridge", None)
        self.assertIsNotNone(bridge)

        payload = {
            "job_id": "job-123",
            "status": "running",
            "pid": 4321,
            "status_path": "/tmp/job-123.json",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("mcp_bridge.DEFAULT_TEMP_DIR", Path(tmpdir)):
                write_status(Path(tmpdir) / "jobs" / "job-123.json", payload)
                result = bridge(job_id="job-123")

        self.assertEqual(result, payload)

    @patch("mcp_bridge.subprocess.run")
    @patch.object(stdlib_os, "kill")
    def test_cancel_task_bridge_reports_structured_result_for_known_pid(
        self, mock_kill, mock_run
    ) -> None:
        bridge = getattr(mcp_bridge, "cancel_task_bridge", None)
        self.assertIsNotNone(bridge)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("mcp_bridge.DEFAULT_TEMP_DIR", Path(tmpdir)):
                status_path = Path(tmpdir) / "jobs" / "job-123.json"
                write_status(
                    status_path,
                    {
                        "job_id": "job-123",
                        "status": "running",
                        "pid": 4321,
                        "status_path": str(status_path),
                    },
                )
                mock_run.return_value.returncode = 0
                mock_run.return_value.stdout = (
                    f"python {mcp_bridge.PROJECT_ROOT / 'mcp_job_runner.py'} --job-id job-123 "
                    f"--status-path {status_path}\n"
                )
                mock_run.return_value.stderr = ""
                result = bridge(job_id="job-123")
                stored = read_status(status_path)

        self.assertEqual(result["job_id"], "job-123")
        self.assertEqual(result["status"], "running")
        self.assertEqual(result["pid"], 4321)
        self.assertEqual(result["signal"], signal.SIGTERM)
        self.assertEqual(stored["status"], "running")
        self.assertEqual(len(mock_kill.call_args_list), 2)
        self.assertEqual(mock_kill.call_args_list[0].args, (4321, 0))
        self.assertEqual(mock_kill.call_args_list[1].args, (4321, signal.SIGTERM))

    @patch("mcp_bridge.subprocess.run")
    @patch.object(stdlib_os, "kill")
    def test_cancel_task_bridge_marks_queued_job_cancelled_immediately(
        self, mock_kill, mock_run
    ) -> None:
        bridge = getattr(mcp_bridge, "cancel_task_bridge", None)
        self.assertIsNotNone(bridge)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("mcp_bridge.DEFAULT_TEMP_DIR", Path(tmpdir)):
                status_path = Path(tmpdir) / "jobs" / "job-123.json"
                write_status(
                    status_path,
                    {
                        "job_id": "job-123",
                        "status": "queued",
                        "pid": 8765,
                        "runner_pid": 4321,
                        "status_path": str(status_path),
                    },
                )
                mock_run.return_value.returncode = 0
                mock_run.return_value.stdout = (
                    f"python {mcp_bridge.PROJECT_ROOT / 'mcp_job_runner.py'} --job-id job-123 "
                    f"--status-path {status_path}\n"
                )
                mock_run.return_value.stderr = ""
                result = bridge(job_id="job-123")
                stored = read_status(status_path)

        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(stored["status"], "cancelled")
        self.assertEqual(result["signal"], signal.SIGTERM)
        self.assertEqual(mock_kill.call_args_list[0].args, (4321, 0))
        self.assertEqual(mock_kill.call_args_list[1].args, (4321, signal.SIGTERM))

    @patch("mcp_bridge.subprocess.run")
    @patch.object(stdlib_os, "kill")
    def test_cancel_task_bridge_reports_stale_pid_without_signalling(
        self, mock_kill, mock_run
    ) -> None:
        bridge = getattr(mcp_bridge, "cancel_task_bridge", None)
        self.assertIsNotNone(bridge)

        def kill_side_effect(pid: int, sig: int) -> None:
            if sig == 0:
                raise ProcessLookupError("no such process")

        mock_kill.side_effect = kill_side_effect

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("mcp_bridge.DEFAULT_TEMP_DIR", Path(tmpdir)):
                write_status(
                    Path(tmpdir) / "jobs" / "job-123.json",
                    {"job_id": "job-123", "status": "running", "pid": 4321},
                )
                result = bridge(job_id="job-123")

        self.assertEqual(result["job_id"], "job-123")
        self.assertEqual(result["status"], "running")
        self.assertEqual(result["pid"], 4321)
        self.assertEqual(result["error"]["code"], "stale_pid")
        mock_run.assert_not_called()
        self.assertEqual(mock_kill.call_args_list[0].args, (4321, 0))
        self.assertEqual(len(mock_kill.call_args_list), 1)

    @patch("mcp_bridge.subprocess.run")
    @patch.object(stdlib_os, "kill")
    def test_cancel_task_bridge_rejects_pid_for_unrelated_process(
        self, mock_kill, mock_run
    ) -> None:
        bridge = getattr(mcp_bridge, "cancel_task_bridge", None)
        self.assertIsNotNone(bridge)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("mcp_bridge.DEFAULT_TEMP_DIR", Path(tmpdir)):
                write_status(
                    Path(tmpdir) / "jobs" / "job-123.json",
                    {"job_id": "job-123", "status": "running", "pid": 4321},
                )
                mock_run.return_value.returncode = 0
                mock_run.return_value.stdout = (
                    "python /tmp/unrelated.py --job-id other-job\n"
                )
                mock_run.return_value.stderr = ""
                result = bridge(job_id="job-123")

        self.assertEqual(result["job_id"], "job-123")
        self.assertEqual(result["status"], "running")
        self.assertEqual(result["pid"], 4321)
        self.assertEqual(result["error"]["code"], "pid_mismatch")
        self.assertEqual(len(mock_kill.call_args_list), 1)
        self.assertEqual(mock_kill.call_args_list[0].args, (4321, 0))
        mock_run.assert_called_once()

    @patch("mcp_bridge.subprocess.run")
    @patch.object(stdlib_os, "kill")
    def test_cancel_task_bridge_requires_exact_job_marker_match(
        self, mock_kill, mock_run
    ) -> None:
        bridge = getattr(mcp_bridge, "cancel_task_bridge", None)
        self.assertIsNotNone(bridge)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("mcp_bridge.DEFAULT_TEMP_DIR", Path(tmpdir)):
                status_path = Path(tmpdir) / "jobs" / "job-123.json"
                write_status(
                    status_path,
                    {
                        "job_id": "job-123",
                        "status": "running",
                        "pid": 4321,
                        "runner_pid": 4321,
                        "status_path": str(status_path),
                    },
                )
                mock_run.return_value.returncode = 0
                mock_run.return_value.stdout = (
                    f"python {mcp_bridge.PROJECT_ROOT / 'mcp_job_runner.py'} --job-id job-1234 "
                    f"--status-path {Path(tmpdir) / 'jobs' / 'job-1234.json'}\n"
                )
                mock_run.return_value.stderr = ""
                result = bridge(job_id="job-123")

        self.assertEqual(result["error"]["code"], "pid_mismatch")
        self.assertEqual(len(mock_kill.call_args_list), 1)
        self.assertEqual(mock_kill.call_args_list[0].args, (4321, 0))

    @patch("mcp_bridge.subprocess.run")
    @patch.object(stdlib_os, "kill")
    def test_cancel_task_bridge_returns_structured_result_when_sigterm_races_with_exit(
        self, mock_kill, mock_run
    ) -> None:
        bridge = getattr(mcp_bridge, "cancel_task_bridge", None)
        self.assertIsNotNone(bridge)

        def kill_side_effect(pid: int, sig: int) -> None:
            if sig == 0:
                return None
            raise ProcessLookupError("no such process")

        mock_kill.side_effect = kill_side_effect

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("mcp_bridge.DEFAULT_TEMP_DIR", Path(tmpdir)):
                status_path = Path(tmpdir) / "jobs" / "job-123.json"
                write_status(
                    status_path,
                    {
                        "job_id": "job-123",
                        "status": "running",
                        "pid": 8765,
                        "runner_pid": 4321,
                        "status_path": str(status_path),
                    },
                )
                mock_run.return_value.returncode = 0
                mock_run.return_value.stdout = (
                    f"python {mcp_bridge.PROJECT_ROOT / 'mcp_job_runner.py'} --job-id job-123 "
                    f"--status-path {status_path}\n"
                )
                mock_run.return_value.stderr = ""
                result = bridge(job_id="job-123")

        self.assertEqual(result["error"]["code"], "stale_pid")
        self.assertEqual(mock_kill.call_args_list[0].args, (4321, 0))
        self.assertEqual(mock_kill.call_args_list[1].args, (4321, signal.SIGTERM))


class MCPJobRunnerTests(unittest.TestCase):
    def test_runner_marks_job_running_and_maps_zero_exit_to_succeeded(self) -> None:
        runner = importlib.import_module("mcp_job_runner")

        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            _create_launchable_project_root(project_root)
            status_path, runtime_config_path = _create_runner_job_fixture(project_root)
            child = _FakeRunnerChildProcess(pid=4321, returncode=0)
            updates = []
            real_update_status = runner.update_status

            def tracking_update_status(
                path: Path, **changes: object
            ) -> dict[str, object]:
                updates.append(dict(changes))
                return real_update_status(path, **changes)

            with (
                patch("mcp_bridge.PROJECT_ROOT", project_root),
                patch.object(
                    runner.subprocess, "Popen", return_value=child
                ) as mock_popen,
                patch.object(
                    runner, "update_status", side_effect=tracking_update_status
                ),
            ):
                result = runner.run_job(
                    job_id="job-123",
                    status_path=status_path,
                    runtime_config_path=runtime_config_path,
                )

            stored = read_status(status_path)
            log_exists = Path(stored["log_path"]).exists()

        self.assertEqual(updates[0]["status"], "running")
        self.assertEqual(updates[0]["pid"], 4321)
        self.assertTrue(updates[0]["log_path"].endswith("job-123.log"))
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(stored["status"], "succeeded")
        self.assertEqual(stored["pid"], 4321)
        self.assertTrue(log_exists)
        self.assertIn("examples/main.py", " ".join(mock_popen.call_args.args[0]))
        self.assertTrue(mock_popen.call_args.kwargs["start_new_session"])

    def test_runner_maps_non_zero_exit_to_failed(self) -> None:
        runner = importlib.import_module("mcp_job_runner")

        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            _create_launchable_project_root(project_root)
            status_path, runtime_config_path = _create_runner_job_fixture(project_root)
            child = _FakeRunnerChildProcess(pid=4321, returncode=7)

            with (
                patch("mcp_bridge.PROJECT_ROOT", project_root),
                patch.object(runner.subprocess, "Popen", return_value=child),
            ):
                result = runner.run_job(
                    job_id="job-123",
                    status_path=status_path,
                    runtime_config_path=runtime_config_path,
                )

            stored = read_status(status_path)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["exit_code"], 7)
        self.assertEqual(stored["status"], "failed")
        self.assertEqual(stored["exit_code"], 7)

    def test_runner_marks_job_cancelled_on_keyboard_interrupt(self) -> None:
        runner = importlib.import_module("mcp_job_runner")

        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            _create_launchable_project_root(project_root)
            status_path, runtime_config_path = _create_runner_job_fixture(project_root)
            child = _FakeRunnerChildProcess(
                pid=4321,
                returncode=0,
                wait_side_effect=[KeyboardInterrupt(), 130],
            )

            with (
                patch("mcp_bridge.PROJECT_ROOT", project_root),
                patch.object(runner.subprocess, "Popen", return_value=child),
            ):
                result = runner.run_job(
                    job_id="job-123",
                    status_path=status_path,
                    runtime_config_path=runtime_config_path,
                )

            stored = read_status(status_path)

        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(stored["status"], "cancelled")
        self.assertEqual(stored["exit_code"], 130)
        self.assertEqual(child.terminate_calls, 1)

    def test_runner_registers_sigterm_handler_and_marks_job_cancelled(self) -> None:
        runner = importlib.import_module("mcp_job_runner")
        handlers: dict[int, object] = {}

        def fake_signal(sig: int, handler: object) -> object:
            handlers[sig] = handler
            return None

        def trigger_sigterm(timeout: int | None = None) -> int:
            del timeout
            handler = handlers[signal.SIGTERM]
            return handler(signal.SIGTERM, None)

        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            _create_launchable_project_root(project_root)
            status_path, runtime_config_path = _create_runner_job_fixture(project_root)
            child = _FakeRunnerChildProcess(
                pid=4321,
                returncode=0,
                wait_side_effect=[trigger_sigterm, 143],
            )

            with (
                patch("mcp_bridge.PROJECT_ROOT", project_root),
                patch.object(runner.subprocess, "Popen", return_value=child),
                patch("signal.signal", side_effect=fake_signal),
            ):
                result = runner.run_job(
                    job_id="job-123",
                    status_path=status_path,
                    runtime_config_path=runtime_config_path,
                )

            stored = read_status(status_path)

        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(stored["status"], "cancelled")
        self.assertEqual(stored["signal"], signal.SIGTERM)
        self.assertEqual(child.terminate_calls, 1)

    def test_runner_uses_process_group_signals_when_available(self) -> None:
        runner = importlib.import_module("mcp_job_runner")

        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            _create_launchable_project_root(project_root)
            status_path, runtime_config_path = _create_runner_job_fixture(project_root)
            child = _FakeRunnerChildProcess(
                pid=4321,
                returncode=0,
                wait_side_effect=[KeyboardInterrupt(), 143],
            )

            with (
                patch("mcp_bridge.PROJECT_ROOT", project_root),
                patch.object(runner.subprocess, "Popen", return_value=child),
                patch.object(runner.os, "getpgid", return_value=4321),
                patch.object(runner.os, "killpg") as mock_killpg,
            ):
                result = runner.run_job(
                    job_id="job-123",
                    status_path=status_path,
                    runtime_config_path=runtime_config_path,
                )

        self.assertEqual(result["status"], "cancelled")
        mock_killpg.assert_called_once_with(4321, signal.SIGTERM)
        self.assertEqual(child.terminate_calls, 0)

    def test_runner_marks_job_failed_on_timeout(self) -> None:
        runner = importlib.import_module("mcp_job_runner")

        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            _create_launchable_project_root(project_root)
            status_path, runtime_config_path = _create_runner_job_fixture(project_root)
            child = _FakeRunnerChildProcess(
                pid=4321,
                returncode=0,
                wait_side_effect=[
                    subprocess.TimeoutExpired(cmd=["python"], timeout=3),
                    9,
                ],
            )

            with (
                patch("mcp_bridge.PROJECT_ROOT", project_root),
                patch.object(runner.subprocess, "Popen", return_value=child),
            ):
                result = runner.run_job(
                    job_id="job-123",
                    status_path=status_path,
                    runtime_config_path=runtime_config_path,
                    timeout_sec=3,
                )

            stored = read_status(status_path)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["code"], "timeout")
        self.assertEqual(stored["exit_code"], 9)
        self.assertEqual(child.kill_calls, 1)

    def test_runner_marks_job_failed_on_spawn_failure(self) -> None:
        runner = importlib.import_module("mcp_job_runner")

        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            _create_launchable_project_root(project_root)
            status_path, runtime_config_path = _create_runner_job_fixture(project_root)

            with (
                patch("mcp_bridge.PROJECT_ROOT", project_root),
                patch.object(runner.subprocess, "Popen", side_effect=OSError("boom")),
            ):
                result = runner.run_job(
                    job_id="job-123",
                    status_path=status_path,
                    runtime_config_path=runtime_config_path,
                )

            stored = read_status(status_path)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["code"], "spawn_failed")
        self.assertEqual(stored["status"], "failed")

    def test_runner_cleans_up_child_if_running_status_write_fails(self) -> None:
        runner = importlib.import_module("mcp_job_runner")

        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            _create_launchable_project_root(project_root)
            status_path, runtime_config_path = _create_runner_job_fixture(project_root)
            child = _FakeRunnerChildProcess(pid=4321, returncode=143)
            real_update_status = runner.update_status
            call_count = 0

            def flaky_update_status(path: Path, **changes: object) -> dict[str, object]:
                nonlocal call_count
                call_count += 1
                if call_count == 1 and changes.get("status") == "running":
                    raise RuntimeError("status write failed")
                return real_update_status(path, **changes)

            with (
                patch("mcp_bridge.PROJECT_ROOT", project_root),
                patch.object(runner.subprocess, "Popen", return_value=child),
                patch.object(runner, "update_status", side_effect=flaky_update_status),
            ):
                with self.assertRaisesRegex(RuntimeError, "status write failed"):
                    runner.run_job(
                        job_id="job-123",
                        status_path=status_path,
                        runtime_config_path=runtime_config_path,
                    )

            stored = read_status(status_path)

        self.assertEqual(child.terminate_calls, 1)
        self.assertEqual(stored["status"], "failed")
        self.assertEqual(stored["exit_code"], 143)


class HealthCheckTests(unittest.TestCase):
    def test_health_check_reports_runner_entrypoint_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            examples_dir = project_root / "examples"
            examples_dir.mkdir()
            (examples_dir / "main.py").write_text("print('ok')\n", encoding="utf-8")
            config_path = project_root / "config.json"
            config_path.write_text(
                json.dumps({"agent": {"task": "demo"}}), encoding="utf-8"
            )

            with patch("mcp_bridge.PROJECT_ROOT", project_root):
                result = health_check(config_path)

        self.assertEqual(result["status"], "error")
        self.assertIn("runner_path", result)
        self.assertEqual(result["runner_path"], str(project_root / "mcp_job_runner.py"))
        self.assertFalse(result["runner_exists"])

    def test_health_check_rejects_directory_config_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            _create_launchable_project_root(project_root)
            config_dir = project_root / "config-dir"
            config_dir.mkdir()

            with patch("mcp_bridge.PROJECT_ROOT", project_root):
                result = health_check(config_dir)

        self.assertEqual(result["status"], "error")
        self.assertTrue(result["config_exists"])
        self.assertFalse(result["config_is_file"])
        self.assertFalse(result["config_loadable"])

    def test_health_check_rejects_malformed_config_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            config_path = _create_launchable_project_root(project_root)
            config_path.write_text("{not valid json}\n", encoding="utf-8")

            with patch("mcp_bridge.PROJECT_ROOT", project_root):
                result = health_check(config_path)

        self.assertEqual(result["status"], "error")
        self.assertTrue(result["config_is_file"])
        self.assertFalse(result["config_loadable"])
        self.assertIn("config_error", result)


class ExampleConfigTests(unittest.TestCase):
    def test_get_example_config_returns_agent_task(self) -> None:
        config = get_example_config()
        self.assertIn("agent", config)
        self.assertIn("task", config["agent"])

    def test_get_example_config_returns_brain_llm_fields(self) -> None:
        config = get_example_config()

        self.assertIn("brain_llm", config)
        self.assertIn("provider", config["brain_llm"])
        self.assertIn("model_name", config["brain_llm"])
        self.assertIn("api_key", config["brain_llm"])
        self.assertIn("base_url", config["brain_llm"])

    @patch.dict(
        os.environ,
        {"API_KEY": "sk-test", "BASE_URL": "https://example.test/v1"},
        clear=True,
    )
    def test_get_example_config_expands_env_placeholders_from_custom_config(
        self,
    ) -> None:
        config = {
            "brain_llm": {
                "provider": "gpt",
                "model_name": "gpt-5.4",
                "api_key": "$API_KEY",
                "base_url": "$BASE_URL",
            },
            "actor_llm": {
                "provider": "gpt",
                "model_name": "gpt-5.4",
                "api_key": "$API_KEY",
                "base_url": "$BASE_URL",
            },
            "memory_llm": {
                "provider": "gpt",
                "model_name": "gpt-5.4",
                "api_key": "$API_KEY",
                "base_url": "$BASE_URL",
            },
            "planner_llm": {
                "provider": "gpt",
                "model_name": "gpt-5.4",
                "api_key": "$API_KEY",
                "base_url": "$BASE_URL",
            },
            "agent": {"task": "demo"},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            resolved = get_example_config(config_path)

        self.assertEqual(resolved["brain_llm"]["api_key"], "sk-test")
        self.assertEqual(resolved["brain_llm"]["base_url"], "https://example.test/v1")


class ExampleTemplateTests(unittest.TestCase):
    def test_examples_config_keeps_turix_template_defaults(self) -> None:
        config_path = (
            Path(__file__).resolve().parent.parent / "examples" / "config.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(config["brain_llm"]["provider"], "gpt")
        self.assertEqual(config["brain_llm"]["model_name"], "gpt-5.4")
        self.assertTrue(config["brain_llm"]["api_key"])
        self.assertTrue(config["brain_llm"]["base_url"])

        self.assertEqual(config["actor_llm"]["provider"], "gpt")
        self.assertEqual(config["actor_llm"]["model_name"], "gpt-5.4")


class MCPServerModuleTests(unittest.TestCase):
    def test_mcp_server_exposes_tool_functions(self) -> None:
        import mcp_server

        self.assertTrue(callable(mcp_server.run_task))
        self.assertTrue(callable(mcp_server.resume_task))
        self.assertTrue(callable(mcp_server.get_task_status))
        self.assertTrue(callable(mcp_server.cancel_task))
        self.assertTrue(callable(mcp_server.get_example_config_tool))
        self.assertTrue(callable(mcp_server.health_check_tool))

    def test_mcp_server_get_task_status_reflects_job_lifecycle_updates(self) -> None:
        import mcp_server

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("mcp_bridge.DEFAULT_TEMP_DIR", Path(tmpdir)):
                status_path = Path(tmpdir) / "jobs" / "job-123.json"
                write_status(
                    status_path,
                    {
                        "job_id": "job-123",
                        "status": "queued",
                        "status_path": str(status_path),
                    },
                )
                self.assertEqual(
                    mcp_server.get_task_status(job_id="job-123")["status"], "queued"
                )

                update_status(status_path, status="running", current_step=1)
                self.assertEqual(
                    mcp_server.get_task_status(job_id="job-123")["status"], "running"
                )

                update_status(status_path, status="succeeded", exit_code=0)
                self.assertEqual(
                    mcp_server.get_task_status(job_id="job-123")["status"],
                    "succeeded",
                )


class OpenCodeConfigTests(unittest.TestCase):
    def test_opencode_config_uses_portable_local_command(self) -> None:
        config_path = Path(__file__).resolve().parent.parent / "opencode.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))

        turix = config["mcp"]["turix"]
        self.assertEqual(turix["type"], "local")
        self.assertTrue(turix["command"][0].endswith("python"))
        self.assertEqual(turix["command"][1], "mcp_server.py")
        self.assertTrue(turix["enabled"])


class ConfigEnvPlaceholderTests(unittest.TestCase):
    @patch.dict(
        os.environ,
        {"API_KEY": "sk-test", "BASE_URL": "https://example.test/v1"},
        clear=True,
    )
    def test_resolve_env_placeholders_expands_nested_values(self) -> None:
        config = {
            "brain_llm": {
                "api_key": "$API_KEY",
                "base_url": "${BASE_URL}",
            },
            "agent": {
                "task": "demo",
                "tags": ["$API_KEY", "literal"],
            },
        }

        resolved = resolve_env_placeholders(config)

        self.assertEqual(resolved["brain_llm"]["api_key"], "sk-test")
        self.assertEqual(resolved["brain_llm"]["base_url"], "https://example.test/v1")
        self.assertEqual(resolved["agent"]["tags"][0], "sk-test")
        self.assertEqual(resolved["agent"]["tags"][1], "literal")

    @patch.dict(os.environ, {}, clear=True)
    def test_resolve_env_placeholders_maps_missing_values_to_none(self) -> None:
        config = {"brain_llm": {"api_key": "$MISSING_API_KEY"}}

        resolved = resolve_env_placeholders(config)

        self.assertIsNone(resolved["brain_llm"]["api_key"])


if __name__ == "__main__":
    unittest.main()
