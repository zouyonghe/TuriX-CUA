import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mcp_bridge import (
    BridgeInputError,
    build_runtime_config,
    get_example_config,
    run_task_bridge,
    write_runtime_config,
)


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

    @patch("mcp_bridge.subprocess.run")
    def test_run_task_bridge_dry_run_skips_subprocess(self, mock_run) -> None:
        result = run_task_bridge(task="open Chrome", dry_run=True)

        self.assertEqual(result["status"], "dry_run")
        self.assertIn("examples/main.py", " ".join(result["command"]))
        mock_run.assert_not_called()

    @patch("mcp_bridge.subprocess.run")
    def test_run_task_bridge_executes_subprocess(self, mock_run) -> None:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "ok"
        mock_run.return_value.stderr = ""

        result = run_task_bridge(task="open Chrome", dry_run=False)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["stdout"], "ok")
        mock_run.assert_called_once()


class ExampleConfigTests(unittest.TestCase):
    def test_get_example_config_returns_agent_task(self) -> None:
        config = get_example_config()
        self.assertIn("agent", config)
        self.assertIn("task", config["agent"])

    def test_example_config_preserves_upstream_placeholder_llm_settings(self) -> None:
        config = get_example_config()

        self.assertEqual(config["brain_llm"]["provider"], "turix")
        self.assertEqual(config["brain_llm"]["model_name"], "turix-brain")
        self.assertEqual(config["brain_llm"]["api_key"], "your_api_key_here")
        self.assertEqual(config["brain_llm"]["base_url"], "https://turixapi.io/v1")


class MCPServerModuleTests(unittest.TestCase):
    def test_mcp_server_exposes_tool_functions(self) -> None:
        import mcp_server

        self.assertTrue(callable(mcp_server.run_task))
        self.assertTrue(callable(mcp_server.resume_task))
        self.assertTrue(callable(mcp_server.get_example_config_tool))
        self.assertTrue(callable(mcp_server.health_check_tool))


class OpenCodeConfigTests(unittest.TestCase):
    def test_opencode_config_uses_portable_local_command(self) -> None:
        config_path = Path(__file__).resolve().parent.parent / "opencode.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))

        turix = config["mcp"]["turix"]
        self.assertEqual(turix["type"], "local")
        self.assertEqual(turix["command"], ["python", "mcp_server.py"])
        self.assertTrue(turix["enabled"])


if __name__ == "__main__":
    unittest.main()
