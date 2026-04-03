import io
import json
import subprocess
import sys
import tempfile
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import patch


def _load_main_module():
    wrapper_path = Path(__file__).resolve().parents[1] / "examples" / "main.py"
    spec = spec_from_file_location("main_for_compat_tests", wrapper_path)
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class MainCompatibilityTests(unittest.TestCase):
    def test_examples_main_executes_cli_help(self) -> None:
        wrapper_path = Path(__file__).resolve().parents[1] / "examples" / "main.py"

        self.assertTrue(
            wrapper_path.exists(), msg=f"Missing compatibility wrapper: {wrapper_path}"
        )

        result = subprocess.run(
            [sys.executable, str(wrapper_path), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Run the TuriX agent.", result.stdout)

    def test_examples_main_queues_mcp_runtime_configs_before_running_agent(
        self,
    ) -> None:
        main_module = _load_main_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            runtime_config_path = (
                project_root / ".turix_tmp" / "mcp" / "turix-mcp-wrapper-test.json"
            )
            runtime_config_path.parent.mkdir(parents=True, exist_ok=True)
            runtime_config_path.write_text(
                json.dumps(
                    {
                        "agent": {
                            "task": "Open Safari and stop.",
                            "use_plan": False,
                            "use_skills": False,
                            "resume": False,
                            "agent_id": None,
                            "max_steps": 8,
                        }
                    }
                ),
                encoding="utf-8",
            )
            queued_result = {
                "status": "queued",
                "job_id": "job-123",
                "status_path": "/tmp/job-123.json",
            }
            stdout = io.StringIO()

            with (
                patch.object(main_module, "project_root", project_root),
                patch.object(
                    main_module,
                    "build_llm",
                    side_effect=AssertionError(
                        "main should queue MCP runtime configs before building LLMs"
                    ),
                ),
                patch.object(
                    main_module, "run_task_bridge", return_value=queued_result
                ) as mock_bridge,
                patch("sys.stdout", stdout),
            ):
                result = main_module.main(str(runtime_config_path))

        self.assertEqual(result, queued_result)
        self.assertEqual(json.loads(stdout.getvalue()), queued_result)
        mock_bridge.assert_called_once_with(
            task="Open Safari and stop.",
            config_path=runtime_config_path,
            use_plan=False,
            use_skills=False,
            resume=False,
            agent_id=None,
            max_steps=8,
            dry_run=False,
            timeout_sec=None,
        )
