import fcntl
import multiprocessing
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from job_status import (
    build_progress_update,
    read_status,
    update_status,
    update_status_if_current,
    write_status,
)


def _update_status_in_subprocess(path: str, queue: multiprocessing.Queue) -> None:
    try:
        queue.put(update_status(path, status="running", pid=4321))
    except Exception as exc:  # pragma: no cover - forwarded to parent for assertions
        queue.put({"error": f"{type(exc).__name__}: {exc}"})


class JobStatusTests(unittest.TestCase):
    def test_build_progress_update_serializes_live_agent_fields(self) -> None:
        class _FakeAction:
            def __init__(self, payload: dict[str, object]) -> None:
                self.payload = payload

            def model_dump(self, exclude_unset: bool = True) -> dict[str, object]:
                del exclude_unset
                return dict(self.payload)

        class _FakeModelOutput:
            def __init__(self, action: list[object]) -> None:
                self.action = action

        class _FakeHistory:
            def __init__(self) -> None:
                self.history = [object(), object()]

        class _FakeAgent:
            next_goal = "Open Google Chrome"
            current_state = {"step_evaluate": "Success"}
            history = _FakeHistory()
            save_temp_file_path = "/tmp/agent-123"

        payload = build_progress_update(
            agent=_FakeAgent(),
            model_output=_FakeModelOutput([_FakeAction({"wait": {}})]),
            step=3,
        )

        self.assertEqual(payload["current_step"], 3)
        self.assertEqual(payload["next_goal"], "Open Google Chrome")
        self.assertEqual(payload["last_actions"], [{"wait": {}}])
        self.assertEqual(payload["last_step_evaluate"], "Success")
        self.assertTrue(payload["wait_this_step"])
        self.assertEqual(payload["history_length"], 2)
        self.assertEqual(payload["memory_path"], "/tmp/agent-123/memory.jsonl")

    def test_write_status_creates_new_job_record(self) -> None:
        payload = {
            "job_id": "job-123",
            "status": "queued",
            "created_at": "2026-04-02T00:00:00Z",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "job-123.json"
            written_path = write_status(path, payload)
            data = read_status(path)

        self.assertEqual(written_path, path)
        self.assertEqual(data, payload)

    def test_update_status_replaces_existing_record_atomically(self) -> None:
        initial = {
            "job_id": "job-123",
            "status": "queued",
            "created_at": "2026-04-02T00:00:00Z",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "job-123.json"
            write_status(path, initial)

            with patch("job_status.os.replace", wraps=os.replace) as mock_replace:
                updated = update_status(path, status="running", pid=4321)

            stored = read_status(path)

        self.assertEqual(updated["status"], "running")
        self.assertEqual(updated["pid"], 4321)
        self.assertEqual(stored, updated)
        mock_replace.assert_called_once()
        temp_path, target_path = mock_replace.call_args.args
        self.assertEqual(Path(temp_path).parent, path.parent)
        self.assertNotEqual(Path(temp_path), path)
        self.assertEqual(Path(target_path), path)

    def test_read_status_returns_structured_error_for_missing_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "job-404.json"
            data = read_status(path)

        self.assertEqual(data["status"], "missing")
        self.assertEqual(data["job_id"], "job-404")
        self.assertEqual(data["path"], str(path))
        self.assertEqual(data["error"]["code"], "job_not_found")
        self.assertIn("job-404.json", data["error"]["message"])

    def test_update_status_preserves_untouched_fields_on_partial_update(self) -> None:
        initial = {
            "job_id": "job-123",
            "status": "queued",
            "created_at": "2026-04-02T00:00:00Z",
            "memory_path": "/tmp/memory.json",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "job-123.json"
            write_status(path, initial)

            updated = update_status(path, status="running")

        self.assertEqual(updated["job_id"], "job-123")
        self.assertEqual(updated["status"], "running")
        self.assertEqual(updated["created_at"], "2026-04-02T00:00:00Z")
        self.assertEqual(updated["memory_path"], "/tmp/memory.json")

    def test_update_status_uses_lock_file_to_serialize_writers(self) -> None:
        initial = {
            "job_id": "job-123",
            "status": "queued",
            "created_at": "2026-04-02T00:00:00Z",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "job-123.json"
            lock_path = path.with_name(f"{path.name}.lock")
            write_status(path, initial)

            ctx = multiprocessing.get_context("spawn")
            queue = ctx.Queue()
            process = ctx.Process(
                target=_update_status_in_subprocess, args=(str(path), queue)
            )

            with lock_path.open("a+", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                process.start()
                process.join(timeout=0.3)
                self.assertIsNone(process.exitcode)
                self.assertEqual(read_status(path)["status"], "queued")
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

            process.join(timeout=3)
            result = queue.get(timeout=1)
            queue.close()
            stored = read_status(path)

        self.assertEqual(process.exitcode, 0)
        self.assertNotIn("error", result)
        self.assertEqual(result["status"], "running")
        self.assertEqual(stored["status"], "running")
        self.assertEqual(stored["pid"], 4321)

    def test_update_status_checks_filesystem_instead_of_payload_error_marker(
        self,
    ) -> None:
        payload = {
            "job_id": "job-123",
            "status": "missing",
            "created_at": "2026-04-02T00:00:00Z",
            "error": {
                "code": "job_not_found",
                "message": "stale payload",
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "job-123.json"
            write_status(path, payload)

            updated = update_status(path, status="running")

        self.assertEqual(updated["status"], "running")
        self.assertEqual(updated["error"]["code"], "job_not_found")

    def test_update_status_if_current_preserves_newer_on_disk_state(self) -> None:
        initial = {
            "job_id": "job-123",
            "status": "succeeded",
            "created_at": "2026-04-02T00:00:00Z",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "job-123.json"
            write_status(path, initial)

            current = update_status_if_current(
                path, expected_status="queued", status="cancelled"
            )

        self.assertEqual(current["status"], "succeeded")

    def test_read_status_rejects_malformed_json_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "job-123.json"
            path.write_text("{not valid json}\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Invalid job status"):
                read_status(path)

    def test_read_status_rejects_non_object_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "job-123.json"
            path.write_text('["not", "an object"]\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Invalid job status"):
                read_status(path)


if __name__ == "__main__":
    unittest.main()
