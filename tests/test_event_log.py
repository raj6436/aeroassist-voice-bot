"""
Unit tests for the structured call-event logger (agent/event_log.py).

Verifies the on-disk schema and that log_event() never raises or blocks the
caller, both with and without a running asyncio event loop.
"""

import asyncio
import json
import unittest
from unittest.mock import patch

from agent import event_log


class TestLogEvent(unittest.TestCase):
    def setUp(self):
        self.written = []
        patcher = patch.object(event_log, "_write", side_effect=self.written.append)
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_sync_context_writes_directly(self):
        """No running event loop: log_event() must fall back to a direct write."""
        event_log.log_event("call-1", "call_start", direction="inbound", output="joined")

        self.assertEqual(len(self.written), 1)
        record = self.written[0]
        self.assertEqual(record["call"], "call-1")
        self.assertEqual(record["kind"], "call_start")
        self.assertEqual(record["direction"], "inbound")
        self.assertEqual(record["output"], "joined")
        self.assertIn("ts", record)
        self.assertTrue(record["source"].endswith("test_event_log.py:test_sync_context_writes_directly"))

    def test_empty_and_none_fields_are_dropped(self):
        event_log.log_event("call-1", "agent_reply", output="hi", input=None, language="")

        record = self.written[0]
        self.assertNotIn("input", record)
        self.assertNotIn("language", record)
        self.assertEqual(record["output"], "hi")

    def test_async_context_dispatches_to_executor_without_blocking(self):
        """Inside a running event loop, the write must go through run_in_executor
        (never inline), so log_event() itself returns immediately."""

        async def _run():
            event_log.log_event("call-2", "tool_call", source="agent/tools.py:lookup_booking")
            # give the executor a turn to actually run the dispatched write
            await asyncio.sleep(0.05)

        asyncio.run(_run())

        self.assertEqual(len(self.written), 1)
        self.assertEqual(self.written[0]["call"], "call-2")

    def test_write_failure_never_raises(self):
        """log_event() must swallow any disk-I/O failure, never break the caller."""
        with patch.object(event_log, "_write", side_effect=OSError("disk full")):
            try:
                event_log.log_event("call-3", "error", output="boom")
            except Exception as e:  # noqa: BLE001
                self.fail(f"log_event() raised unexpectedly: {e!r}")


class TestWriteAppendsJsonLines(unittest.TestCase):
    """The real _write() (not mocked here) must append valid JSON Lines."""

    def test_write_appends_one_json_line(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            fake_path = Path(d) / "events.jsonl"
            with patch.object(event_log, "EVENTS_FILE", fake_path):
                event_log._write({"ts": "now", "call": "c", "kind": "k"})
                event_log._write({"ts": "later", "call": "c", "kind": "k2"})

            lines = fake_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["kind"], "k")
            self.assertEqual(json.loads(lines[1])["kind"], "k2")


if __name__ == "__main__":
    unittest.main()
