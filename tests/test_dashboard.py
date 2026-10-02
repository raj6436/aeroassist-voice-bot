"""
Unit tests for the operational dashboard (dashboard/server.py).

Everything that could touch a network (LiveKit Cloud, Deepgram, Gemini,
Cartesia) is mocked. These tests only verify our own routing/response-shaping
logic, and that the dashboard only reads local files (events.jsonl,
aeroassist.log) rather than writing to or otherwise affecting the voice
pipeline. No outbound-calling endpoints exist yet by design.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

import dashboard.server as dash


class TestDashboardRoutes(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(dash.app)

    def test_index_serves_html(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("AeroAssist Dashboard", response.text)

    @patch("dashboard.server._http")
    @patch("dashboard.server.api.LiveKitAPI")
    def test_status_reports_services_and_telephony_mode(self, mock_livekit_api_cls, mock_http):
        mock_http.return_value = (True, "{}")
        mock_lk = MagicMock()
        mock_lk.room.list_rooms = AsyncMock(return_value=MagicMock())
        mock_lk.aclose = AsyncMock()
        mock_livekit_api_cls.return_value = mock_lk

        response = self.client.get("/api/status")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("Deepgram (STT)", body["services"])
        self.assertIn("Gemini (LLM)", body["services"])
        self.assertIn("Cartesia (TTS)", body["services"])
        self.assertTrue(body["services"]["LiveKit Cloud"]["ok"])
        # no outbound-calling / SIP status is reported - that feature is on hold
        self.assertEqual(body["telephony"]["outbound_calling"], "not enabled")
        self.assertIn("Connector", body["telephony"]["mode"])
        mock_lk.aclose.assert_awaited_once()

    @patch("dashboard.server.api.LiveKitAPI")
    def test_status_marks_livekit_unreachable_on_failure(self, mock_livekit_api_cls):
        mock_lk = MagicMock()
        mock_lk.room.list_rooms = AsyncMock(side_effect=RuntimeError("unreachable"))
        mock_lk.aclose = AsyncMock()
        mock_livekit_api_cls.return_value = mock_lk

        response = self.client.get("/api/status")

        self.assertFalse(response.json()["services"]["LiveKit Cloud"]["ok"])

    @patch("dashboard.server.api.LiveKitAPI")
    def test_rooms_lists_participants(self, mock_livekit_api_cls):
        mock_lk = MagicMock()
        room = MagicMock(name="twilio-call-abcd1234", creation_time=123)
        room.name = "twilio-call-abcd1234"
        mock_lk.room.list_rooms = AsyncMock(return_value=MagicMock(rooms=[room]))
        participant = MagicMock(identity="twctp_xyz", kind=0)
        mock_lk.room.list_participants = AsyncMock(return_value=MagicMock(participants=[participant]))
        mock_lk.aclose = AsyncMock()
        mock_livekit_api_cls.return_value = mock_lk

        response = self.client.get("/api/rooms")

        self.assertEqual(response.status_code, 200)
        rooms = response.json()
        self.assertEqual(len(rooms), 1)
        self.assertEqual(rooms[0]["name"], "twilio-call-abcd1234")

    def test_events_reads_and_filters_by_call(self):
        with tempfile.TemporaryDirectory() as d:
            events_file = Path(d) / "events.jsonl"
            events_file.write_text(
                json.dumps({"ts": "t1", "call": "call-a", "kind": "call_start"}) + "\n"
                + json.dumps({"ts": "t2", "call": "call-b", "kind": "call_start"}) + "\n",
                encoding="utf-8",
            )
            with patch.object(dash, "EVENTS_FILE", events_file):
                response = self.client.get("/api/events?call=call-a")

        self.assertEqual(response.status_code, 200)
        events = response.json()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["call"], "call-a")

    def test_events_empty_when_no_file(self):
        with patch.object(dash, "EVENTS_FILE", Path("/nonexistent/events.jsonl")):
            response = self.client.get("/api/events")
        self.assertEqual(response.json(), [])

    def test_logs_reports_placeholder_when_no_log_file(self):
        with patch.object(dash, "LOG_FILE", Path("/nonexistent/aeroassist.log")):
            response = self.client.get("/api/logs")
        self.assertIn("no log file yet", response.json()["lines"][0])

    def test_logs_strips_debug_lines_and_ansi(self):
        with tempfile.TemporaryDirectory() as d:
            log_file = Path(d) / "aeroassist.log"
            log_file.write_text(
                "DEBUG:asyncio:Using selector\n"
                "\x1b[32mINFO\x1b[0m aeroassist - Connecting to room x\n",
                encoding="utf-8",
            )
            with patch.object(dash, "LOG_FILE", log_file):
                response = self.client.get("/api/logs")

        lines = response.json()["lines"]
        self.assertEqual(len(lines), 1)
        self.assertNotIn("\x1b", lines[0])

    @patch("dashboard.server.settings")
    def test_token_issues_jwt_for_browser_test_client(self, mock_settings):
        mock_settings.livekit_url = "wss://example.livekit.cloud"
        mock_settings.livekit_api_key = "APIkey123"
        mock_settings.livekit_api_secret = "secret123456789012345678901234"

        response = self.client.get("/api/token")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["url"], "wss://example.livekit.cloud")
        self.assertTrue(body["room"].startswith("web-"))
        self.assertTrue(body["token"])

    def test_no_outbound_dialing_routes_exist(self):
        """Phase 3 (outbound calling) is explicitly on hold - these routes must not exist."""
        for path in ("/api/dial", "/api/check-number", "/api/verify-number", "/api/calls"):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 404, f"{path} should not be implemented yet")


if __name__ == "__main__":
    unittest.main()
