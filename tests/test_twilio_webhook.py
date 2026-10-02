"""
Unit tests for the Twilio Media Streams webhook.

No real LiveKit API call is made: livekit.api.LiveKitAPI is mocked so the
test only verifies our own request-building and TwiML-generation logic.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from telephony.twilio_webhook import AEROASSIST_AGENT_NAME, app


class TestTwilioVoiceWebhook(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch("telephony.twilio_webhook.api.LiveKitAPI")
    def test_inbound_call_returns_stream_twiml(self, mock_livekit_api_cls):
        mock_lkapi = MagicMock()
        mock_lkapi.connector.connect_twilio_call = AsyncMock(
            return_value=MagicMock(connect_url="wss://example.livekit.cloud/mock-stream")
        )
        mock_lkapi.aclose = AsyncMock()
        mock_livekit_api_cls.return_value = mock_lkapi

        response = self.client.post(
            "/twilio/voice",
            data={"CallSid": "CAxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"},
        )

        self.assertEqual(response.status_code, 200)

        # connect_twilio_call was actually called, exactly once
        mock_lkapi.connector.connect_twilio_call.assert_called_once()

        # inbound direction was used
        sent_request = mock_lkapi.connector.connect_twilio_call.call_args[0][0]
        self.assertEqual(
            sent_request.twilio_call_direction,
            sent_request.TwilioCallDirection.TWILIO_CALL_DIRECTION_INBOUND,
        )

        # a room_name was generated
        self.assertTrue(sent_request.room_name)
        self.assertTrue(sent_request.room_name.startswith("twilio-call-"))

        # the agents (explicit dispatch) parameter is present, targeting the
        # existing AeroAssist worker's actual (unmodified) agent_name
        self.assertEqual(len(sent_request.agents), 1)
        self.assertEqual(sent_request.agents[0].agent_name, AEROASSIST_AGENT_NAME)

        # the returned connect_url appears in the TwiML response
        self.assertIn("wss://example.livekit.cloud/mock-stream", response.text)
        self.assertIn("<Connect>", response.text)
        self.assertIn("<Stream", response.text)

        # no real LiveKit API instance/connection was created
        mock_livekit_api_cls.assert_called_once()
        mock_lkapi.aclose.assert_awaited_once()

    @patch("telephony.twilio_webhook.api.LiveKitAPI")
    def test_connector_failure_returns_502(self, mock_livekit_api_cls):
        mock_lkapi = MagicMock()
        mock_lkapi.connector.connect_twilio_call = AsyncMock(side_effect=RuntimeError("boom"))
        mock_lkapi.aclose = AsyncMock()
        mock_livekit_api_cls.return_value = mock_lkapi

        response = self.client.post("/twilio/voice", data={})

        self.assertEqual(response.status_code, 502)
        self.assertIn("<Response>", response.text)


if __name__ == "__main__":
    unittest.main()
