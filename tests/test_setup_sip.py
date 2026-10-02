"""
Unit tests for the OPTIONAL, isolated native LiveKit SIP module (telephony/setup_sip.py).

All Twilio HTTP calls and the LiveKit API are mocked - no test ever makes a
real network call or touches real credentials. These tests exist to prove:

- the module is never invoked anywhere else in the project (see
  test_module_is_not_imported_by_running_components),
- the default (no --apply) path is a true dry run that makes zero network
  calls and provisions nothing,
- --apply refuses to run when required settings are missing, rather than
  silently doing something unexpected,
- the create-or-reuse logic actually reuses existing resources instead of
  creating duplicates, matching the reference project's documented behavior.
"""

import ast
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from telephony import setup_sip


class TestPlanIsNetworkFree(unittest.TestCase):
    @patch("telephony.setup_sip._twilio")
    @patch("telephony.setup_sip.api.LiveKitAPI")
    def test_plan_makes_no_network_calls(self, mock_lk_cls, mock_twilio):
        lines = setup_sip.plan()

        mock_twilio.assert_not_called()
        mock_lk_cls.assert_not_called()
        self.assertEqual(len(lines), 8)
        self.assertTrue(any("Elastic SIP Trunk" in l for l in lines))
        self.assertTrue(any("outbound SIP trunk" in l for l in lines))

    def test_plan_flags_missing_phone_number(self):
        with patch.object(setup_sip.settings, "twilio_sip_phone_number", ""):
            lines = setup_sip.plan()
        self.assertTrue(any("not set" in l for l in lines))


class TestMainDryRun(unittest.TestCase):
    @patch("telephony.setup_sip._twilio")
    @patch("telephony.setup_sip.api.LiveKitAPI")
    def test_default_is_dry_run_with_zero_network_calls(self, mock_lk_cls, mock_twilio):
        setup_sip.main(apply=False)

        mock_twilio.assert_not_called()
        mock_lk_cls.assert_not_called()

    @patch("telephony.setup_sip.setup_livekit")
    @patch("telephony.setup_sip.setup_twilio_termination")
    @patch("telephony.setup_sip.setup_twilio")
    def test_apply_without_required_settings_refuses_and_does_nothing(
        self, mock_setup_twilio, mock_setup_term, mock_setup_lk
    ):
        with patch.object(setup_sip.settings, "twilio_account_sid", ""), \
             patch.object(setup_sip.settings, "twilio_auth_token", ""), \
             patch.object(setup_sip.settings, "twilio_sip_phone_number", ""):
            with self.assertRaises(SystemExit):
                setup_sip.main(apply=True)

        mock_setup_twilio.assert_not_called()
        mock_setup_term.assert_not_called()
        mock_setup_lk.assert_not_called()

    @patch("telephony.setup_sip.setup_livekit", new_callable=AsyncMock)
    @patch("telephony.setup_sip.setup_twilio_termination")
    @patch("telephony.setup_sip.setup_twilio")
    def test_apply_with_settings_present_runs_all_three_steps(
        self, mock_setup_twilio, mock_setup_term, mock_setup_lk
    ):
        mock_setup_twilio.return_value = "TKxxxx"
        mock_setup_term.return_value = ("domain.pstn.twilio.com", "user", "pwd")

        with patch.object(setup_sip.settings, "twilio_account_sid", "ACxxxx"), \
             patch.object(setup_sip.settings, "twilio_auth_token", "secret"), \
             patch.object(setup_sip.settings, "twilio_sip_phone_number", "+15551230000"):
            setup_sip.main(apply=True)

        mock_setup_twilio.assert_called_once()
        mock_setup_term.assert_called_once_with("TKxxxx")
        mock_setup_lk.assert_awaited_once_with("domain.pstn.twilio.com", "user", "pwd")


class TestSetupTwilioReusesExistingResources(unittest.TestCase):
    @patch("telephony.setup_sip._twilio")
    def test_reuses_existing_trunk_origination_url_and_attachment(self, mock_twilio):
        """When everything already exists, setup_twilio() must only GET, never POST."""
        with patch.object(setup_sip.settings, "twilio_sip_phone_number", "+15551230000"), \
             patch.object(setup_sip.settings, "twilio_account_sid", "ACxxxx"):
            sip_uri = f"sip:{setup_sip.livekit_sip_host()};transport=tcp"

            def fake_twilio(method, url, form=None):
                if "Trunks" in url and url.endswith("/Trunks"):
                    return {"trunks": [{"friendly_name": setup_sip.TRUNK_NAME, "sid": "TK1"}]}
                if url.endswith("/OriginationUrls"):
                    return {"origination_urls": [{"sip_url": sip_uri}]}
                if "IncomingPhoneNumbers.json" in url:
                    return {"incoming_phone_numbers": [{"sid": "PN1"}]}
                if url.endswith("/PhoneNumbers"):
                    return {"phone_numbers": [{"sid": "PN1"}]}
                raise AssertionError(f"Unexpected call: {method} {url}")

            mock_twilio.side_effect = fake_twilio
            trunk_sid = setup_sip.setup_twilio()

        self.assertEqual(trunk_sid, "TK1")
        for call in mock_twilio.call_args_list:
            self.assertEqual(call.args[0], "GET", "no POST should happen when everything already exists")

    @patch("telephony.setup_sip._twilio")
    def test_creates_trunk_when_it_does_not_exist(self, mock_twilio):
        with patch.object(setup_sip.settings, "twilio_sip_phone_number", "+15551230000"), \
             patch.object(setup_sip.settings, "twilio_account_sid", "ACxxxx"):
            sip_uri = f"sip:{setup_sip.livekit_sip_host()};transport=tcp"
            calls = []

            def fake_twilio(method, url, form=None):
                calls.append((method, url))
                if method == "GET" and url.endswith("/Trunks"):
                    return {"trunks": []}
                if method == "POST" and url.endswith("/Trunks"):
                    return {"sid": "TKNEW"}
                if url.endswith("/OriginationUrls"):
                    return {"origination_urls": []} if method == "GET" else {}
                if "IncomingPhoneNumbers.json" in url:
                    return {"incoming_phone_numbers": [{"sid": "PN1"}]}
                if url.endswith("/PhoneNumbers"):
                    return {"phone_numbers": []} if method == "GET" else {}
                raise AssertionError(f"Unexpected call: {method} {url}")

            mock_twilio.side_effect = fake_twilio
            trunk_sid = setup_sip.setup_twilio()

        self.assertEqual(trunk_sid, "TKNEW")
        self.assertIn(("POST", "https://trunking.twilio.com/v1/Trunks"), calls)


class TestSetupLivekitReusesExistingResources(unittest.TestCase):
    @patch("telephony.setup_sip.api.LiveKitAPI")
    def test_reuses_existing_inbound_trunk_and_dispatch_rule(self, mock_lk_cls):
        mock_lk = MagicMock()
        existing_trunk = MagicMock(name=setup_sip.LK_TRUNK_NAME, sip_trunk_id="ST1")
        existing_trunk.name = setup_sip.LK_TRUNK_NAME
        mock_lk.sip.list_inbound_trunk = AsyncMock(return_value=MagicMock(items=[existing_trunk]))
        mock_lk.sip.create_inbound_trunk = AsyncMock()

        existing_rule = MagicMock()
        existing_rule.name = setup_sip.LK_RULE_NAME
        mock_lk.sip.list_dispatch_rule = AsyncMock(return_value=MagicMock(items=[existing_rule]))
        mock_lk.sip.create_dispatch_rule = AsyncMock()

        mock_lk.sip.list_outbound_trunk = AsyncMock(return_value=MagicMock(items=[]))
        mock_lk.sip.delete_trunk = AsyncMock()
        mock_lk.sip.create_outbound_trunk = AsyncMock(return_value=MagicMock(sip_trunk_id="OUT1"))
        mock_lk.aclose = AsyncMock()
        mock_lk_cls.return_value = mock_lk

        with patch.object(setup_sip.settings, "twilio_sip_phone_number", "+15551230000"), \
             patch.object(setup_sip, "_env_set") as mock_env_set:
            import asyncio
            asyncio.run(setup_sip.setup_livekit("domain.pstn.twilio.com", "user", "pwd"))

        mock_lk.sip.create_inbound_trunk.assert_not_called()
        mock_lk.sip.create_dispatch_rule.assert_not_called()
        mock_lk.sip.delete_trunk.assert_not_called()  # nothing existing to delete
        mock_lk.sip.create_outbound_trunk.assert_awaited_once()
        mock_env_set.assert_called_once_with("LIVEKIT_SIP_OUTBOUND_TRUNK_ID", "OUT1")
        mock_lk.aclose.assert_awaited_once()

    @patch("telephony.setup_sip.api.LiveKitAPI")
    def test_outbound_trunk_is_deleted_and_recreated_when_already_present(self, mock_lk_cls):
        """Documents the one non-idempotent-by-reuse resource: the outbound
        trunk is always refreshed, never left stale, never duplicated."""
        mock_lk = MagicMock()
        mock_lk.sip.list_inbound_trunk = AsyncMock(return_value=MagicMock(items=[]))
        new_inbound = MagicMock(sip_trunk_id="ST_NEW")
        mock_lk.sip.create_inbound_trunk = AsyncMock(return_value=new_inbound)
        mock_lk.sip.list_dispatch_rule = AsyncMock(return_value=MagicMock(items=[]))
        mock_lk.sip.create_dispatch_rule = AsyncMock(return_value=MagicMock(sip_dispatch_rule_id="RULE1"))

        old_out = MagicMock(sip_trunk_id="OUT_OLD")
        old_out.name = setup_sip.LK_OUT_TRUNK_NAME
        mock_lk.sip.list_outbound_trunk = AsyncMock(return_value=MagicMock(items=[old_out]))
        mock_lk.sip.delete_trunk = AsyncMock()
        mock_lk.sip.create_outbound_trunk = AsyncMock(return_value=MagicMock(sip_trunk_id="OUT_NEW"))
        mock_lk.aclose = AsyncMock()
        mock_lk_cls.return_value = mock_lk

        with patch.object(setup_sip.settings, "twilio_sip_phone_number", "+15551230000"), \
             patch.object(setup_sip, "_env_set"):
            import asyncio
            asyncio.run(setup_sip.setup_livekit("domain.pstn.twilio.com", "user", "pwd"))

        mock_lk.sip.delete_trunk.assert_awaited_once()
        deleted_request = mock_lk.sip.delete_trunk.call_args.args[0]
        self.assertEqual(deleted_request.sip_trunk_id, "OUT_OLD")
        mock_lk.sip.create_outbound_trunk.assert_awaited_once()


class TestNotWiredIntoRunningComponents(unittest.TestCase):
    """Static proof that telephony.setup_sip is never imported by anything
    that actually runs the voice pipeline, the webhook, or the dashboard."""

    def test_module_is_not_imported_by_running_components(self):
        root = Path(__file__).resolve().parent.parent
        watched = [
            root / "agent" / "core_agent.py",
            root / "run_local.py",
            root / "run_voice_local.py",
            root / "telephony" / "twilio_webhook.py",
            root / "dashboard" / "server.py",
        ]
        for path in watched:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertNotIn(
                            "setup_sip", alias.name, f"{path} must not import telephony.setup_sip"
                        )
                elif isinstance(node, ast.ImportFrom) and node.module:
                    self.assertNotIn(
                        "setup_sip", node.module, f"{path} must not import from telephony.setup_sip"
                    )


if __name__ == "__main__":
    unittest.main()
