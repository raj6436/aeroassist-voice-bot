"""
OPTIONAL, isolated native LiveKit SIP telephony path for AeroAssist.

    Inbound : Caller -> Twilio number -> Twilio Elastic SIP Trunk -> LiveKit SIP -> room -> agent
    Outbound: (not wired up anywhere yet) LiveKit outbound trunk -> Twilio termination domain -> PSTN -> callee

This is adapted from aeroassist-voice-bot-main/telephony/setup_sip.py, but:

- Uses a SEPARATE setting, ``TWILIO_SIP_PHONE_NUMBER``, never the number the
  existing Twilio Media Streams -> LiveKit Connector webhook
  (telephony/twilio_webhook.py) is configured with. Attaching a number to a
  SIP trunk replaces its Voice Webhook routing on Twilio's side - reusing the
  Connector's number here would silently break the existing inbound flow.
- Is never imported by agent/core_agent.py, run_local.py,
  telephony/twilio_webhook.py, or dashboard/server.py. Nothing in the running
  system invokes this module; it only runs if someone explicitly executes it.
- Defaults to a DRY RUN: it only prints the exact resources it would create
  or reuse (no network calls at all) unless invoked with --apply.
- Re-running with --apply is safe: every Twilio/LiveKit resource is looked up
  by name/value first and reused if it already exists - nothing here creates
  duplicates (the one exception, noted inline, is the LiveKit outbound trunk,
  which is deleted and recreated each --apply run so its credentials/address
  stay in sync - that is churn, not duplication).

Run:
    python -m telephony.setup_sip            # dry run - prints the plan only
    python -m telephony.setup_sip --apply     # actually provisions resources
"""

import asyncio
import base64
import json
import os
import re
import secrets
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from livekit import api

from config.settings import settings

TRUNK_NAME = "aeroassist-livekit"
LK_TRUNK_NAME = "aeroassist-twilio-inbound"
LK_RULE_NAME = "aeroassist-call-rooms"
ROOM_PREFIX = "call-"
LK_OUT_TRUNK_NAME = "aeroassist-twilio-outbound"
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def livekit_sip_host() -> str:
    host = settings.livekit_url.replace("wss://", "").replace("https://", "").rstrip("/")
    return host.replace(".livekit.cloud", ".sip.livekit.cloud")


def plan() -> list[str]:
    """Exactly what --apply would create or reuse. Makes no network calls."""
    number = settings.twilio_sip_phone_number or "<TWILIO_SIP_PHONE_NUMBER not set>"
    return [
        f"Twilio Elastic SIP Trunk '{TRUNK_NAME}' (reused if it already exists by this name)",
        f"Twilio Origination URL on that trunk -> sip:{livekit_sip_host()};transport=tcp",
        f"Attach Twilio number {number} to that trunk (reused if already attached)",
        f"Twilio termination domain on that trunk (reused if already set)",
        f"Twilio SIP Credential List 'aeroassist-livekit' for outbound auth (reused if password is already known)",
        f"LiveKit inbound SIP trunk '{LK_TRUNK_NAME}' for number {number} (reused if it already exists)",
        f"LiveKit SIP dispatch rule '{LK_RULE_NAME}' (rooms named '{ROOM_PREFIX}<caller>', reused if it already exists)",
        f"LiveKit outbound SIP trunk '{LK_OUT_TRUNK_NAME}' (recreated each --apply run to keep credentials in sync)",
    ]


def _twilio(method: str, url: str, form: dict | None = None) -> dict:
    token = base64.b64encode(
        f"{settings.twilio_account_sid}:{settings.twilio_auth_token}".encode()
    ).decode()
    data = urllib.parse.urlencode(form).encode() if form is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Authorization": f"Basic {token}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = r.read().decode()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Twilio {method} {url} -> {e.code}: {e.read().decode()[:300]}") from None


def setup_twilio() -> str:
    """Create or reuse the Twilio Elastic SIP Trunk, its origination URL, and the
    phone-number attachment. Uses settings.twilio_sip_phone_number only."""
    sip_uri = f"sip:{livekit_sip_host()};transport=tcp"

    trunks = _twilio("GET", "https://trunking.twilio.com/v1/Trunks").get("trunks", [])
    trunk = next((t for t in trunks if t["friendly_name"] == TRUNK_NAME), None)
    if trunk:
        print(f"[twilio] reusing trunk {trunk['sid']}")
    else:
        trunk = _twilio("POST", "https://trunking.twilio.com/v1/Trunks", {"FriendlyName": TRUNK_NAME})
        print(f"[twilio] created trunk {trunk['sid']}")
    base = f"https://trunking.twilio.com/v1/Trunks/{trunk['sid']}"

    urls = _twilio("GET", f"{base}/OriginationUrls").get("origination_urls", [])
    if any(u["sip_url"] == sip_uri for u in urls):
        print(f"[twilio] origination URL already set: {sip_uri}")
    else:
        _twilio("POST", f"{base}/OriginationUrls", {
            "FriendlyName": "livekit", "SipUrl": sip_uri, "Priority": "1", "Weight": "1", "Enabled": "true",
        })
        print(f"[twilio] origination URL added: {sip_uri}")

    numbers = _twilio(
        "GET",
        f"https://api.twilio.com/2010-04-01/Accounts/{settings.twilio_account_sid}/IncomingPhoneNumbers.json"
        f"?PhoneNumber={urllib.parse.quote(settings.twilio_sip_phone_number)}",
    ).get("incoming_phone_numbers", [])
    if not numbers:
        raise RuntimeError(f"{settings.twilio_sip_phone_number} not found in this Twilio account")
    pn_sid = numbers[0]["sid"]
    attached = _twilio("GET", f"{base}/PhoneNumbers").get("phone_numbers", [])
    if any(p["sid"] == pn_sid for p in attached):
        print(f"[twilio] {settings.twilio_sip_phone_number} already attached to trunk")
    else:
        _twilio("POST", f"{base}/PhoneNumbers", {"PhoneNumberSid": pn_sid})
        print(f"[twilio] {settings.twilio_sip_phone_number} attached to trunk")
    return trunk["sid"]


def _env_set(key: str, value: str) -> None:
    text = ENV_PATH.read_text(encoding="utf-8") if ENV_PATH.exists() else ""
    if re.search(rf"^{key}=", text, flags=re.M):
        text = re.sub(rf"^{key}=.*$", f"{key}={value}", text, flags=re.M)
    else:
        text = text.rstrip() + f"{chr(10)}{key}={value}{chr(10)}"
    ENV_PATH.write_text(text, encoding="utf-8")


def setup_twilio_termination(trunk_sid: str) -> tuple[str, str, str]:
    """Give the trunk a termination domain + credentials so LiveKit can dial out
    through it later (outbound dialing itself is not wired up anywhere yet)."""
    sid = settings.twilio_account_sid
    base = f"https://trunking.twilio.com/v1/Trunks/{trunk_sid}"
    acct = f"https://api.twilio.com/2010-04-01/Accounts/{sid}"

    trunk = _twilio("GET", base)
    domain = trunk.get("domain_name")
    if not domain or "pstn.twilio.com" not in domain:
        domain = f"aeroassist-{sid[-8:].lower()}.pstn.twilio.com"
        trunk = _twilio("POST", base, {"DomainName": domain})
        print(f"[twilio] termination domain set: {domain}")
    else:
        print(f"[twilio] termination domain: {domain}")

    user = os.getenv("TWILIO_SIP_USERNAME") or "aeroassist"
    pwd = os.getenv("TWILIO_SIP_PASSWORD")
    cls = _twilio("GET", f"{acct}/SIP/CredentialLists.json").get("credential_lists", [])
    cl = next((c for c in cls if c["friendly_name"] == "aeroassist-livekit"), None)
    if cl and pwd:
        print("[twilio] credential list exists")
    else:
        if cl:  # exists but we lost the password: recreate
            _twilio("DELETE", f"{acct}/SIP/CredentialLists/{cl['sid']}.json")
        pwd = "Aa1-" + secrets.token_urlsafe(18)
        cl = _twilio("POST", f"{acct}/SIP/CredentialLists.json", {"FriendlyName": "aeroassist-livekit"})
        _twilio("POST", f"{acct}/SIP/CredentialLists/{cl['sid']}/Credentials.json", {"Username": user, "Password": pwd})
        _env_set("TWILIO_SIP_USERNAME", user)
        _env_set("TWILIO_SIP_PASSWORD", pwd)
        print("[twilio] credential list created (username/password saved to .env)")

    attached = _twilio("GET", f"{base}/CredentialLists").get("credential_lists", [])
    if not any(a["sid"] == cl["sid"] for a in attached):
        _twilio("POST", f"{base}/CredentialLists", {"CredentialListSid": cl["sid"]})
        print("[twilio] credential list attached to trunk")
    return domain, user, pwd


async def setup_livekit(domain: str, user: str, pwd: str) -> None:
    """Create or reuse the LiveKit inbound trunk + dispatch rule, and (re)create
    the LiveKit outbound trunk. Uses settings.twilio_sip_phone_number only -
    the existing Connector number/webhook is never referenced here."""
    lk = api.LiveKitAPI(
        settings.livekit_url.replace("wss://", "https://"),
        settings.livekit_api_key,
        settings.livekit_api_secret,
    )
    try:
        trunks = (await lk.sip.list_inbound_trunk(api.ListSIPInboundTrunkRequest())).items
        trunk = next((t for t in trunks if t.name == LK_TRUNK_NAME), None)
        if trunk:
            print(f"[livekit] reusing inbound trunk {trunk.sip_trunk_id}")
        else:
            trunk = await lk.sip.create_inbound_trunk(api.CreateSIPInboundTrunkRequest(
                trunk=api.SIPInboundTrunkInfo(name=LK_TRUNK_NAME, numbers=[settings.twilio_sip_phone_number])
            ))
            print(f"[livekit] created inbound trunk {trunk.sip_trunk_id}")

        rules = (await lk.sip.list_dispatch_rule(api.ListSIPDispatchRuleRequest())).items
        if any(r.name == LK_RULE_NAME for r in rules):
            print("[livekit] dispatch rule already exists")
        else:
            rule = await lk.sip.create_dispatch_rule(api.CreateSIPDispatchRuleRequest(
                rule=api.SIPDispatchRule(dispatch_rule_individual=api.SIPDispatchRuleIndividual(room_prefix=ROOM_PREFIX)),
                name=LK_RULE_NAME,
                trunk_ids=[trunk.sip_trunk_id],
            ))
            print(f"[livekit] created dispatch rule {rule.sip_dispatch_rule_id} (rooms: {ROOM_PREFIX}<caller>)")

        outs = (await lk.sip.list_outbound_trunk(api.ListSIPOutboundTrunkRequest())).items
        out = next((t for t in outs if t.name == LK_OUT_TRUNK_NAME), None)
        if out:
            await lk.sip.delete_trunk(api.DeleteSIPTrunkRequest(sip_trunk_id=out.sip_trunk_id))
        out = await lk.sip.create_outbound_trunk(api.CreateSIPOutboundTrunkRequest(
            trunk=api.SIPOutboundTrunkInfo(
                name=LK_OUT_TRUNK_NAME, address=domain, numbers=[settings.twilio_sip_phone_number],
                auth_username=user, auth_password=pwd,
            )
        ))
        print(f"[livekit] outbound trunk ready {out.sip_trunk_id} -> {domain}")
        _env_set("LIVEKIT_SIP_OUTBOUND_TRUNK_ID", out.sip_trunk_id)
    finally:
        await lk.aclose()


def main(apply: bool) -> None:
    print("AeroAssist - OPTIONAL native LiveKit SIP telephony path")
    print("This is separate from, and does not modify, the existing Twilio")
    print("Media Streams -> LiveKit Connector flow or its phone number.\n")
    print("Resources this would create or reuse (never duplicated, see module docstring):")
    for line in plan():
        print(f"  - {line}")

    if not apply:
        print("\nDry run only. No network calls were made and nothing was created.")
        print("Re-run with --apply to actually provision these resources.")
        return

    missing = [
        name for name, value in [
            ("TWILIO_ACCOUNT_SID", settings.twilio_account_sid),
            ("TWILIO_AUTH_TOKEN", settings.twilio_auth_token),
            ("TWILIO_SIP_PHONE_NUMBER", settings.twilio_sip_phone_number),
        ] if not value
    ]
    if missing:
        raise SystemExit(
            "Cannot --apply: missing required settings in .env: " + ", ".join(missing) +
            " (see .env.example). Nothing was created."
        )

    trunk_sid = setup_twilio()
    domain, user, pwd = setup_twilio_termination(trunk_sid)
    asyncio.run(setup_livekit(domain, user, pwd))
    print(
        f"\nDone. Inbound: {settings.twilio_sip_phone_number} -> room "
        f"'{ROOM_PREFIX}...' -> agent. Outbound dialing is still disabled in code."
    )


if __name__ == "__main__":
    main(apply="--apply" in sys.argv)
