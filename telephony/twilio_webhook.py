"""
Twilio Media Streams webhook for AeroAssist.

Receives Twilio's inbound-call webhook, asks LiveKit Cloud's Connector
service to bridge the call into a new LiveKit room via Media Streams
(no SIP trunking involved), and returns TwiML pointing Twilio at the
returned WebSocket URL.

This module is fully isolated from the existing LiveKit worker
(agent/core_agent.py) and does not change its behavior in any way — it
only reuses the existing config.settings for LiveKit credentials.
"""

import logging
import uuid
from xml.sax.saxutils import quoteattr

from fastapi import FastAPI, Request, Response
from livekit import api

from config.settings import settings

logger = logging.getLogger("aeroassist.telephony")

app = FastAPI(title="AeroAssist Twilio Webhook")

# The existing AeroAssist worker (agent/core_agent.py, run_local.py) registers
# with WorkerOptions' default agent_name="" (automatic dispatch, unmodified —
# confirmed by inspecting livekit.agents.WorkerOptions' default and the
# worker's own registration logs, which all show "agent_name": ""). LiveKit
# Connector rooms dispatch agents explicitly via this `agents` field rather
# than through automatic discovery, so it must be set even though the name
# itself is empty, to match our worker's actual (unmodified) configuration.
AEROASSIST_AGENT_NAME = ""


def _new_room_name(call_sid: str | None) -> str:
    """Build a unique LiveKit room name for this phone call."""
    suffix = call_sid[-8:] if call_sid else uuid.uuid4().hex[:8]
    return f"twilio-call-{suffix}"


@app.get("/twilio/voice")
async def twilio_voice_reachability_check() -> Response:
    """
    Some Twilio Console flows send a GET request to a webhook URL to check
    reachability before saving it (the real call always POSTs). Without this,
    that check gets a 405 and the Console reports the URL as unreachable.
    This does not participate in call handling in any way.
    """
    return Response(content="OK", media_type="text/plain")


@app.post("/twilio/voice")
async def twilio_voice(request: Request) -> Response:
    """
    Twilio's "a call comes in" webhook target.

    Returns TwiML that streams the call's audio to a LiveKit room via
    LiveKit Cloud's Twilio Connector.
    """
    logger.info("Incoming Twilio voice webhook received")

    # CallSid is only used for room-name traceability; never log the full
    # form body, since it may contain caller phone numbers.
    call_sid = None
    try:
        form = await request.form()
        call_sid = form.get("CallSid")
    except Exception:
        logger.warning("Could not parse Twilio webhook form body; continuing without CallSid")

    room_name = _new_room_name(call_sid)
    logger.info("Generated LiveKit room name: %s", room_name)

    lkapi = api.LiveKitAPI(
        url=settings.livekit_url,
        api_key=settings.livekit_api_key,
        api_secret=settings.livekit_api_secret,
    )
    try:
        logger.info("Connector request started for room %s", room_name)
        connect_response = await lkapi.connector.connect_twilio_call(
            api.ConnectTwilioCallRequest(
                twilio_call_direction=(
                    api.ConnectTwilioCallRequest.TwilioCallDirection.TWILIO_CALL_DIRECTION_INBOUND
                ),
                room_name=room_name,
                agents=[api.RoomAgentDispatch(agent_name=AEROASSIST_AGENT_NAME)],
            )
        )
        logger.info("Connector request succeeded for room %s", room_name)
    except Exception:
        logger.exception("Connector request failed for room %s", room_name)
        return Response(
            content=(
                "<Response><Say>Sorry, we are unable to connect your call "
                "right now.</Say></Response>"
            ),
            media_type="application/xml",
            status_code=502,
        )
    finally:
        await lkapi.aclose()

    twiml = (
        "<Response><Connect>"
        f"<Stream url={quoteattr(connect_response.connect_url)}/>"
        "</Connect></Response>"
    )
    return Response(content=twiml, media_type="application/xml")
