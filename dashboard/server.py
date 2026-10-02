"""
AeroAssist local operational dashboard: service health, live rooms, a
structured call-event viewer, a worker log tail, and a "talk to the agent"
browser test client.

This is a separate FastAPI app/process from the voice worker (agent/core_agent.py)
and the Twilio webhook (telephony/twilio_webhook.py). It only READS state
(service reachability pings, the events.jsonl file, the worker's log file,
LiveKit Cloud's room list) — it never touches an active call. Adapted from
aeroassist-voice-bot-main/dashboard/server.py for this project's actual
stack (Gemini/Cartesia/Deepgram over the existing Twilio Connector, not
Azure OpenAI/Speech over native SIP). Outbound calling is intentionally not
included here yet.

Run:  python -m dashboard.server      ->  http://localhost:8000
"""

import asyncio
import json
import re
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from livekit import api

from config.settings import settings

ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = ROOT / "logs" / "aeroassist.log"
EVENTS_FILE = ROOT / "logs" / "events.jsonl"
ANSI = re.compile(r"\x1b\[[0-9;]*m")

app = FastAPI(title="AeroAssist Dashboard")


def _http(url: str, headers: dict | None = None) -> tuple[bool, str]:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return True, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except Exception as e:  # noqa: BLE001
        return False, str(e)[:120]


def _lk() -> api.LiveKitAPI:
    return api.LiveKitAPI(
        settings.livekit_url.replace("wss://", "https://"), settings.livekit_api_key, settings.livekit_api_secret
    )


def _check_services() -> dict:
    s = settings
    out = {}
    ok, m = _http("https://api.deepgram.com/v1/projects", {"Authorization": f"Token {s.deepgram_api_key}"})
    out["Deepgram (STT)"] = {"ok": ok, "detail": "reachable" if ok else m}

    ok, m = _http(f"https://generativelanguage.googleapis.com/v1beta/models?key={s.gemini_api_key}")
    out["Gemini (LLM)"] = {"ok": ok, "detail": "reachable" if ok else m}

    ok, m = _http("https://api.cartesia.ai/voices", {"X-API-Key": s.cartesia_api_key, "Cartesia-Version": "2024-06-10"})
    out["Cartesia (TTS)"] = {"ok": ok, "detail": "reachable" if ok else m}

    return out


@app.get("/api/status")
async def status():
    services = await asyncio.to_thread(_check_services)
    lk = _lk()
    try:
        await lk.room.list_rooms(api.ListRoomsRequest())
        services["LiveKit Cloud"] = {"ok": True, "detail": settings.livekit_url}
    except Exception as e:  # noqa: BLE001
        services["LiveKit Cloud"] = {"ok": False, "detail": str(e)[:120]}
    finally:
        await lk.aclose()
    return {
        "services": services,
        "telephony": {
            "mode": "Twilio Media Streams -> LiveKit Connector (telephony/twilio_webhook.py)",
            "outbound_calling": "not enabled",
        },
    }


@app.get("/api/rooms")
async def rooms():
    lk = _lk()
    try:
        out = []
        for r in (await lk.room.list_rooms(api.ListRoomsRequest())).rooms:
            parts = (await lk.room.list_participants(api.ListParticipantsRequest(room=r.name))).participants
            out.append({
                "name": r.name,
                "created": r.creation_time,
                "participants": [
                    {"identity": p.identity, "kind": api.ParticipantInfo.Kind.Name(p.kind)} for p in parts
                ],
            })
        return out
    finally:
        await lk.aclose()


@app.get("/api/logs")
async def logs(lines: int = 200):
    if not LOG_FILE.exists():
        return {"lines": ["(no log file yet - restart the worker with `python run_local.py dev`)"]}
    text = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    return {"lines": [ANSI.sub("", ln)[:400] for ln in text if not ln.startswith("DEBUG:")]}


@app.get("/api/events")
async def events(limit: int = 300, call: str = ""):
    """Structured call events (newest last), optionally for one call/room."""
    if not EVENTS_FILE.exists():
        return []
    out = []
    for ln in EVENTS_FILE.read_text(encoding="utf-8", errors="replace").splitlines()[-5000:]:
        try:
            rec = json.loads(ln)
        except ValueError:
            continue
        if not call or rec.get("call") == call:
            out.append(rec)
    return out[-limit:]


@app.get("/api/token")
async def token():
    """Issue a short-lived LiveKit token for the browser 'talk to the agent' test client."""
    room = f"web-{uuid.uuid4().hex[:6]}"
    jwt = (
        api.AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
        .with_identity(f"web-user-{uuid.uuid4().hex[:4]}")
        .with_grants(api.VideoGrants(room_join=True, room=room))
        .to_jwt()
    )
    return {"token": jwt, "url": settings.livekit_url, "room": room}


@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "index.html")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
