"""
Structured, timestamped call-event log (JSON Lines), read by the dashboard.

Every record has: ts, call (room name), kind, source (file:function that
logged it), and free-form fields such as input / output. Adapted from the
same design in aeroassist-voice-bot-main's agent/event_log.py, with one
deliberate change: the actual file write is dispatched to a worker thread
via the running event loop's executor instead of happening inline, so a
call to log_event() from a session event handler never blocks the
voice pipeline's event loop on disk I/O.

Safe to call from any sync context on the agent's event loop.
"""

import asyncio
import json
import os
import sys
import threading
from datetime import datetime
from pathlib import Path

EVENTS_FILE = Path(__file__).resolve().parent.parent / "logs" / "events.jsonl"
_lock = threading.Lock()


def _write(record: dict) -> None:
    try:
        EVENTS_FILE.parent.mkdir(exist_ok=True)
        with _lock, EVENTS_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:  # logging must never break a call
        pass


def log_event(call: str, kind: str, **fields) -> None:
    """Append one event. `source` is filled automatically from the caller's file and function."""
    frame = sys._getframe(1)
    record = {
        "ts": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "call": call,
        "kind": kind,
        "source": f"{os.path.relpath(frame.f_code.co_filename, EVENTS_FILE.parent.parent)}:{frame.f_code.co_name}",
        **{k: v for k, v in fields.items() if v not in (None, "")},
    }
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    try:
        if loop is not None:
            loop.run_in_executor(None, _write, record)
        else:
            # No running loop (e.g. called from a sync script/test) - write directly.
            _write(record)
    except Exception:  # logging must never break the caller
        pass
