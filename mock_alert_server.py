"""
mock_alert_server.py -- Small FastAPI server that simulates an "inbox" /
Slack webhook to test alerting WITHOUT real SMTP/Slack credentials.

Receives alerts sent by AlertDispatcher (via the senders defined below /
in bridge/senders.py) and logs them to a JSONL file on disk -- no real
email is ever sent, but the HTTP flow is real, so the whole chain
(AlertAgent -> AlertDispatcher -> HTTP sender -> server -> log) can be
tested end to end.

Run:
    pip install fastapi uvicorn
    uvicorn mock_alert_server:app --reload --port 8000

Endpoints:
    POST /email   {subject, to, body, priority, source_agent, ...}  -> log
    POST /slack    {text, priority, source_agent, ...}               -> log
    GET  /alerts   -> the last N logged alerts (email+slack)
    GET  /health   -> {"status": "ok"}

Log file: email_alert_log.jsonl (one JSON line per alert, created
automatically next to this script).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

LOG_PATH = Path(__file__).parent / "email_alert_log.jsonl"

app = FastAPI(title="Mock Alert Inbox", description="Test server for agentic_drift_stress's email/Slack alerting")


class EmailPayload(BaseModel):
    subject: str
    to: str
    body: str
    priority: Optional[str] = None
    source_agent: Optional[str] = None


class SlackPayload(BaseModel):
    text: str
    priority: Optional[str] = None
    source_agent: Optional[str] = None


def _append_log(channel: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    entry = {
        "channel": channel,
        "received_at": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/email")
def receive_email(payload: EmailPayload) -> Dict[str, Any]:
    entry = _append_log("email", payload.model_dump())
    return {"status": "logged", "entry": entry}


@app.post("/slack")
def receive_slack(payload: SlackPayload) -> Dict[str, Any]:
    entry = _append_log("slack", payload.model_dump())
    return {"status": "logged", "entry": entry}


@app.get("/alerts")
def list_alerts(limit: int = 50) -> List[Dict[str, Any]]:
    if not LOG_PATH.exists():
        return []
    lines = LOG_PATH.read_text(encoding="utf-8").strip().splitlines()
    entries = [json.loads(line) for line in lines[-limit:]]
    return list(reversed(entries))  # most recent first


@app.delete("/alerts")
def clear_alerts() -> Dict[str, str]:
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    return {"status": "cleared"}
