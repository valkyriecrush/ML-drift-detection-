"""
senders.py -- All sender factories for AlertDispatcher.register() (see
alert_agent.py / orchestrator.py), grouped here since they all implement
the same shape: Callable[[Alert], None].

Merges the former `alert_log_senders.py` and `configure_real_channels.py`,
which each held senders with no relationship beyond "same pattern, same
consumer (AlertDispatcher)". Splitting them into "test" vs "prod" files
added no real boundary -- just a historical split. Grouped here in two
sections:

1. MOCK / TEST -- no infra required (local file) or a local mock FastAPI
   server (mock_alert_server.py). Used by default by test_no_drift.py /
   test_normal_drift.py / test_severe_drift.py.
2. REAL -- Slack (incoming webhook) + email (SMTP). Requires real external
   credentials, wire in via build_real_dispatcher().

Without any REAL sender registered, DEFAULT_ROUTING (alert_agent.py) still
routes WARNING/CRITICAL/PAGE alerts to AlertChannel.SLACK / EMAIL / PAGER,
but AlertDispatcher only knows the LOG sender by default: other channels
are just logged as "not sent" until something is explicitly registered
(mock or real) via dispatcher.register(...).
"""

from __future__ import annotations

import html as html_lib
import json
import logging
import os
import smtplib
import ssl
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import requests

from alert_agent import Alert, AlertChannel, AlertDispatcher, AlertPriority

logger = logging.getLogger(__name__)


# ======================================================================= #
# 0. HTML rendering for a SINGLE alert email -- same visual language as
# run_scenario.py::_build_executive_email (badge + accent color by
# severity, card layout), applied here to the per-Alert email sent by
# AlertDispatcher itself. Before this, an individual alert email was
# plain-text only (see the old make_email_sender body below) -- fine for
# a routine WARNING, not for the PAGE/CRITICAL alerts that are the whole
# reason EMAIL is even in DEFAULT_ROUTING: those need to be immediately
# scannable on a phone, not a wall of text.
# ======================================================================= #

_ALERT_PRIORITY_STYLE = {
    AlertPriority.PAGE: ("#7C0A0A", "\U0001F6A8"),      # dark red, rotating light
    AlertPriority.CRITICAL: ("#C0392B", "\U0001F534"),  # red, red circle
    AlertPriority.WARNING: ("#D68910", "\U0001F7E1"),   # amber, yellow circle
    AlertPriority.INFO: ("#2E86C1", "\U0001F535"),      # blue, blue circle
}


def _render_alert_html(alert: Alert) -> str:
    """One alert = one compact incident card: colored header by priority
    (impossible to miss at a glance, even skimming an inbox), the full
    message and recommended action (never truncated -- this is the ONE
    email an on-call engineer gets for this alert, it has to be
    self-sufficient), and the raw context as a table for anyone who wants
    the numbers behind the verdict without opening the attached report."""
    color, badge = _ALERT_PRIORITY_STYLE.get(alert.priority, ("#5D6D7E", "\u26AA"))
    e = html_lib.escape
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    context_rows = "".join(
        f'<tr><td style="padding:3px 12px 3px 0;color:#5D6D7E;font-size:12px;white-space:nowrap;vertical-align:top;">{e(str(k))}</td>'
        f'<td style="padding:3px 0;font-size:12px;color:#1C2833;font-family:monospace;">{e(json.dumps(v, default=str))}</td></tr>'
        for k, v in (alert.context or {}).items()
    )
    context_html = (
        f'<table role="presentation" cellpadding="0" cellspacing="0" style="margin-top:8px;">{context_rows}</table>'
        if context_rows else ""
    )

    action_html = (
        f'<p style="margin:12px 0 0;padding:12px 16px;background-color:#FDF2F2;border-left:3px solid {color};'
        f'font-size:13px;line-height:1.5;color:#1C2833;"><strong>Recommended action:</strong><br>{e(alert.recommended_action)}</p>'
        if alert.recommended_action else ""
    )

    return f"""\
<html>
  <body style="margin:0;padding:24px;background-color:#F4F6F7;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
      <tr>
        <td align="center">
          <table role="presentation" width="600" cellpadding="0" cellspacing="0"
                 style="background-color:#FFFFFF;border-radius:8px;overflow:hidden;
                        box-shadow:0 1px 3px rgba(0,0,0,0.08);">
            <tr>
              <td style="background-color:{color};padding:16px 24px;">
                <span style="color:#FFFFFF;font-size:16px;font-weight:700;">
                  {badge} [{e(alert.priority.value.upper())}] {e(alert.title)}
                </span>
              </td>
            </tr>
            <tr>
              <td style="padding:24px;">
                <table role="presentation" cellpadding="0" cellspacing="0" style="margin-bottom:16px;">
                  <tr><td style="padding:2px 12px 2px 0;color:#5D6D7E;font-size:13px;">Source agent</td>
                      <td style="padding:2px 0;font-size:13px;color:#1C2833;font-weight:600;">{e(alert.source_agent)}</td></tr>
                  <tr><td style="padding:2px 12px 2px 0;color:#5D6D7E;font-size:13px;">Remediation type</td>
                      <td style="padding:2px 0;font-size:13px;color:#1C2833;font-weight:600;">{e(alert.remediation_type.value)}</td></tr>
                  <tr><td style="padding:2px 12px 2px 0;color:#5D6D7E;font-size:13px;">Detected at</td>
                      <td style="padding:2px 0;font-size:13px;color:#1C2833;">{e(alert.timestamp)}</td></tr>
                </table>
                <p style="margin:0;font-size:14px;line-height:1.5;color:#1C2833;white-space:pre-line;">{e(alert.message)}</p>
                {action_html}
                {context_html}
              </td>
            </tr>
            <tr>
              <td style="padding:12px 24px;background-color:#F8F9F9;border-top:1px solid #EAECEE;">
                <span style="font-size:11px;color:#909497;">
                  Sent automatically by the drift monitoring pipeline &middot; {generated_at}
                </span>
              </td>
            </tr>
          </table>
        </td>
      </tr>
    </table>
  </body>
</html>
"""


# ======================================================================= #
# 1. MOCK / TEST -- no real external infra required
# ======================================================================= #

# ----------------------------------------------------------------------- #
# 1a. Via mock_alert_server.py (real HTTP, local server required)
# ----------------------------------------------------------------------- #
def make_http_email_sender(base_url: str = "http://127.0.0.1:8000", to: str = "ml-oncall@example.com") -> Callable[[Alert], None]:
    def _sender(alert: Alert) -> None:
        payload = {
            "subject": f"[{alert.priority.value.upper()}] {alert.title}",
            "to": to,
            "body": f"{alert.message}\n\nRecommended action:\n{alert.recommended_action or '(none)'}",
            "priority": alert.priority.value,
            "source_agent": alert.source_agent,
            "html_body": _render_alert_html(alert),
        }
        resp = requests.post(f"{base_url}/email", json=payload, timeout=5)
        resp.raise_for_status()
        logger.info(f"[EMAIL SENT] channel=http to={to} subject='{payload['subject']}' -> {base_url}/email")
    return _sender


def make_http_slack_sender(base_url: str = "http://127.0.0.1:8000") -> Callable[[Alert], None]:
    def _sender(alert: Alert) -> None:
        requests.post(
            f"{base_url}/slack",
            json={
                "text": f"[{alert.source_agent}] {alert.title} :: {alert.message}",
                "priority": alert.priority.value,
                "source_agent": alert.source_agent,
            },
            timeout=5,
        )
    return _sender


# ----------------------------------------------------------------------- #
# 1b. Local file, no server (default for the 3 test_*.py scripts)
# ----------------------------------------------------------------------- #
def make_file_email_sender(log_path: str = "email_alert_log.jsonl") -> Callable[[Alert], None]:
    path = Path(log_path)

    def _sender(alert: Alert) -> None:
        entry = {
            "channel": "email",
            "received_at": datetime.now(timezone.utc).isoformat(),
            "subject": f"[{alert.priority.value.upper()}] {alert.title}",
            "to": "ml-oncall@example.com",
            "body": f"{alert.message}\n\nRecommended action:\n{alert.recommended_action or '(none)'}",
            "priority": alert.priority.value,
            "source_agent": alert.source_agent,
            "remediation_type": alert.remediation_type.value,
            "context": alert.context,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        logger.info(f"[EMAIL SENT] channel=file to=ml-oncall@example.com subject='{entry['subject']}' -> {path}")

    return _sender


# ======================================================================= #
# 2. REAL -- Slack (incoming webhook) + email (SMTP)
# ======================================================================= #
#
# Requires:
# - Slack: a Slack incoming webhook (Slack > Apps > Incoming Webhooks),
#   env var SLACK_WEBHOOK_URL. Needs `pip install requests`.
# - Email: an SMTP-capable account (Gmail with an app password, SES,
#   Sendgrid SMTP relay, an internal server...). Env vars SMTP_HOST,
#   SMTP_PORT, SMTP_USER, SMTP_PASSWORD, ALERT_EMAIL_FROM, ALERT_EMAIL_TO.
#   Uses smtplib (stdlib, nothing to install).

def make_slack_sender(webhook_url: str) -> Callable[[Alert], None]:
    def _slack_sender(alert: Alert) -> None:
        emoji = {
            AlertPriority.INFO: ":information_source:",
            AlertPriority.WARNING: ":warning:",
            AlertPriority.CRITICAL: ":rotating_light:",
            AlertPriority.PAGE: ":sos:",
        }[alert.priority]
        text = f"{emoji} *[{alert.source_agent}] {alert.title}*\n{alert.message}"
        resp = requests.post(webhook_url, json={"text": text}, timeout=5)
        resp.raise_for_status()
    return _slack_sender


def _send_plain_email(
    host: str, port: int, user: str, password: str, from_addr: str, to_addr: str,
    subject: str, body: str, attachments: Optional[List[Union[str, Path]]] = None,
    html_body: Optional[str] = None,
) -> None:
    """Low-level core, shared by make_email_sender() (a single alert) and
    send_report_email() (the end-of-scenario summary report) -- avoids
    duplicating the smtplib/STARTTLS logic in two places.

    `attachments`: optional list of file paths attached as-is (e.g. the
    full technical diagnostic report, see diagnostic_report.py).
    `html_body`: optional HTML alternative rendered by clients that
    support it (Gmail/Outlook/Apple Mail...); `body` (plain text) always
    stays the fallback for clients that don't. Both are meant to stay
    SHORT -- the detailed technical content belongs in an attachment, not
    in the email body itself."""
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg.set_content(body)
    if html_body:
        msg.add_alternative(html_body, subtype="html")
    for raw_path in attachments or []:
        path = Path(raw_path)
        data = path.read_bytes()
        _MIME_BY_SUFFIX = {".md": ("text", "markdown"), ".pdf": ("application", "pdf")}
        maintype, subtype = _MIME_BY_SUFFIX.get(path.suffix, ("application", "octet-stream"))
        msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=path.name)
    context = ssl.create_default_context()
    with smtplib.SMTP(host, port, timeout=10) as server:
        server.starttls(context=context)
        server.login(user, password)
        server.send_message(msg)


def make_email_sender(host: str, port: int, user: str, password: str, from_addr: str, to_addr: str) -> Callable[[Alert], None]:
    def _email_sender(alert: Alert) -> None:
        subject = f"[{alert.priority.value.upper()}] {alert.title}"
        _send_plain_email(
            host=host, port=port, user=user, password=password,
            from_addr=from_addr, to_addr=to_addr,
            subject=subject,
            body=(
                f"Source: {alert.source_agent}\n"
                f"Priority: {alert.priority.value}\n\n"
                f"{alert.message}\n\n"
                f"Recommended action:\n{alert.recommended_action or '(none)'}"
            ),
            # AUDIT FIX (unprofessional / low-visibility alert emails): an
            # individual alert -- the PAGE/CRITICAL ones especially -- used
            # to go out as bare plain text, no different in appearance from
            # a routine INFO line. _render_alert_html() gives it the same
            # severity-colored card treatment as the executive summary
            # email (run_scenario.py::_build_executive_email), so a CRITICAL
            # or PAGE alert is unmistakable at a glance in an inbox, with
            # the full message/recommended_action/context inline -- this
            # IS the notification, it must be self-sufficient, not a teaser
            # for an attachment.
            html_body=_render_alert_html(alert),
        )
        # AUDIT FIX (missing delivery confirmation): the previous version
        # of this sender returned silently on success -- SMTP not raising
        # was the only signal anything happened. AlertDispatcher.dispatch()
        # now also logs a generic "[ALERT DELIVERED]" line per channel (see
        # alert_agent.py), but this one is SMTP-specific and confirms the
        # actual recipient/subject that went out, for anyone grepping mail
        # logs specifically rather than the agent's own logger.
        logger.info(f"[EMAIL SENT] channel=smtp host={host} to={to_addr} subject='{subject}'")
    return _email_sender


def _require_env(name: str) -> str:
    """Raises with an actionable message as soon as the value is read,
    rather than letting an opaque KeyError propagate (this used to crash
    all of run_scenario.py, including the local-file fallback, over a
    single forgotten env var)."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"SMTP_HOST is set but {name} is missing/empty. "
            f"Required variables for real SMTP sending: SMTP_HOST, SMTP_USER, "
            f"SMTP_PASSWORD, ALERT_EMAIL_FROM, ALERT_EMAIL_TO "
            f"(see .env.example or bridge/senders.py)."
        )
    return value


def send_report_email(
    subject: str, body: str, attachments: Optional[List[Union[str, Path]]] = None,
    html_body: Optional[str] = None,
) -> bool:
    """Sends the end-of-scenario summary report (NOT an individual Alert)
    as a single email, reading the SMTP_* env vars directly. Returns False
    (without raising) if SMTP_HOST isn't configured -- deliberately
    best-effort, a report that doesn't go out should never crash
    run_scenario.py.

    `body`/`html_body` are meant to be a short, professional executive
    summary (see run_scenario.py::_build_executive_email); the full
    technical detail (PSI per feature, SHAP rationale, remediation
    actions) belongs in `attachments` (typically the Markdown file already
    written to disk by diagnostic_report.py::report_node), not in the
    body itself."""
    smtp_host = os.environ.get("SMTP_HOST")
    if not smtp_host:
        return False
    to_addr = _require_env("ALERT_EMAIL_TO")
    _send_plain_email(
        host=smtp_host,
        port=int(os.environ.get("SMTP_PORT", "587")),
        user=_require_env("SMTP_USER"),
        password=_require_env("SMTP_PASSWORD"),
        from_addr=_require_env("ALERT_EMAIL_FROM"),
        to_addr=to_addr,
        subject=subject,
        body=body,
        attachments=attachments,
        html_body=html_body,
    )
    logger.info(f"[EMAIL SENT] channel=smtp host={smtp_host} to={to_addr} subject='{subject}'")
    return True


def build_real_dispatcher() -> AlertDispatcher:
    """Creates an AlertDispatcher with Slack + email actually wired in, to
    pass to AlertAgent(dispatcher=...) instead of the default LOG-only one."""
    dispatcher = AlertDispatcher()  # default routing (DEFAULT_ROUTING) kept

    slack_webhook = os.environ.get("SLACK_WEBHOOK_URL")
    if slack_webhook:
        dispatcher.register(AlertChannel.SLACK, make_slack_sender(slack_webhook))

    smtp_host = os.environ.get("SMTP_HOST")
    if smtp_host:
        dispatcher.register(AlertChannel.EMAIL, make_email_sender(
            host=smtp_host,
            port=int(os.environ.get("SMTP_PORT", "587")),
            user=_require_env("SMTP_USER"),
            password=_require_env("SMTP_PASSWORD"),
            from_addr=_require_env("ALERT_EMAIL_FROM"),
            to_addr=_require_env("ALERT_EMAIL_TO"),
        ))

    # AlertChannel.PAGER (PagerDuty, Opsgenie...): same idea, not provided
    # here -- wire a sender via dispatcher.register(AlertChannel.PAGER, ...)

    return dispatcher


if __name__ == "__main__":
    # Usage: wire this dispatcher into orchestrator.py instead of the
    # default AlertDispatcher(), by changing how _alert_agent is created:
    #
    #   from senders import build_real_dispatcher
    #   from alert_agent import AlertAgent
    #   _alert_agent = AlertAgent(dispatcher=build_real_dispatcher())
    #
    # then run with the required environment variables exported:
    #   export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/xxx
    #   export SMTP_HOST=smtp.gmail.com SMTP_USER=... SMTP_PASSWORD=...
    #   export ALERT_EMAIL_FROM=alerts@mycompany.com ALERT_EMAIL_TO=ml-oncall@mycompany.com
    dispatcher = build_real_dispatcher()
    print("Wired channels:", list(dispatcher._senders.keys()))
