"""
test_email_sender_real_smtp.py -- REAL (not mocked) integration test for
make_email_sender() / build_real_dispatcher() in bridge/senders.py.

Why a real test instead of a mock?
A smtplib.SMTP mock only proves the code CALLS starttls(), login(),
send_message() in the right order -- it doesn't prove that:
  - the SMTP server actually accepts this host/port/TLS,
  - the credentials are valid,
  - the message actually goes out and arrives in a real inbox.
This test fills that gap by sending a REAL email to a REAL inbox.

Skipped by default (no env vars = no test runs, no risk of breaking CI or
sending an email on every `pytest tests/`). It only activates if you
explicitly export:

    export SMTP_HOST=smtp.gmail.com
    export SMTP_PORT=587
    export SMTP_USER=your_address@gmail.com
    export SMTP_PASSWORD=your_16_letter_app_password
    export ALERT_EMAIL_FROM=your_address@gmail.com
    export ALERT_EMAIL_TO=your_address@gmail.com

Then run ONLY this file (it isn't meant to run on every `pytest tests/` in
CI, only on demand):

    pytest tests/test_email_sender_real_smtp.py -v -s

Verification: check the ALERT_EMAIL_TO inbox (and Spam folder) after the
run. The test only checks that smtplib didn't raise -- it can't verify by
code that the email visually arrived.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from alert_agent import Alert, AlertPriority
from senders import make_email_sender

# Load .env (project root) if present, BEFORE reading REQUIRED_ENV_VARS
# below -- otherwise this test would never see values set in .env, only
# ones exported manually in the current shell.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

REQUIRED_ENV_VARS = (
    "SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "ALERT_EMAIL_FROM", "ALERT_EMAIL_TO",
)

_missing = [v for v in REQUIRED_ENV_VARS if not os.environ.get(v)]

pytestmark = pytest.mark.skipif(
    bool(_missing),
    reason=(
        "Real integration test disabled: missing SMTP env vars "
        f"({', '.join(_missing) or 'none'}). See this file's docstring "
        "to export them and enable a real send."
    ),
)


def test_real_smtp_sends_an_actual_email_end_to_end():
    """Sends a REAL email via the REAL configured SMTP server. No mock: if
    the host/port/credentials are wrong, this test fails with the actual
    smtplib exception (SMTPAuthenticationError, gaierror, etc.), exactly
    the error a user would see in production."""
    sender = make_email_sender(
        host=os.environ["SMTP_HOST"],
        port=int(os.environ.get("SMTP_PORT", "587")),
        user=os.environ["SMTP_USER"],
        password=os.environ["SMTP_PASSWORD"],
        from_addr=os.environ["ALERT_EMAIL_FROM"],
        to_addr=os.environ["ALERT_EMAIL_TO"],
    )

    alert = Alert(
        source_agent="test_email_sender_real_smtp",
        priority=AlertPriority.CRITICAL,
        title="Real integration test -- agentic_drift_stress",
        message=(
            "This email confirms bridge/senders.py:make_email_sender() "
            "does send a real email via a real SMTP server. "
            "No action required, this is a test."
        ),
        recommended_action="None -- this is a test email.",
    )

    # No assertion on a return value: make_email_sender() returns nothing
    # (None). The test IS the absence of an exception from smtplib --
    # exactly how AlertDispatcher._send_with_retry() treats it in
    # production (see alert_agent.py).
    sender(alert)

    print(
        f"\nEmail sent to {os.environ['ALERT_EMAIL_TO']} via "
        f"{os.environ['SMTP_HOST']}:{os.environ.get('SMTP_PORT', '587')}. "
        "Check the inbox (and Spam folder)."
    )
