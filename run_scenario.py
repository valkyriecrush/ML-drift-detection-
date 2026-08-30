"""
run_scenario.py -- Logic shared by the 3 test_*.py scripts, using the REAL
pipeline (see bridge/real_scenario_runner.py): real LightGBM model, real
preprocessing pipeline (drift_simulator/src/*.py), real drift simulators
(NoDriftSimulator / NormalDriftSimulator / SevereDriftSimulator).

1) bootstraps the agent (agentic_drift_stress) on the real baseline,
2) wires AlertAgent to an EMAIL sender (local file + FastAPI mock if running),
3) runs the scenario's batch through the single graph
   (orchestrator.build_graph()), once with trigger="batch" (data drift /
   proxy target drift path) then once with trigger="labeled_window"
   (performance drift / confirmed target drift path) -- see the single-graph
   redesign documented in agentic_drift_stress/orchestrator.py,
4) prints + returns a summary.
"""

from __future__ import annotations

import html as html_lib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "agentic_drift_stress"))
sys.path.insert(0, str(ROOT / "bridge"))

# Load .env BEFORE any os.environ access below (build_real_dispatcher /
# send_report_email read SMTP_HOST/SLACK_WEBHOOK_URL/etc.). If .env doesn't

# still works alongside it.
load_dotenv(ROOT / ".env.txt")

import orchestrator  # noqa: E402
from alert_agent import AlertAgent, AlertDispatcher, AlertChannel  # noqa: E402
from performance_drift_agent import TaskType  # noqa: E402
from real_scenario_runner import (  # noqa: E402
    build_baseline_calculator, get_baseline_predictions, run_real_scenario, RealScenarioBatch,
)
from feature_importance import compute_feature_intelligence_context  # noqa: E402
from senders import (  # noqa: E402
    make_file_email_sender, make_http_email_sender, build_real_dispatcher, send_report_email,
)

EMAIL_LOG_PATH = ROOT / "email_alert_log.jsonl"
REPORTS_DIR = ROOT / "reports"
MOCK_SERVER_URL = "http://127.0.0.1:8000"

# Status -> accent color for the HTML email (badge + left border), from most
# to least severe. Falls back to the "unknown" entry for anything else.
_STATUS_STYLE = {
    "PAGE": ("#7C0A0A", "🚨"),
    "CRITICAL": ("#C0392B", "🔴"),
    "DEGRADED": ("#D68910", "🟡"),
    "WARNING": ("#D68910", "🟡"),
    "INFO": ("#2E86C1", "🔵"),
    "HEALTHY": ("#1E8449", "🟢"),
    "NONE": ("#1E8449", "🟢"),
    "UNKNOWN": ("#5D6D7E", "⚪"),
}


def _mock_server_is_up() -> bool:
    try:
        return requests.get(f"{MOCK_SERVER_URL}/health", timeout=0.5).status_code == 200
    except Exception:
        return False


def _wire_alerting() -> None:
    """EMAIL channel priority, from most real to most simulated:
    1) real SMTP if SMTP_HOST is set (see bridge/senders.py) -- actually
       goes out over the internet, lands in a real inbox.
    2) mock_alert_server.py if running (local HTTP, never delivered to a
       real recipient -- useful for testing the flow without credentials).
    3) local file, always active alongside the other two, for a readable
       trace even without a server/SMTP.
    """
    dispatcher = AlertDispatcher()
    file_sender = make_file_email_sender(str(EMAIL_LOG_PATH))
    dispatcher.register(AlertChannel.EMAIL, file_sender)

    smtp_configured = bool(os.environ.get("SMTP_HOST"))
    if smtp_configured:
        real_dispatcher = build_real_dispatcher()
        real_email_sender = real_dispatcher._senders.get(AlertChannel.EMAIL)
        real_slack_sender = real_dispatcher._senders.get(AlertChannel.SLACK)
        if real_email_sender:
            def _file_and_real(alert, _file=file_sender, _real=real_email_sender):
                _file(alert)
                _real(alert)
            dispatcher.register(AlertChannel.EMAIL, _file_and_real)
        if real_slack_sender:
            dispatcher.register(AlertChannel.SLACK, real_slack_sender)
        print(f"[run_scenario] SMTP_HOST detected -> REAL emails sent via {os.environ['SMTP_HOST']} "
              f"(in addition to the local log {EMAIL_LOG_PATH.name}).")
    elif _mock_server_is_up():
        http_sender = make_http_email_sender(MOCK_SERVER_URL)

        def _both(alert, _file=file_sender, _http=http_sender):
            _file(alert)
            _http(alert)

        dispatcher.register(AlertChannel.EMAIL, _both)
        print(f"[run_scenario] mock_alert_server detected on {MOCK_SERVER_URL} -> alerts also sent over HTTP "
              f"(SIMULATION -- no real email is sent; export SMTP_HOST for real delivery).")
    else:
        print(f"[run_scenario] EMAIL alerts logged only to {EMAIL_LOG_PATH.name} "
              f"(SIMULATION -- no real email is sent; export SMTP_HOST/SMTP_USER/SMTP_PASSWORD/"
              f"ALERT_EMAIL_FROM/ALERT_EMAIL_TO for real delivery, see bridge/senders.py).")

    orchestrator._alert_agent = AlertAgent(dispatcher=dispatcher)


def _pick_report_attachment(summary: Dict[str, Any]) -> "tuple[Optional[Path], str]":
    """Picks which on-disk diagnostic report to attach to the email:
    PDF (`report_pdf_path`) when available -- opens on any device without
    a Markdown renderer, the professional format we actually want to send
    someone -- falling back to the Markdown file (`report_path`) if PDF
    rendering failed this cycle (best-effort, see
    diagnostic_report.py::DiagnosticReportGenerator.save()). Returns
    (None, "") if neither file is actually present on disk."""
    pdf_path = summary.get("report_pdf_path")
    if pdf_path and Path(pdf_path).exists():
        return Path(pdf_path), "PDF"
    md_path = summary.get("report_path")
    if md_path and Path(md_path).exists():
        return Path(md_path), "Markdown"
    return None, ""


def _build_executive_email(summary: Dict[str, Any]) -> "tuple[str, str, str]":
    """Builds a SHORT, professional email (subject, plain-text body,
    HTML body) -- an executive summary someone would actually forward to
    a stakeholder, not a raw technical dump. The full technical detail
    (PSI per feature, SHAP rationale, remediation actions, all alerts)
    already lives in the Markdown diagnostic report persisted to disk by
    report_node (see diagnostic_report.py) -- that file is attached
    separately, it is never inlined into the body here.

    The HTML body is the one most mail clients (Gmail/Outlook/Apple Mail)
    render; the plain-text body is the fallback for clients that don't
    render HTML (and for `email_alert_log.jsonl` readability)."""
    status = (summary.get("report_status") or "unknown").upper()
    color, badge = _STATUS_STYLE.get(status, _STATUS_STYLE["UNKNOWN"])
    subject = f"[{status}] Monitoring report -- {summary['scenario']} ({summary['n_alerts']} alert(s))"

    exec_summary = summary.get("report_executive_summary") or (
        "No executive summary available (diagnostic report was not generated this cycle)."
    )
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Prefer the PDF (professional, opens on any device without a Markdown
    # renderer); PDF rendering is best-effort (see diagnostic_report.py),
    # so fall back to the Markdown file when it wasn't produced this cycle.
    attachment_path, attachment_kind = _pick_report_attachment(summary)

    # -- Plain-text fallback -------------------------------------------------
    text_lines = [
        f"Model monitoring cycle completed for '{summary['scenario']}' (model_id: {summary.get('model_id', 'n/a')}).",
        "",
        f"Overall status: {status}",
        "",
        exec_summary,
        "",
        f"{summary['n_alerts']} alert(s) generated this cycle.",
    ]
    if attachment_path:
        text_lines += [
            "",
            f"The full technical report (PSI per feature, SHAP-based rationale, "
            f"remediation actions) is attached as a {attachment_kind} file.",
        ]
    text_lines += ["", "-- Sent automatically by the drift monitoring pipeline."]
    text_body = "\n".join(text_lines)

    # -- HTML version: a compact "incident/report card", not a wall of text --
    e = html_lib.escape
    metric_rows = [
        ("Scenario", summary["scenario"]),
        ("Model ID", summary.get("model_id", "n/a")),
        ("Status", status),
        ("Alerts this cycle", str(summary["n_alerts"])),
        ("Generated at", generated_at),
    ]
    metrics_html = "".join(
        f'<tr><td style="padding:4px 12px 4px 0;color:#5D6D7E;font-size:13px;white-space:nowrap;">{e(k)}</td>'
        f'<td style="padding:4px 0;font-size:13px;color:#1C2833;font-weight:600;">{e(v)}</td></tr>'
        for k, v in metric_rows
    )
    attachment_note_html = (
        f'<p style="margin:16px 0 0;font-size:13px;color:#5D6D7E;">'
        f'📎 The full technical report (PSI per feature, SHAP-based rationale, remediation actions) '
        f'is attached as a {e(attachment_kind)} file.</p>'
    ) if attachment_path else ""

    html_body = f"""\
<html>
  <body style="margin:0;padding:0;background-color:#F4F6F7;font-family:Segoe UI,Helvetica,Arial,sans-serif;">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background-color:#F4F6F7;padding:24px 0;">
      <tr>
        <td align="center">
          <table role="presentation" width="560" cellpadding="0" cellspacing="0"
                 style="background-color:#FFFFFF;border-radius:8px;overflow:hidden;
                        box-shadow:0 1px 3px rgba(0,0,0,0.08);">
            <tr>
              <td style="background-color:{color};padding:16px 24px;">
                <span style="color:#FFFFFF;font-size:16px;font-weight:600;">
                  {badge} Model monitoring report -- {e(status)}
                </span>
              </td>
            </tr>
            <tr>
              <td style="padding:24px;">
                <p style="margin:0 0 16px;font-size:14px;color:#1C2833;">
                  Monitoring cycle completed for <strong>{e(summary['scenario'])}</strong>.
                </p>
                <table role="presentation" cellpadding="0" cellspacing="0" style="margin-bottom:16px;">
                  {metrics_html}
                </table>
                <p style="margin:0;padding:12px 16px;background-color:#F8F9F9;border-left:3px solid {color};
                          font-size:13px;line-height:1.5;color:#1C2833;">
                  {e(exec_summary)}
                </p>
                {attachment_note_html}
              </td>
            </tr>
            <tr>
              <td style="padding:12px 24px;background-color:#F8F9F9;border-top:1px solid #EAECEE;">
                <span style="font-size:11px;color:#909497;">
                  Sent automatically by the drift monitoring pipeline.
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
    return subject, text_body, html_body


def _generate_and_send_report(summary: Dict[str, Any]) -> None:
    """Sends the short, structured executive-summary email built by
    _build_executive_email() (HTML with a plain-text fallback), with the
    full technical diagnostic report (already written to disk by
    diagnostic_report.py::report_node) attached -- instead of dumping the
    whole technical detail into the email body. Best effort: never
    crashes the scenario if sending fails, same as AlertDispatcher for
    individual alerts."""
    subject, text_body, html_body = _build_executive_email(summary)

    attachment_path, attachment_kind = _pick_report_attachment(summary)
    attachments = [attachment_path] if attachment_path else []
    if not attachment_path and summary.get("report_path"):
        print(f"[run_scenario] Diagnostic report path set ({summary.get('report_path')}) but file not found -- sending summary only.")
    elif attachment_kind == "Markdown":
        print("[run_scenario] PDF rendering unavailable this cycle -- attaching the Markdown report instead.")

    try:
        sent = send_report_email(subject=subject, body=text_body, html_body=html_body, attachments=attachments)
    except Exception as exc:  # best effort, see docstring
        print(f"[run_scenario] Failed to email the report: {exc}")
        return

    if sent:
        attached = f" (attached: {attachment_path.name})" if attachment_path else ""
        print(f"[run_scenario] Executive summary emailed to {os.environ.get('ALERT_EMAIL_TO')}{attached}.")
    else:
        print("[run_scenario] Report NOT emailed (SMTP_HOST not configured, see .env.example) "
              f"-- full technical report available at {attachment_path or summary.get('report_path') or REPORTS_DIR}.")


def run_scenario(scenario_name: str, model_id: str, severity: str = "severe") -> Dict[str, Any]:
    _wire_alerting()

    bc, cfg = build_baseline_calculator()
    tracked = cfg["baseline"]["tracked_features"]
    y_true_b, y_pred_b, y_proba_b = get_baseline_predictions(bc, cfg)

    # Always computed, for all 3 scenarios: without it, feature_intelligence_node
    # fires (once severity reaches WARNING/CRITICAL) but produces no verdict,
    # and the agent can't distinguish drift on Insulin/Glucose (strongly
    # predictive, see feature_importance.py) from drift on a minor feature --
    # the decision (which severity, which alert) would ignore real model impact.
    fi_context = compute_feature_intelligence_context(bc, cfg)

    orchestrator.bootstrap_agents(
        model_id=model_id,
        task_type=TaskType.CLASSIFICATION,
        reference_features=bc.baseline_df[tracked],
        y_reference=y_true_b,
        y_true_baseline=y_true_b,
        y_pred_baseline=y_pred_b,
        y_proba_baseline=y_proba_b,
        critical_classes=[1],
        critical_recall_floor=0.5,
        use_calibration=True,
        feature_intelligence_context=fi_context,
    )

    batch: RealScenarioBatch = run_real_scenario(scenario_name, bc, cfg, severity=severity)

    # Single compiled graph, invoked TWICE with the same thread_id=model_id
    # (same checkpointer) -- only state["trigger"] changes the path taken
    # inside the graph.
    graph = orchestrator.build_graph()
    config = {"configurable": {"thread_id": model_id}}
    batch_id = f"{batch.name}_batch"

    orchestrator.stage_batch(batch_id, features_df=batch.features_df, y_pred=batch.y_pred)
    fast_state = graph.invoke(
        {"model_id": model_id, "batch_id": batch_id, "trigger": "batch"}, config=config
    )
    orchestrator.clear_batch(batch_id)

    # features_df also provided here (not just y_true/y_pred/y_proba): see
    # build_graph()'s note on process_root_cause(), which needs a fresh
    # data_drift_report computed on the SAME rows as this labeled_window
    # cycle, not a report left over from the previous 'batch' cycle.
    label_batch_id = f"{batch.name}_labels"
    orchestrator.stage_batch(
        label_batch_id,
        features_df=batch.features_df,
        y_true=batch.y_true,
        y_pred=batch.y_pred,
        y_proba=batch.y_proba,
    )
    slow_state = graph.invoke(
        {"model_id": model_id, "batch_id": label_batch_id, "trigger": "labeled_window"}, config=config
    )
    orchestrator.clear_batch(label_batch_id)

    print(f"\n{'=' * 70}")
    print(f"SCENARIO: {batch.name}  (real pipeline: real LightGBM model + real preprocessing)")
    print(f"{'=' * 70}")
    native_status = batch.report.get("overall_drift_status", batch.report).get("status", batch.report.get("severity"))
    print(f"Simulator native status (drift_simulator, its own PSI convention): {native_status}")
    print()
    print("-- trigger='batch' (data drift + proxy target drift) -- our agent --")
    print("  Data drift severity  :", fast_state["data_drift_report"].overall_severity)
    print("  Target drift status  :", fast_state["target_drift_result"].status)
    verdicts = fast_state.get("combined_feature_verdicts", [])
    if verdicts:
        print("  Feature intelligence verdicts (SHAP/VIF):")
        for v in verdicts:
            print(f"    - {v}")
    print()
    print("-- trigger='labeled_window' (performance + confirmed target drift) -- our agent --")
    print("  Performance severity :", slow_state["performance_drift_result"].severity)
    print("  Target drift status  :", slow_state["target_drift_result"].status)
    print()
    alerts = slow_state.get("alerts", [])
    print(f"  Alerts generated this cycle: {len(alerts)}")
    for a in alerts:
        print(f"    - [{a['priority']}] ({a['source_agent']}) {a['title']}")

    summary = {
        "scenario": batch.name,
        "model_id": model_id,
        "native_simulator_status": native_status,
        "data_drift_severity": str(fast_state["data_drift_report"].overall_severity),
        "target_drift_status_fast": str(fast_state["target_drift_result"].status),
        "performance_severity": str(slow_state["performance_drift_result"].severity),
        "target_drift_status_slow": str(slow_state["target_drift_result"].status),
        "n_alerts": len(alerts),
        "alerts": alerts,
        # From diagnostic_report.py::report_node, via the "labeled_window"
        # invocation (the more complete of the two -- see build_graph()):
        # the full Markdown report is on disk at report_path, the short
        # plain-language summary is report_executive_summary.
        "report_path": slow_state.get("report_path"),
        "report_pdf_path": slow_state.get("report_pdf_path"),
        "report_status": slow_state.get("report_status"),
        "report_executive_summary": slow_state.get("report_executive_summary"),
    }

    _generate_and_send_report(summary)

    return summary
