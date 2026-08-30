"""
test_remediation_survives_the_graph.py -- Regression test for a bug found
during a manual audit: RemediationAction objects produced by
remediation_node were silently LOST between "remediation" and
"diagnostic_report" when running through the FULL compiled LangGraph
(orchestrator.build_graph().invoke(...)), even though:
  - RemediationAgent.process(alerts) worked fine called directly,
  - orchestrator.remediation_node(state) worked fine called directly
    (outside the graph).

Root cause: `_remediation_action_objects` was returned by remediation_node
but never DECLARED as a key in `MonitoringState` (the TypedDict schema
passed to StateGraph(...)) -- unlike its sibling `_alert_objects`, which
was declared and worked correctly. A key returned by a node but absent
from the schema is not tracked as a channel by the compiled graph, so it
never reaches the next node (diagnostic_report.py::report_node), which
silently received an empty list and wrote "None generated this cycle" to
every diagnostic report -- no exception, no warning, just wrong output.

This test exercises the REAL compiled graph end-to-end (same call shape as
run_scenario.py: two graph.invoke() calls, "batch" then "labeled_window",
same thread_id) on a real severe-drift scenario, and asserts that
RemediationAction objects for CRITICAL/PAGE alerts actually survive the
full alerting -> remediation -> diagnostic_report chain.
"""

from __future__ import annotations

import orchestrator
from alert_agent import AlertAgent, AlertDispatcher
from performance_drift_agent import TaskType
from real_scenario_runner import build_baseline_calculator, get_baseline_predictions, run_real_scenario


def test_remediation_actions_are_not_lost_through_the_compiled_graph():
    orchestrator._alert_agent = AlertAgent(dispatcher=AlertDispatcher())  # LOG-only, no real I/O

    bc, cfg = build_baseline_calculator()
    tracked = cfg["baseline"]["tracked_features"]
    y_true_b, y_pred_b, y_proba_b = get_baseline_predictions(bc, cfg)

    orchestrator.bootstrap_agents(
        model_id="test_remediation_survives_graph",
        task_type=TaskType.CLASSIFICATION,
        reference_features=bc.baseline_df[tracked],
        y_reference=y_true_b,
        y_true_baseline=y_true_b,
        y_pred_baseline=y_pred_b,
        y_proba_baseline=y_proba_b,
        critical_classes=[1],
        critical_recall_floor=0.5,
        use_calibration=True,
    )

    batch = run_real_scenario("severe_drift", bc, cfg, severity="severe")
    graph = orchestrator.build_graph()
    config = {"configurable": {"thread_id": "test_remediation_survives_graph"}}

    batch_id = f"{batch.name}_batch"
    orchestrator.stage_batch(batch_id, features_df=batch.features_df, y_pred=batch.y_pred)
    graph.invoke({"model_id": "test_remediation_survives_graph", "batch_id": batch_id, "trigger": "batch"}, config=config)
    orchestrator.clear_batch(batch_id)

    label_batch_id = f"{batch.name}_labels"
    orchestrator.stage_batch(
        label_batch_id, features_df=batch.features_df,
        y_true=batch.y_true, y_pred=batch.y_pred, y_proba=batch.y_proba,
    )
    slow_state = graph.invoke(
        {"model_id": "test_remediation_survives_graph", "batch_id": label_batch_id, "trigger": "labeled_window"},
        config=config,
    )
    orchestrator.clear_batch(label_batch_id)

    alerts = slow_state.get("_alert_objects", [])
    actions = slow_state.get("_remediation_action_objects", [])

    # severe_drift always raises at least one CRITICAL/PAGE alert with a
    # non-NONE remediation_type (data_pipeline_investigation/retrain/
    # escalate) -- if this ever becomes 0, the scenario itself changed and
    # this test needs revisiting, not the assertion below.
    assert alerts, "severe_drift produced no alerts at all -- scenario broken, not this test's concern"

    # The actual regression: actions must not silently vanish between
    # remediation_node and report_node inside the compiled graph.
    assert len(actions) == len(alerts), (
        f"{len(alerts)} alert(s) were raised but only {len(actions)} RemediationAction(s) "
        "survived through the compiled graph -- regression of the "
        "_remediation_action_objects/MonitoringState bug."
    )

    # And the diagnostic report (written to disk by report_node, same
    # cycle) must reflect the same actions, not an empty list.
    report_path = slow_state.get("report_path")
    assert report_path, "report_node did not run or did not set report_path"
    report_text = open(report_path, encoding="utf-8").read()
    assert "None generated this cycle" not in report_text
    assert f"## Remediation actions ({len(actions)})" in report_text
