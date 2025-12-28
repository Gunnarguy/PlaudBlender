"""Streamlit component: AI Workflow Orchestration Panel

Full-featured workflow management UI:
- Submit AI workflows (transcription + ETL + summary)
- Monitor workflow progress in real-time
- View and manage workflow results
- Configure workflow templates
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
import time

import streamlit as st

from src.plaud_workflow import (
    PlaudWorkflowClient,
    WorkflowStatus,
    WorkflowResult,
    TaskType,
)
from src.plaud_client import PlaudClient
from src.database import SessionLocal, init_db
from src.database.models import ChronosRecording as ChronosRecordingDB


def _status_emoji(status: WorkflowStatus) -> str:
    """Get status emoji."""
    return {
        WorkflowStatus.PENDING: "⏳",
        WorkflowStatus.PROCESSING: "⚙️",
        WorkflowStatus.SUCCESS: "✅",
        WorkflowStatus.FAILED: "❌",
        WorkflowStatus.CANCELLED: "🚫",
    }.get(status, "❓")


def _task_type_emoji(task_type: str) -> str:
    """Get task type emoji."""
    return {
        "AUDIO_TRANSCRIBE": "🎙️",
        "AI_ETL": "🔄",
        "AI_SUMMARY": "📝",
    }.get(task_type, "📋")


def render_workflow_submit() -> None:
    """Render workflow submission form."""
    st.subheader("🚀 Submit New Workflow")
    st.caption(
        "Process recordings through Plaud's AI pipeline: transcription, "
        "structured extraction, and summarization."
    )

    # Get available recordings
    init_db()
    session = SessionLocal()
    try:
        recordings = (
            session.query(ChronosRecordingDB)
            .filter(ChronosRecordingDB.processing_status.in_(["pending", "completed"]))
            .order_by(ChronosRecordingDB.created_at.desc())
            .limit(100)
            .all()
        )

        if not recordings:
            st.info(
                "No recordings available. Use Controls → Ingest to fetch recordings first."
            )
            return

        # Recording selector
        recording_options = {
            f"{r.title or r.recording_id} ({r.created_at.strftime('%Y-%m-%d')})": r.recording_id
            for r in recordings
        }

        col1, col2 = st.columns([2, 1])
        with col1:
            selected_label = st.selectbox(
                "Select recording to process",
                options=list(recording_options.keys()),
                key="workflow_recording",
            )
        with col2:
            st.write("")  # Spacer
            st.write("")
            manual_id = st.text_input(
                "Or enter recording ID",
                placeholder="plaud_file_id",
                key="workflow_manual_id",
            )

        file_id = (
            manual_id.strip()
            if manual_id.strip()
            else recording_options.get(selected_label)
        )

        st.divider()

        # Workflow configuration
        st.markdown("**Workflow Configuration**")

        config_cols = st.columns(3)
        with config_cols[0]:
            language = st.selectbox(
                "Language",
                options=["en", "zh", "es", "fr", "de", "ja", "ko", "pt"],
                index=0,
                key="workflow_language",
                help="Primary language of the recording",
            )
        with config_cols[1]:
            enable_diarization = st.checkbox(
                "Speaker diarization",
                value=True,
                key="workflow_diarization",
                help="Identify and label different speakers",
            )
        with config_cols[2]:
            include_summary = st.checkbox(
                "AI Summary",
                value=True,
                key="workflow_summary",
                help="Generate an AI-powered summary",
            )

        # Advanced options
        with st.expander("Advanced Options", expanded=False):
            adv_cols = st.columns(2)
            with adv_cols[0]:
                template_id = st.text_input(
                    "ETL Template ID (optional)",
                    placeholder="tpl_custom_extraction",
                    key="workflow_template",
                    help="Custom template for structured data extraction",
                )
                model = st.selectbox(
                    "LLM Model",
                    options=["gemini", "openai", "claude"],
                    index=0,
                    key="workflow_model",
                )
            with adv_cols[1]:
                workflow_name = st.text_input(
                    "Workflow name",
                    value="chronos_workflow",
                    key="workflow_name",
                )
                wait_for_result = st.checkbox(
                    "Wait for completion",
                    value=False,
                    key="workflow_wait",
                    help="Block until workflow finishes (can take several minutes)",
                )

        # Submit button
        st.divider()
        submit_cols = st.columns([2, 1, 1])
        with submit_cols[0]:
            if st.button(
                "🚀 Submit Workflow",
                type="primary",
                use_container_width=True,
                disabled=not file_id,
            ):
                _submit_workflow(
                    file_id=file_id,
                    language=language,
                    enable_diarization=enable_diarization,
                    include_summary=include_summary,
                    template_id=template_id.strip() if template_id else None,
                    model=model,
                    workflow_name=workflow_name,
                    wait_for_result=wait_for_result,
                )

    finally:
        session.close()


def _submit_workflow(
    file_id: str,
    language: str,
    enable_diarization: bool,
    include_summary: bool,
    template_id: Optional[str],
    model: str,
    workflow_name: str,
    wait_for_result: bool,
) -> None:
    """Submit a workflow and optionally wait for results."""
    try:
        client = PlaudWorkflowClient()

        with st.spinner("Submitting workflow..."):
            workflow_id = client.submit_workflow(
                file_id=file_id,
                template_id=template_id,
                language=language,
                enable_diarization=enable_diarization,
                include_summary=include_summary,
                workflow_name=workflow_name,
                model=model,
            )

        st.success(f"✅ Workflow submitted: `{workflow_id}`")

        # Store in session for tracking
        if "active_workflows" not in st.session_state:
            st.session_state.active_workflows = {}
        st.session_state.active_workflows[workflow_id] = {
            "file_id": file_id,
            "submitted_at": datetime.now(),
            "status": "PENDING",
        }

        if wait_for_result:
            with st.spinner("⏳ Waiting for workflow to complete..."):
                result = client.wait_for_workflow(workflow_id, timeout=600)
                _display_workflow_result(result)
        else:
            st.info(
                f"Workflow is processing. Check the **Monitor** tab for status updates."
            )

    except Exception as e:
        st.error(f"Failed to submit workflow: {e}")


def _display_workflow_result(result: WorkflowResult) -> None:
    """Display a workflow result."""
    status_emoji = _status_emoji(result.status)

    if result.status == WorkflowStatus.SUCCESS:
        st.success(
            f"{status_emoji} Workflow completed: "
            f"{result.tasks_completed}/{result.tasks_total} tasks"
        )
    elif result.status == WorkflowStatus.FAILED:
        st.error(
            f"{status_emoji} Workflow failed: {result.error_message or 'Unknown error'}"
        )
    else:
        st.warning(f"{status_emoji} Workflow status: {result.status.value}")

    # Show results in tabs
    if result.transcript or result.summary or result.extracted_data:
        result_tabs = st.tabs(
            ["📄 Transcript", "📝 Summary", "📊 Extracted Data", "🔧 Raw"]
        )

        with result_tabs[0]:
            if result.transcript:
                st.text_area(
                    "Transcript",
                    value=result.transcript,
                    height=300,
                    key=f"transcript_{result.workflow_id}",
                )
            else:
                st.info("No transcript available.")

        with result_tabs[1]:
            if result.summary:
                st.markdown(result.summary)
            else:
                st.info("No summary available.")

        with result_tabs[2]:
            if result.extracted_data:
                st.json(result.extracted_data)
            else:
                st.info("No extracted data available.")

        with result_tabs[3]:
            st.json(result.raw_response)


def render_workflow_monitor() -> None:
    """Render workflow monitoring dashboard."""
    st.subheader("📊 Workflow Monitor")
    st.caption("Track active workflows and view their status.")

    # Get active workflows from session
    active_workflows = st.session_state.get("active_workflows", {})

    if not active_workflows:
        st.info(
            "No active workflows. Submit a workflow from the **Submit** tab to get started."
        )

        # Manual workflow ID lookup
        with st.expander("🔍 Look up workflow by ID"):
            lookup_id = st.text_input(
                "Workflow ID",
                placeholder="workflow_id_here",
                key="workflow_lookup",
            )
            if st.button("Look up", disabled=not lookup_id.strip()):
                _lookup_workflow(lookup_id.strip())
        return

    # Refresh button
    if st.button("🔄 Refresh All", use_container_width=False):
        _refresh_all_workflows()

    # Display active workflows
    for workflow_id, info in active_workflows.items():
        with st.expander(
            f"⚙️ {workflow_id[:20]}... · {info.get('status', 'UNKNOWN')}",
            expanded=info.get("status") == "PROCESSING",
        ):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**File ID:** `{info.get('file_id', 'N/A')}`")
            with col2:
                submitted = info.get("submitted_at")
                if submitted:
                    st.write(f"**Submitted:** {submitted.strftime('%H:%M:%S')}")
            with col3:
                st.write(f"**Status:** {info.get('status', 'UNKNOWN')}")

            # Action buttons
            btn_cols = st.columns(3)
            with btn_cols[0]:
                if st.button(
                    "🔄 Refresh",
                    key=f"refresh_{workflow_id}",
                    use_container_width=True,
                ):
                    _refresh_workflow(workflow_id)
            with btn_cols[1]:
                if st.button(
                    "📥 Get Results",
                    key=f"results_{workflow_id}",
                    use_container_width=True,
                    disabled=info.get("status") != "SUCCESS",
                ):
                    _get_workflow_results(workflow_id)
            with btn_cols[2]:
                if st.button(
                    "🗑️ Remove",
                    key=f"remove_{workflow_id}",
                    use_container_width=True,
                ):
                    del st.session_state.active_workflows[workflow_id]
                    st.rerun()


def _refresh_workflow(workflow_id: str) -> None:
    """Refresh a single workflow's status."""
    try:
        client = PlaudWorkflowClient()
        status = client.get_workflow_status(workflow_id)

        if "active_workflows" in st.session_state:
            if workflow_id in st.session_state.active_workflows:
                st.session_state.active_workflows[workflow_id]["status"] = status.get(
                    "status", "UNKNOWN"
                )
                st.session_state.active_workflows[workflow_id][
                    "progress"
                ] = f"{status.get('completed_tasks', 0)}/{status.get('total_tasks', 0)}"

        st.success(f"Status: {status.get('status')}")
        st.json(status)
    except Exception as e:
        st.error(f"Failed to refresh: {e}")


def _refresh_all_workflows() -> None:
    """Refresh all active workflows."""
    active = st.session_state.get("active_workflows", {})
    if not active:
        return

    try:
        client = PlaudWorkflowClient()
        for workflow_id in active.keys():
            try:
                status = client.get_workflow_status(workflow_id)
                active[workflow_id]["status"] = status.get("status", "UNKNOWN")
            except Exception as e:
                active[workflow_id]["status"] = f"ERROR: {e}"

        st.session_state.active_workflows = active
        st.rerun()
    except Exception as e:
        st.error(f"Failed to refresh workflows: {e}")


def _get_workflow_results(workflow_id: str) -> None:
    """Get and display workflow results."""
    try:
        client = PlaudWorkflowClient()
        result = client.get_workflow_results(workflow_id)
        _display_workflow_result(result)
    except Exception as e:
        st.error(f"Failed to get results: {e}")


def _lookup_workflow(workflow_id: str) -> None:
    """Look up a workflow by ID."""
    try:
        client = PlaudWorkflowClient()

        with st.spinner("Looking up workflow..."):
            status = client.get_workflow_status(workflow_id)

        st.success(f"Found workflow: {status.get('status')}")

        # Add to active workflows
        if "active_workflows" not in st.session_state:
            st.session_state.active_workflows = {}

        st.session_state.active_workflows[workflow_id] = {
            "file_id": "unknown",
            "submitted_at": datetime.now(),
            "status": status.get("status", "UNKNOWN"),
        }

        st.json(status)

    except Exception as e:
        st.error(f"Workflow not found: {e}")


def render_workflow_history() -> None:
    """Render workflow history (from database)."""
    st.subheader("📜 Workflow History")
    st.caption("View past workflow runs stored in the database.")

    # This would query a workflows table if we had one
    st.info(
        "Workflow history is stored in session memory. "
        "For persistent history, workflows are linked to recordings in the database."
    )

    # Show session history
    active = st.session_state.get("active_workflows", {})
    completed = [w for w in active.items() if w[1].get("status") == "SUCCESS"]
    failed = [w for w in active.items() if w[1].get("status") == "FAILED"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Workflows", len(active))
    with col2:
        st.metric("Completed", len(completed))
    with col3:
        st.metric("Failed", len(failed))


def render_workflow_panel() -> None:
    """Main workflow panel component."""
    st.header("⚡ AI Workflow Orchestration")
    st.caption(
        "Submit and monitor AI processing workflows. Each workflow can include "
        "transcription, structured extraction, and AI summarization."
    )

    try:
        # Quick health check
        client = PlaudWorkflowClient()
    except Exception as e:
        st.error(f"Failed to initialize workflow client: {e}")
        st.info(
            "Make sure your Plaud OAuth is configured (run `python plaud_setup.py`)."
        )
        return

    # Tab layout
    tab1, tab2, tab3 = st.tabs(["🚀 Submit", "📊 Monitor", "📜 History"])

    with tab1:
        render_workflow_submit()

    with tab2:
        render_workflow_monitor()

    with tab3:
        render_workflow_history()


if __name__ == "__main__":
    st.set_page_config(page_title="Workflow Orchestration", layout="wide")
    render_workflow_panel()
