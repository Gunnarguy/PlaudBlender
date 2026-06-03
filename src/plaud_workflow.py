"""
Plaud Workflow API Client - AI Workflow orchestration for transcription and extraction.

Based on Plaud API documentation:
- Submit multi-step AI workflows (transcription + ETL)
- Monitor workflow progress
- Retrieve structured results
- Webhook integration for async processing

Usage:
    workflow_client = PlaudWorkflowClient()

    # Submit a workflow
    workflow_id = workflow_client.submit_workflow(
        file_id="audio_file_123",
        template_id="tpl_healthcare"
    )

    # Check status
    status = workflow_client.get_workflow_status(workflow_id)

    # Get results when complete
    results = workflow_client.get_workflow_results(workflow_id)
"""

import time
import json
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

import requests

from .plaud_oauth import PlaudOAuthClient
from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Plaud API base URL for workflows
PLAUD_API_BASE = "https://api.plaud.ai/api"


class WorkflowStatus(Enum):
    """Workflow execution status."""

    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class TaskType(Enum):
    """Available workflow task types."""

    AUDIO_TRANSCRIBE = "AUDIO_TRANSCRIBE"
    AI_ETL = "AI_ETL"
    AI_SUMMARY = "AI_SUMMARY"


@dataclass
class WorkflowTask:
    """Definition of a single workflow task."""

    task_type: TaskType
    task_params: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"task_type": self.task_type.value, "task_params": self.task_params}


@dataclass
class WorkflowResult:
    """Result from a completed workflow."""

    workflow_id: str
    status: WorkflowStatus
    tasks_completed: int
    tasks_total: int
    transcript: Optional[str] = None
    extracted_data: Optional[Dict[str, Any]] = None
    summary: Optional[str] = None
    error_message: Optional[str] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)


class PlaudWorkflowClient:
    """
    Client for Plaud AI Workflow API.

    Enables orchestration of multi-step AI processing:
    - Audio transcription with speaker diarization
    - Structured data extraction (ETL) with custom templates
    - AI-powered summarization
    """

    def __init__(self, oauth_client: Optional[PlaudOAuthClient] = None):
        """
        Initialize workflow client.

        Args:
            oauth_client: PlaudOAuthClient instance (auto-created if not provided)
        """
        self.oauth = oauth_client or PlaudOAuthClient()

    def _get_headers(self) -> Dict[str, str]:
        """Get authorization headers for API requests."""
        return {
            "Authorization": f"Bearer {self.oauth.get_access_token()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make authenticated API request with retry on auth failure."""
        url = f"{PLAUD_API_BASE}{endpoint}"
        headers = self._get_headers()

        response = requests.request(method, url, headers=headers, **kwargs)

        # Handle token refresh on auth failure
        if response.status_code in (401, 422):
            logger.info("Token rejected, refreshing...")
            try:
                self.oauth.refresh_access_token()
            except Exception as exc:
                logger.error(f"Refresh failed: {exc}")
                raise
            headers = self._get_headers()
            response = requests.request(method, url, headers=headers, **kwargs)

        response.raise_for_status()
        return response.json()

    def submit_workflow(
        self,
        file_id: str,
        template_id: Optional[str] = None,
        language: Optional[str] = None,
        enable_diarization: Optional[bool] = None,
        include_summary: bool = True,
        workflow_name: str = "chronos_workflow",
        model: str = "openai",
    ) -> str:
        """
        Submit a multi-step AI workflow for processing.

        Args:
            file_id: Plaud file/recording ID to process
            template_id: Custom ETL template ID (optional)
            language: Language code for transcription (default from config)
            enable_diarization: Enable speaker diarization (default from config)
            include_summary: Include AI summary task
            workflow_name: Name for the workflow
            model: LLM model to use (gemini, openai, claude)

        Returns:
            workflow_id for tracking
        """
        if language is None:
            language = settings.plaud_default_language
        if enable_diarization is None:
            enable_diarization = settings.plaud_enable_diarization
        tasks: List[Dict[str, Any]] = []

        # Task 1: Audio Transcription
        tasks.append(
            {
                "task_type": TaskType.AUDIO_TRANSCRIBE.value,
                "task_params": {
                    "file_id": file_id,
                    "language": language,
                    "diarization": enable_diarization,
                },
            }
        )

        # Task 2: AI ETL (if template provided)
        if template_id:
            tasks.append(
                {
                    "task_type": TaskType.AI_ETL.value,
                    "task_params": {
                        "template_id": template_id,
                        "language": language,
                        "model": model,
                    },
                }
            )

        # Task 3: AI Summary (optional)
        if include_summary:
            tasks.append(
                {
                    "task_type": TaskType.AI_SUMMARY.value,
                    "task_params": {"language": language, "model": model},
                }
            )

        workflow_data = {
            "workflows": tasks,
            "metadata": {"workflow_name": workflow_name},
        }

        logger.info(f"Submitting workflow for file {file_id} with {len(tasks)} tasks")

        response = self._request("POST", "/workflow/submit", json=workflow_data)
        workflow_id = response.get("id") or response.get("workflow_id")

        logger.info(f"✅ Workflow submitted: {workflow_id}")
        return workflow_id  # type: ignore[return-value]

    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """
        Check the status of a workflow.

        Args:
            workflow_id: Workflow ID to check

        Returns:
            Status info including progress
        """
        response = self._request("GET", f"/workflow/{workflow_id}/status")
        return {
            "status": response.get("status"),
            "completed_tasks": response.get("completed_tasks", 0),
            "total_tasks": response.get("total_tasks", 0),
            "current_task": response.get("current_task"),
            "error": response.get("error"),
        }

    def get_workflow_results(self, workflow_id: str) -> WorkflowResult:
        """
        Get the results of a completed workflow.

        Args:
            workflow_id: Workflow ID to retrieve

        Returns:
            WorkflowResult with transcript, extracted data, and summary
        """
        response = self._request("GET", f"/workflow/{workflow_id}/result")

        # Parse results
        status = WorkflowStatus(response.get("status", "FAILED"))

        result = WorkflowResult(
            workflow_id=workflow_id,
            status=status,
            tasks_completed=response.get("completed_tasks", 0),
            tasks_total=response.get("total_tasks", 0),
            raw_response=response,
        )

        # Extract task outputs
        task_results = response.get("task_results", [])
        for task_result in task_results:
            task_type = task_result.get("task_type")
            output = task_result.get("output", {})

            if task_type == TaskType.AUDIO_TRANSCRIBE.value:
                result.transcript = output.get("transcript") or output.get("text")
            elif task_type == TaskType.AI_ETL.value:
                result.extracted_data = output.get("extracted_data") or output
            elif task_type == TaskType.AI_SUMMARY.value:
                result.summary = output.get("summary") or output.get("text")

        return result

    def wait_for_workflow(
        self,
        workflow_id: str,
        poll_interval: float = 5.0,
        timeout: Optional[float] = None,
    ) -> WorkflowResult:
        """
        Wait for a workflow to complete (polling).

        Args:
            workflow_id: Workflow ID to monitor
            poll_interval: Seconds between status checks
            timeout: Maximum seconds to wait (default from config)

        Returns:
            WorkflowResult when complete

        Raises:
            TimeoutError: If workflow doesn't complete in time
        """
        if timeout is None:
            timeout = float(settings.plaud_workflow_timeout)
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Workflow {workflow_id} timed out after {timeout}s")

            status_info = self.get_workflow_status(workflow_id)
            status = status_info.get("status")

            logger.info(
                f"Workflow {workflow_id}: {status} "
                f"({status_info['completed_tasks']}/{status_info['total_tasks']} tasks)"
            )

            if status == WorkflowStatus.SUCCESS.value:
                return self.get_workflow_results(workflow_id)
            elif status in (
                WorkflowStatus.FAILED.value,
                WorkflowStatus.CANCELLED.value,
            ):
                result = WorkflowResult(
                    workflow_id=workflow_id,
                    status=WorkflowStatus(status),
                    tasks_completed=status_info.get("completed_tasks", 0),
                    tasks_total=status_info.get("total_tasks", 0),
                    error_message=status_info.get("error"),
                )
                return result

            time.sleep(poll_interval)

    def process_recording(
        self,
        file_id: str,
        language: Optional[str] = None,
        wait_for_result: bool = True,
        timeout: Optional[float] = None,
    ) -> WorkflowResult:
        """
        Convenience method to fully process a recording.

        Submits a workflow with transcription and summary,
        optionally waiting for completion.

        Args:
            file_id: Plaud recording file ID
            language: Language code (default from config)
            wait_for_result: If True, blocks until complete
            timeout: Max seconds to wait (default from config)

        Returns:
            WorkflowResult with all outputs
        """
        if language is None:
            language = settings.plaud_default_language
        if timeout is None:
            timeout = float(settings.plaud_workflow_timeout)
        workflow_id = self.submit_workflow(
            file_id=file_id, language=language, include_summary=True
        )

        if wait_for_result:
            return self.wait_for_workflow(workflow_id, timeout=timeout)
        else:
            # Return pending result
            return WorkflowResult(
                workflow_id=workflow_id,
                status=WorkflowStatus.PENDING,
                tasks_completed=0,
                tasks_total=2,  # transcription + summary
            )


def get_workflow_client() -> PlaudWorkflowClient:
    """Get a PlaudWorkflowClient instance."""
    return PlaudWorkflowClient()


# ═══════════════════════════════════════════════════════════════════════════════
# AI Summary Templates & Batch Operations
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SummaryTemplate:
    """Pre-configured AI Summary template."""

    id: str
    name: str
    description: str
    prompt: Optional[str] = None
    model: str = "openai"

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "model": self.model,
        }
        if self.prompt:
            d["prompt"] = self.prompt
        return d


# Built-in templates for common recording types
BUILTIN_TEMPLATES: List[SummaryTemplate] = [
    SummaryTemplate(
        id="general",
        name="General Summary",
        description="Concise overview of recording content, key points, and action items",
        prompt=(
            "Provide a concise summary of this recording. Include:\n"
            "1. Main topic(s) discussed\n"
            "2. Key points and takeaways (bullet list)\n"
            "3. Any action items or follow-ups mentioned\n"
            "4. Overall tone and context\n"
            "Keep the summary clear and scannable. Use 3-5 sentences for the overview, "
            "then bullet points for details."
        ),
    ),
    SummaryTemplate(
        id="meeting",
        name="Meeting Notes",
        description="Meeting attendees, agenda items, decisions made, and action items with owners",
        prompt=(
            "Extract structured meeting notes from this recording:\n"
            "## Attendees\n"
            "List all participants/speakers identified.\n\n"
            "## Agenda Items\n"
            "List each topic discussed, in order.\n\n"
            "## Key Decisions\n"
            "List every decision made, with context.\n\n"
            "## Action Items\n"
            "For each action item: what needs to be done, who owns it, "
            "and any deadline mentioned.\n\n"
            "## Open Questions\n"
            "List unresolved questions or items deferred to later."
        ),
    ),
    SummaryTemplate(
        id="brainstorm",
        name="Brainstorm / Ideas",
        description="Capture all ideas discussed, group by theme, highlight most promising",
        prompt=(
            "This is a brainstorming or ideation session. Extract:\n"
            "## Ideas Generated\n"
            "List EVERY idea mentioned, no matter how small or speculative. "
            "Group related ideas together under theme headings.\n\n"
            "## Most Promising Ideas\n"
            "Identify the 2-3 ideas that received the most discussion, "
            "enthusiasm, or follow-up questions.\n\n"
            "## Constraints & Concerns\n"
            "Note any limitations, blockers, or concerns raised.\n\n"
            "## Next Steps\n"
            "Any concrete actions to explore ideas further."
        ),
    ),
    SummaryTemplate(
        id="daily_log",
        name="Daily Log",
        description="Stream-of-consciousness daily recording → structured timeline with key activities",
        prompt=(
            "This is a stream-of-consciousness daily recording. Transform it into a structured log:\n"
            "## Timeline\n"
            "Create a chronological timeline of activities, tasks, and events "
            "mentioned. Use approximate timestamps where possible.\n\n"
            "## Key Activities\n"
            "Summarize the main things accomplished or worked on.\n\n"
            "## Thoughts & Reflections\n"
            "Capture any reflective moments, plans, or ideas for the future.\n\n"
            "## People & Interactions\n"
            "Note any people mentioned or conversations referenced.\n\n"
            "## Mood & Energy\n"
            "Brief note on overall energy level and mood throughout the recording."
        ),
    ),
    SummaryTemplate(
        id="interview",
        name="Interview / Conversation",
        description="Q&A format, key statements from each speaker, notable quotes",
        prompt=(
            "This is an interview or structured conversation. Extract:\n"
            "## Participants\n"
            "Identify each speaker and their role (interviewer, interviewee, etc.).\n\n"
            "## Q&A Summary\n"
            "For each major question asked, provide:\n"
            "- The question (paraphrased)\n"
            "- Key points from the answer\n"
            "- Any notable direct quotes\n\n"
            "## Key Takeaways\n"
            "The most important insights or information shared.\n\n"
            "## Notable Quotes\n"
            "Direct quotes that are particularly interesting, insightful, or important.\n\n"
            "## Follow-Up Items\n"
            "Anything that needs follow-up or was left unresolved."
        ),
    ),
]


def get_builtin_templates() -> List[SummaryTemplate]:
    """Return all built-in summary templates."""
    return list(BUILTIN_TEMPLATES)


def get_template_by_id(template_id: str) -> Optional[SummaryTemplate]:
    """Look up a built-in template by ID."""
    for t in BUILTIN_TEMPLATES:
        if t.id == template_id:
            return t
    return None


class PlaudWorkflowManager:
    """High-level workflow orchestration over PlaudWorkflowClient.

    Adds:
    - Template-aware submission
    - Batch summary generation
    - Active workflow tracking
    - Upload → workflow chaining
    """

    def __init__(self, workflow_client: Optional[PlaudWorkflowClient] = None):
        self.client = workflow_client or PlaudWorkflowClient()
        self._active_workflows: Dict[str, Dict[str, Any]] = {}  # workflow_id → metadata

    @property
    def active_workflows(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._active_workflows)

    def submit_summary(
        self,
        file_id: str,
        template: Optional[SummaryTemplate] = None,
        language: Optional[str] = None,
        model: str = "openai",
    ) -> str:
        """Submit an AI Summary workflow for a recording.

        Args:
            file_id: Plaud file ID
            template: Optional SummaryTemplate to use
            language: Language code
            model: LLM provider (gemini/openai/claude)

        Returns:
            workflow_id
        """
        template_id = None
        if template and template.id not in ("general",):
            # Only pass template_id for non-default templates that need ETL
            template_id = template.id if template.prompt else None

        workflow_id = self.client.submit_workflow(
            file_id=file_id,
            template_id=template_id,
            language=language,
            include_summary=True,
            workflow_name=f"summary_{file_id[:8]}",
            model=model,
        )

        self._active_workflows[workflow_id] = {
            "file_id": file_id,
            "template": template.to_dict() if template else None,
            "status": "PENDING",
            "submitted_at": time.time(),
        }
        return workflow_id

    def submit_full_pipeline(
        self,
        file_id: str,
        template_id: str,
        language: Optional[str] = None,
        model: str = "openai",
    ) -> str:
        """Submit full AUDIO_TRANSCRIBE → AI_ETL → AI_SUMMARY workflow.

        Args:
            file_id: Plaud file ID
            template_id: ETL template ID for structured extraction
            language: Language code
            model: LLM provider

        Returns:
            workflow_id
        """
        workflow_id = self.client.submit_workflow(
            file_id=file_id,
            template_id=template_id,
            language=language,
            include_summary=True,
            workflow_name=f"full_pipeline_{file_id[:8]}",
            model=model,
        )

        self._active_workflows[workflow_id] = {
            "file_id": file_id,
            "template_id": template_id,
            "type": "full_pipeline",
            "status": "PENDING",
            "submitted_at": time.time(),
        }
        return workflow_id

    def poll_active(self) -> Dict[str, Dict[str, Any]]:
        """Poll all active workflows and return current statuses.

        Returns:
            Dict mapping workflow_id → {status, completed_tasks, total_tasks, ...}
        """
        results = {}
        completed_ids = []

        for wf_id, meta in self._active_workflows.items():
            try:
                status_info = self.client.get_workflow_status(wf_id)
                status = status_info.get("status", "PENDING")
                meta["status"] = status
                meta["completed_tasks"] = status_info.get("completed_tasks", 0)
                meta["total_tasks"] = status_info.get("total_tasks", 0)
                meta["current_task"] = status_info.get("current_task")
                meta["error"] = status_info.get("error")

                results[wf_id] = dict(meta)

                if status in ("SUCCESS", "FAILED", "CANCELLED"):
                    completed_ids.append(wf_id)

                    if status == "SUCCESS":
                        try:
                            wf_result = self.client.get_workflow_results(wf_id)
                            meta["result"] = {
                                "summary": wf_result.summary,
                                "transcript": wf_result.transcript,
                                "extracted_data": wf_result.extracted_data,
                            }
                            results[wf_id] = dict(meta)
                        except Exception as e:
                            logger.warning(f"Failed to get results for {wf_id}: {e}")

            except Exception as e:
                logger.error(f"Failed to poll workflow {wf_id}: {e}")
                meta["error"] = str(e)
                results[wf_id] = dict(meta)

        return results

    def upload_and_process(
        self,
        plaud_client,
        file_path: str,
        name: Optional[str] = None,
        template: Optional[SummaryTemplate] = None,
        language: Optional[str] = None,
        model: str = "openai",
    ) -> Dict[str, Any]:
        """Upload a local file to Plaud cloud, then submit AI workflow.

        This chains: local file → cloud upload → AI workflow submission.

        Args:
            plaud_client: PlaudClient instance with upload capability
            file_path: Path to local audio file
            name: Display name for the recording
            template: Optional summary template
            language: Language code
            model: LLM provider

        Returns:
            Dict with file_id, workflow_id, and upload metadata
        """
        # Step 1: Upload
        upload_result = plaud_client.upload_file(file_path, name=name)
        file_id = upload_result.get("id") or upload_result.get("file_id")

        if not file_id:
            raise RuntimeError(f"Upload returned no file_id: {upload_result}")

        logger.info(f"📤 Uploaded {file_path} → {file_id}")

        # Step 2: Submit workflow
        workflow_id = self.submit_summary(
            file_id=file_id,
            template=template,
            language=language,
            model=model,
        )

        logger.info(f"🔄 Workflow submitted for {file_id}: {workflow_id}")

        return {
            "file_id": file_id,
            "workflow_id": workflow_id,
            "upload": upload_result,
            "template": template.to_dict() if template else None,
        }


if __name__ == "__main__":
    # Quick test
    client = get_workflow_client()
    print("PlaudWorkflowClient initialized successfully")
