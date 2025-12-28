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
        language: str = "en",
        enable_diarization: bool = True,
        include_summary: bool = True,
        workflow_name: str = "chronos_workflow",
        model: str = "gemini",
    ) -> str:
        """
        Submit a multi-step AI workflow for processing.

        Args:
            file_id: Plaud file/recording ID to process
            template_id: Custom ETL template ID (optional)
            language: Language code for transcription
            enable_diarization: Enable speaker diarization
            include_summary: Include AI summary task
            workflow_name: Name for the workflow
            model: LLM model to use (gemini, openai, claude)

        Returns:
            workflow_id for tracking
        """
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
        return workflow_id

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
        self, workflow_id: str, poll_interval: float = 5.0, timeout: float = 600.0
    ) -> WorkflowResult:
        """
        Wait for a workflow to complete (polling).

        Args:
            workflow_id: Workflow ID to monitor
            poll_interval: Seconds between status checks
            timeout: Maximum seconds to wait

        Returns:
            WorkflowResult when complete

        Raises:
            TimeoutError: If workflow doesn't complete in time
        """
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
        language: str = "en",
        wait_for_result: bool = True,
        timeout: float = 600.0,
    ) -> WorkflowResult:
        """
        Convenience method to fully process a recording.

        Submits a workflow with transcription and summary,
        optionally waiting for completion.

        Args:
            file_id: Plaud recording file ID
            language: Language code
            wait_for_result: If True, blocks until complete
            timeout: Max seconds to wait (if waiting)

        Returns:
            WorkflowResult with all outputs
        """
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


if __name__ == "__main__":
    # Quick test
    client = get_workflow_client()
    print("PlaudWorkflowClient initialized successfully")
