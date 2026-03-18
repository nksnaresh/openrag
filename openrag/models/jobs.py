"""Job lifecycle data contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


@dataclass
class IngestMetadata:
    """Caller-supplied metadata attached to an ingestion job."""

    tenant_id: str
    namespace: str = "default"
    tags: list[str] = field(default_factory=list)
    custom: dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchOptions:
    """Options for batch ingestion of multiple documents."""

    metadata: IngestMetadata
    max_workers: int = 4
    recursive: bool = True
    force_reingest: bool = False
    supported_extensions: list[str] = field(default_factory=lambda: [
        ".pdf", ".docx", ".pptx", ".xlsx", ".doc", ".ppt", ".xls",
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".gif",
        ".py", ".js", ".ts", ".java", ".go", ".rs", ".sql", ".md",
        ".txt", ".html", ".htm", ".mp3", ".wav", ".mp4", ".mov",
    ])


@dataclass
class JobResult:
    """Result from a single document ingestion."""

    job_id: str
    status: JobStatus
    document_id: str | None = None
    block_count: int = 0
    reason: str | None = None
    """Human-readable explanation for skipped/failed jobs."""
    error: str | None = None


@dataclass
class BatchJobResult:
    """Aggregate result from a batch ingestion run."""

    total: int
    successful: list[JobResult] = field(default_factory=list)
    failed: list[JobResult] = field(default_factory=list)
    skipped: list[JobResult] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def success_rate(self) -> float:
        return len(self.successful) / self.total if self.total else 0.0
