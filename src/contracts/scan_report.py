"""
Scan Report Contracts
Structured data models for NIM usage scan results.
Compatible with nim-usage-scanner output format.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from enum import StrEnum
else:
    try:
        from enum import StrEnum
    except ImportError:
        # Python < 3.11 fallback
        class StrEnum(str, Enum):  # noqa: UP042
            pass


from pydantic import BaseModel, Field


class SourceType(StrEnum):
    """Source type of the NIM match."""

    LOCAL = "local"
    HOSTED = "hosted"


class LocalNimMatch(BaseModel):
    """A NIM match found in local source code."""

    file: str = Field(..., description="Path to the file containing the match")
    line: int = Field(..., ge=1, description="Line number of the match")
    column: int = Field(..., ge=1, description="Column number of the match")
    nim_name: str = Field(..., description="Name of the NIM (e.g., nvidia/llama-3.1-70b-instruct)")
    context: str = Field(..., description="Surrounding code context")
    match_type: str = Field(..., description="Type of match: 'image', 'model', 'api', 'env'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")


class HostedNimMatch(BaseModel):
    """A NIM match found in hosted/deployed environment."""

    deployment_id: str = Field(..., description="Unique deployment identifier")
    namespace: str = Field(..., description="Kubernetes namespace or equivalent")
    service_name: str = Field(..., description="Service name")
    nim_name: str = Field(..., description="Name of the NIM")
    image: str = Field(..., description="Container image reference")
    replicas: int = Field(..., ge=0, description="Number of replicas")
    status: str = Field(..., description="Deployment status")
    last_updated: datetime = Field(..., description="Last update timestamp")


class NimFindings(BaseModel):
    """Collection of NIM findings from a scan."""

    local_matches: list[LocalNimMatch] = Field(default_factory=list)
    hosted_matches: list[HostedNimMatch] = Field(default_factory=list)

    def total_matches(self) -> int:
        return len(self.local_matches) + len(self.hosted_matches)

    def local_count(self) -> int:
        return len(self.local_matches)

    def hosted_count(self) -> int:
        return len(self.hosted_matches)

    def unique_nim_names(self) -> set[str]:
        names: set[str] = set()
        for m in self.local_matches:
            names.add(m.nim_name)
        for m in self.hosted_matches:  # type: ignore[assignment]
            names.add(m.nim_name)
        return names


class Summary(BaseModel):
    """Summary statistics for a scan report."""

    repositories_scanned: int = 0
    files_scanned: int = 0
    total_local_matches: int = 0
    total_hosted_matches: int = 0
    unique_nims_found: int = 0
    scan_duration_seconds: float = 0.0
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ScanReport(BaseModel):
    """Complete scan report with findings and metadata."""

    version: str = "1.0"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    scanner_version: str = "ai-infrastructure-chassis"
    source_type: SourceType = SourceType.LOCAL
    repository: str | None = None
    branch: str | None = None
    commit: str | None = None
    findings: NimFindings = Field(default_factory=NimFindings)
    summary: Summary = Field(default_factory=Summary)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScanReport:
        """Create ScanReport from dictionary."""
        return cls(**data)

    def merge(self, other: ScanReport) -> ScanReport:
        """Merge another scan report into this one."""
        merged = ScanReport(
            version=self.version,
            timestamp=self.timestamp,
            scanner_version=self.scanner_version,
            source_type=self.source_type,
            repository=self.repository,
            branch=self.branch,
            commit=self.commit,
            findings=NimFindings(
                local_matches=[*self.findings.local_matches, *other.findings.local_matches],
                hosted_matches=[*self.findings.hosted_matches, *other.findings.hosted_matches],
            ),
            summary=Summary(
                repositories_scanned=self.summary.repositories_scanned
                + other.summary.repositories_scanned,
                files_scanned=self.summary.files_scanned + other.summary.files_scanned,
                total_local_matches=self.summary.total_local_matches
                + other.summary.total_local_matches,
                total_hosted_matches=self.summary.total_hosted_matches
                + other.summary.total_hosted_matches,
                unique_nims_found=len(
                    self.findings.unique_nim_names() | other.findings.unique_nim_names()
                ),
                scan_duration_seconds=self.summary.scan_duration_seconds
                + other.summary.scan_duration_seconds,
                errors=[*self.summary.errors, *other.summary.errors],
                warnings=[*self.summary.warnings, *other.summary.warnings],
            ),
        )
        return merged


__all__ = [
    "SourceType",
    "LocalNimMatch",
    "HostedNimMatch",
    "NimFindings",
    "Summary",
    "ScanReport",
]
