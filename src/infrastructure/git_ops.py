"""
Git Operations Infrastructure
Parallel repository cloning with GitHub token injection using SandboxRunner.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from ..config.repo_registry import RepoConfig, RepoRegistry
from ..sandbox.runner import SandboxRunner, get_sandbox_runner

logger = logging.getLogger(__name__)


@dataclass
class CloneResult:
    """Result of a single repository clone operation."""

    repo_name: str
    repo_url: str
    target_path: Path
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: float
    error_message: str | None = None
    branch: str | None = None
    commit_hash: str | None = None

    @property
    def is_success(self) -> bool:
        return self.success and self.exit_code == 0


@dataclass
class CloneSummary:
    """Summary of batch clone operations."""

    total: int
    successful: int
    failed: int
    results: list[CloneResult] = field(default_factory=list)
    total_duration_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.successful / self.total


class GitOps:
    """
    Parallel Git operations with token injection.

    Uses SandboxRunner for deterministic subprocess execution with timeouts.
    Supports GitHub token injection for private repository access.
    """

    def __init__(
        self,
        sandbox_runner: SandboxRunner | None = None,
        default_branch: str = "main",
        default_depth: int = 1,
        github_token: str | None = None,
        max_concurrent: int = 5,
        clone_timeout: float = 300.0,
    ):
        self.sandbox_runner = sandbox_runner or get_sandbox_runner()
        self.default_branch = default_branch
        self.default_depth = default_depth
        self.github_token = github_token or os.environ.get("GITHUB_TOKEN")
        self.max_concurrent = max_concurrent
        self.clone_timeout = clone_timeout

    def _inject_token(self, url: str) -> str:
        """Inject GitHub token into HTTPS URL for authenticated cloning."""
        if not self.github_token:
            return url

        if url.startswith("https://github.com/"):
            return url.replace("https://github.com/", f"https://{self.github_token}@github.com/")

        return url

    def _build_clone_command(
        self,
        repo: RepoConfig,
        target_dir: Path,
    ) -> list[str]:
        """Build git clone command with appropriate options."""
        url = self._inject_token(repo.url)
        branch = repo.branch_or_default(self.default_branch)
        depth = repo.depth_or_default(self.default_depth)

        cmd = [
            "git",
            "clone",
            "--branch",
            branch,
            "--depth",
            str(depth),
            "--single-branch",
            url,
            str(target_dir),
        ]
        return cmd

    async def clone_repo(
        self,
        repo: RepoConfig,
        base_dir: Path,
    ) -> CloneResult:
        """Clone a single repository."""
        target_path = base_dir / repo.name.replace("/", "_")

        if target_path.exists():
            import shutil

            shutil.rmtree(target_path)

        target_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = self._build_clone_command(repo, target_path)

        import time

        start = time.perf_counter()
        result = await self.sandbox_runner.run(cmd, timeout=self.clone_timeout)
        duration_ms = (time.perf_counter() - start) * 1000

        commit_hash = None
        if result.success:
            try:
                hash_result = await self.sandbox_runner.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=str(target_path),
                    timeout=10.0,
                )
                if hash_result.success:
                    commit_hash = hash_result.stdout.strip()
            except Exception as e:
                # Log the error but don't fail the clone operation
                logger.warning(f"Failed to get commit hash for {repo.name}: {e}")

        return CloneResult(
            repo_name=repo.name,
            repo_url=repo.url,
            target_path=target_path,
            success=result.success,
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            duration_ms=duration_ms,
            error_message=result.error_message,
            branch=repo.branch_or_default(self.default_branch),
            commit_hash=commit_hash,
        )

    async def clone_repos_parallel(
        self,
        repos: list[RepoConfig],
        base_dir: Path,
        max_concurrent: int | None = None,
    ) -> CloneSummary:
        """Clone multiple repositories in parallel with semaphore limiting."""
        import asyncio

        max_concurrent = max_concurrent or self.max_concurrent
        semaphore = asyncio.Semaphore(max_concurrent)

        async def clone_one(repo: RepoConfig) -> CloneResult:
            async with semaphore:
                return await self.clone_repo(repo, base_dir)

        tasks = [clone_one(repo) for repo in repos]
        results: list[CloneResult | BaseException] = await asyncio.gather(
            *tasks, return_exceptions=True
        )

        clone_results: list[CloneResult] = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                clone_results.append(
                    CloneResult(
                        repo_name=repos[i].name,
                        repo_url=repos[i].url,
                        target_path=base_dir / repos[i].name.replace("/", "_"),
                        success=False,
                        exit_code=-1,
                        stdout="",
                        stderr=str(result),
                        duration_ms=0.0,
                        error_message=f"Exception: {result}",
                    )
                )
            else:
                clone_results.append(result)

        successful = sum(1 for r in clone_results if r.is_success)
        failed = len(clone_results) - successful
        total_duration = sum(r.duration_ms for r in clone_results)

        return CloneSummary(
            total=len(clone_results),
            successful=successful,
            failed=failed,
            results=clone_results,
            total_duration_ms=total_duration,
        )

    async def clone_from_registry(
        self,
        registry: RepoRegistry,
        base_dir: Path,
        include_github_only: bool = True,
        include_deprecated: bool = False,
        max_concurrent: int | None = None,
    ) -> CloneSummary:
        """Clone repositories from a RepoRegistry."""
        repos = registry.enabled_scannable_repos()

        if not include_github_only:
            github_only_names = {r.name for r in registry.repos_github_only}
            repos = [r for r in repos if r.name not in github_only_names]

        if include_deprecated:
            pass

        return await self.clone_repos_parallel(repos, base_dir, max_concurrent)

    def get_clone_url(self, repo: RepoConfig) -> str:
        """Get the clone URL with token injected (for display/logging)."""
        return self._inject_token(repo.url)

    @staticmethod
    def sanitize_repo_name(name: str) -> str:
        """Sanitize repository name for use as directory name."""
        return name.replace("/", "_").replace(":", "_")


def inject_github_token(url: str, token: str | None = None) -> str:
    """Standalone function to inject GitHub token into a URL."""
    token = token or os.environ.get("GITHUB_TOKEN")
    if not token:
        return url

    if url.startswith("https://github.com/"):
        return url.replace("https://github.com/", f"https://{token}@github.com/")

    return url


__all__ = [
    "CloneResult",
    "CloneSummary",
    "GitOps",
    "inject_github_token",
]
