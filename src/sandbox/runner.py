"""
Deterministic Sandbox & Tool Runner
Subprocess execution harness with hard timeouts, stdout/stderr capture, and deterministic exit-code parsing.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class ExecutionResult:
    """Result of a sandboxed command execution."""

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_ms: float
    command: list[str]
    cwd: str
    pid: int | None = None

    @property
    def success(self) -> bool:
        """Check if execution was successful (exit code 0, not timed out)."""
        return self.exit_code == 0 and not self.timed_out

    @property
    def error_message(self) -> str | None:
        """Get error message if failed."""
        if self.success:
            return None
        if self.timed_out:
            return f"Command timed out after {self.duration_ms:.0f}ms"
        return f"Command failed with exit code {self.exit_code}: {self.stderr[:500]}"


class SandboxRunner:
    """
    Deterministic subprocess execution harness.

    Features:
    - Hard execution timeouts via asyncio.wait_for
    - stdout/stderr capture
    - Deterministic exit-code parsing
    - Process group kill on timeout with zombie reaping
    - Configurable working directory and environment
    """

    # Exit code constants
    EXIT_SUCCESS = 0
    EXIT_ERROR = 1
    EXIT_TIMEOUT = 124
    EXIT_OOM_KILL = 137
    EXIT_SIGKILL = 137
    EXIT_SIGTERM = 143

    def __init__(
        self,
        default_timeout: float = 300.0,
        default_cwd: str | None = None,
        default_env: dict[str, str] | None = None,
        max_output_size: int = 10 * 1024 * 1024,  # 10MB
    ):
        self.default_timeout = default_timeout
        self.default_cwd = default_cwd or os.getcwd()
        self.default_env = default_env or os.environ.copy()
        self.max_output_size = max_output_size

    async def run(
        self,
        command: str | list[str],
        timeout: float | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        capture_output: bool = True,
        shell: bool = False,
    ) -> ExecutionResult:
        """
        Execute a command with timeout and output capture.

        Args:
            command: Command to execute (string or list of args)
            timeout: Execution timeout in seconds (default: 300s)
            cwd: Working directory (default: current directory)
            env: Environment variables (default: inherited)
            capture_output: Whether to capture stdout/stderr
            shell: Whether to run through shell

        Returns:
            ExecutionResult with exit code, stdout, stderr, and timing
        """
        if isinstance(command, str):
            if not shell:
                command = command.split()
            cmd_list = command
        else:
            cmd_list = command

        timeout = timeout or self.default_timeout
        cwd = cwd or self.default_cwd
        env = env or self.default_env

        start_time = time.monotonic()
        timed_out = False
        pid = None

        try:
            # Create process
            if shell:
                process = await asyncio.create_subprocess_shell(  # nosec B604
                    cmd_list if isinstance(cmd_list, str) else " ".join(cmd_list),
                    stdout=asyncio.subprocess.PIPE
                    if capture_output
                    else asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE
                    if capture_output
                    else asyncio.subprocess.DEVNULL,
                    cwd=cwd,
                    env=env,
                    start_new_session=True,  # Create new process group for clean kill
                )
            else:
                process = await asyncio.create_subprocess_exec(  # nosec B604
                    *cmd_list,
                    stdout=asyncio.subprocess.PIPE
                    if capture_output
                    else asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE
                    if capture_output
                    else asyncio.subprocess.DEVNULL,
                    cwd=cwd,
                    env=env,
                    start_new_session=True,
                )

            pid = process.pid

            # Wait with timeout
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
                exit_code = process.returncode
            except asyncio.TimeoutError:  # noqa: UP041
                timed_out = True
                # Kill entire process group
                with contextlib.suppress(ProcessLookupError, OSError):
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                # Wait for process to actually terminate and reap zombie
                await process.wait()
                exit_code = self.EXIT_TIMEOUT
                stdout_bytes = b""
                stderr_bytes = f"Timeout: process killed after {timeout}s".encode()

            duration_ms = (time.monotonic() - start_time) * 1000

            # Decode output with size limit
            stdout = self._decode_output(stdout_bytes) if capture_output else ""
            stderr = self._decode_output(stderr_bytes) if capture_output else ""

            return ExecutionResult(
                exit_code=exit_code if exit_code is not None else self.EXIT_ERROR,
                stdout=stdout,
                stderr=stderr,
                timed_out=timed_out,
                duration_ms=duration_ms,
                command=cmd_list if isinstance(cmd_list, list) else [cmd_list],
                cwd=cwd,
                pid=pid,
            )

        except Exception as e:
            duration_ms = (time.monotonic() - start_time) * 1000
            return ExecutionResult(
                exit_code=self.EXIT_ERROR,
                stdout="",
                stderr=f"Execution error: {str(e)}",
                timed_out=False,
                duration_ms=duration_ms,
                command=cmd_list if isinstance(cmd_list, list) else [cmd_list],
                cwd=cwd,
                pid=pid,
            )

    def _decode_output(self, data: bytes) -> str:
        """Decode bytes to string with size limit."""
        if len(data) > self.max_output_size:
            data = data[: self.max_output_size] + b"\n... [output truncated]"
        try:
            return data.decode("utf-8", errors="replace")
        except UnicodeDecodeError:
            return data.decode("latin-1", errors="replace")

    def run_sync(
        self,
        command: str | list[str],
        timeout: float | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        capture_output: bool = True,
        shell: bool = False,
    ) -> ExecutionResult:
        """
        Synchronous version of run() for non-async contexts.
        """
        return asyncio.run(
            self.run(  # nosec B604
                command=command,
                timeout=timeout,
                cwd=cwd,
                env=env,
                capture_output=capture_output,
                shell=shell,
            )
        )

    async def run_batch(
        self,
        commands: list[str | list[str]],
        timeout: float | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        max_concurrent: int = 3,
    ) -> list[ExecutionResult]:
        """
        Run multiple commands concurrently with semaphore limiting.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def run_one(cmd: str | list[str]) -> ExecutionResult:
            async with semaphore:
                return await self.run(cmd, timeout, cwd, env)

        tasks = [run_one(cmd) for cmd in commands]
        return await asyncio.gather(*tasks)

    @staticmethod
    def parse_exit_code(exit_code: int) -> dict[str, Any]:
        """
        Parse exit code into structured information.

        Args:
            exit_code: The exit code to parse.

        Returns:
            Dict with keys: success, signal, meaning
        """
        if exit_code == 0:
            return {"success": True, "signal": None, "meaning": "Success"}
        elif exit_code == 124:
            return {"success": False, "signal": "SIGXCPU", "meaning": "Timeout (SIGXCPU)"}
        elif exit_code == 137:
            # 137 = 128 + 9 (SIGKILL) - could be OOM kill or manual kill
            # We can't distinguish without /proc/<pid>/oom_score_adj, so report both
            return {
                "success": False,
                "signal": "SIGKILL",
                "meaning": "Killed by SIGKILL (OOM kill or manual)",
            }
        elif exit_code == 143:
            return {"success": False, "signal": "SIGTERM", "meaning": "Terminated (SIGTERM)"}
        elif 128 < exit_code < 160:
            signal_num = exit_code - 128
            return {
                "success": False,
                "signal": f"SIG{signal_num}",
                "meaning": f"Killed by signal {signal_num}",
            }
        else:
            return {"success": False, "signal": None, "meaning": f"Error code {exit_code}"}


# Convenience functions
_default_runner: SandboxRunner | None = None


def get_sandbox_runner(
    default_timeout: float = 300.0,
    default_cwd: str | None = None,
    default_env: dict[str, str] | None = None,
    max_output_size: int = 10 * 1024 * 1024,
) -> SandboxRunner:
    """Get or create the default sandbox runner."""
    global _default_runner
    if _default_runner is None:
        _default_runner = SandboxRunner(
            default_timeout=default_timeout,
            default_cwd=default_cwd,
            default_env=default_env,
            max_output_size=max_output_size,
        )
    return _default_runner


def set_sandbox_runner(runner: SandboxRunner) -> None:
    """Set the default sandbox runner (useful for testing)."""
    global _default_runner
    _default_runner = runner


__all__ = [
    "SandboxRunner",
    "ExecutionResult",
    "get_sandbox_runner",
    "set_sandbox_runner",
]
