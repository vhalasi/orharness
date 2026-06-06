"""Safe execution of LLM-generated OR-Tools scripts.

v0.1: subprocess with a timeout. The sandbox runs code and captures output;
JSON parsing and result interpretation belong in the pipeline.
"""

import os
import subprocess
import sys
import tempfile
import time

from orharness.models import GeneratedCode, ExecutionResult, ORHarnessConfig
from orharness.exceptions import SandboxTimeoutError


def run_code(
    generated: GeneratedCode,
    config: ORHarnessConfig,
) -> ExecutionResult:
    """Execute generated Python in a subprocess and return captured output.

    - success=True when the process exits with code 0
    - success=False when the process crashes (non-zero exit)
    - raises SandboxTimeoutError if execution exceeds config.timeout_seconds
    """
    script_path = None
    start = time.monotonic()

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
        ) as f:
            f.write(generated.code)
            script_path = f.name

        proc = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=config.timeout_seconds,
        )
    except subprocess.TimeoutExpired as e:
        elapsed = time.monotonic() - start
        stderr = e.stderr or ""
        stdout = e.stdout or ""
        raise SandboxTimeoutError(
            f"Solver exceeded timeout of {config.timeout_seconds}s. "
            f"stderr: {stderr[:500]}"
        ) from e
    finally:
        if script_path and os.path.exists(script_path):
            os.unlink(script_path)

    elapsed = time.monotonic() - start
    success = proc.returncode == 0

    return ExecutionResult(
        success=success,
        raw_output=proc.stdout or None,
        error_message=None if success else (proc.stderr or proc.stdout or None),
        solve_time_seconds=elapsed,
    )
