"""Isolated execution of untrusted Python (dataset references, model-generated code).

Every run happens inside ``bubblewrap`` with all namespaces unshared (no network, a private
PID tree, no IPC), only ``/usr`` mounted read-only, an empty ``/tmp`` and no home directory,
the program read from stdin (so no file in the sandbox contains it or its hidden tests),
host secrets, environment variables or the caller's working tree. The interpreter is the system
``/usr/bin/python3`` in isolated mode (``-I``), so neither the caller's virtual environment nor
its site-packages are visible. CPU time, address space, output file size and wall time are
limited, and captured output is truncated.

There is deliberately no unsandboxed fallback: if ``bwrap`` is missing or cannot create its
namespaces, :func:`run_python` raises :class:`SandboxUnavailable`. A caller must then fail the
admission or score as *not executed*, never as passed or as an ordinary failure.
"""

from __future__ import annotations

import contextlib
import os
import resource
import secrets
import selectors
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from functools import cache

VERSION = "j1.sandbox.v1"
PYTHON = "/usr/bin/python3"
MAX_OUTPUT_CHARS = 8000


class SandboxUnavailable(RuntimeError):
    """bubblewrap cannot provide the required isolation on this host."""


@dataclass(frozen=True)
class Limits:
    cpu_seconds: int = 10
    wall_seconds: float = 20.0
    address_space_bytes: int = 1 << 30
    file_size_bytes: int = 1 << 20
    output_bytes: int = 1 << 20  # aggregate stdout + stderr; RLIMIT_FSIZE does not limit pipes


@dataclass(frozen=True)
class Result:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    output_limited: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out and not self.output_limited


def _command(launcher: str) -> list[str]:
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        raise SandboxUnavailable("bwrap (bubblewrap) is not installed")
    return [
        bwrap,
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--cap-drop",
        "ALL",
        "--ro-bind",
        "/usr",
        "/usr",
        "--symlink",
        "usr/lib",
        "/lib",
        "--symlink",
        "usr/lib64",
        "/lib64",
        "--symlink",
        "usr/bin",
        "/bin",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--chdir",
        "/tmp",
        "--clearenv",
        "--setenv",
        "PATH",
        "/usr/bin",
        "--setenv",
        "PYTHONHASHSEED",
        "0",
        "--hostname",
        "sandbox",
        PYTHON,
        "-I",
        "-S",
        "-c",
        launcher,  # runs the program that arrives on stdin: no file inside the sandbox holds it
    ]


def _apply(limits: Limits):  # type: ignore[no-untyped-def]
    def preexec() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds))
        resource.setrlimit(resource.RLIMIT_AS, (limits.address_space_bytes, limits.address_space_bytes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (limits.file_size_bytes, limits.file_size_bytes))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    return preexec


def run_python(source: str, limits: Limits | None = None) -> Result:
    """Run ``source`` as a script in the sandbox and return its exit status and (truncated) output.

    The interpreter's first statement writes a per-run token to a pipe the host holds; the token arriving
    proves the sandbox was set up. Without it the run is a setup failure (:class:`SandboxUnavailable`).
    The program's own stderr never decides this (a program can print anything, e.g. a fake bwrap error).
    """
    limits = limits or Limits()
    if limits.output_bytes < 1:
        raise ValueError("output_bytes must be positive")
    token = secrets.token_hex(16)
    start_r, start_w = os.pipe()
    launcher = (
        f"import os, sys; os.write({start_w}, b{token!r}); os.close({start_w}); "
        "src = sys.stdin.read(); del os, sys; "
        "exec(compile(src, '<stdin>', 'exec'), {'__name__': '__main__', '__builtins__': __builtins__})"
    )
    try:
        proc = subprocess.Popen(
            _command(launcher),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=_apply(limits),
            pass_fds=(start_w,),
            env={},
            start_new_session=True,
        )
    except BaseException:
        os.close(start_r)
        raise
    finally:
        os.close(start_w)
    # Drain while supplying stdin: communicate() would first buffer arbitrarily large pipe output
    # in the host. Keep byte tails only, cap the aggregate stream and kill/reap the namespace owner.
    out, err, timed_out, output_limited = _drain(proc, source.encode("utf-8"), limits)
    with os.fdopen(start_r, "rb") as f:
        started = f.read(64) == token.encode()
    stderr = err.decode("utf-8", "replace")
    if not started:
        raise SandboxUnavailable(stderr.strip()[:300] or f"sandbox did not start (exit {proc.returncode})")
    if timed_out:
        return Result(-9, out.decode("utf-8", "replace")[-MAX_OUTPUT_CHARS:], "timeout", True)  # kept tail
    if output_limited:
        return Result(
            -9, out.decode("utf-8", "replace")[-MAX_OUTPUT_CHARS:], "output limit exceeded", False, True
        )
    return Result(
        proc.returncode,
        out.decode("utf-8", "replace")[-MAX_OUTPUT_CHARS:],  # the tail holds a completion nonce
        stderr[-MAX_OUTPUT_CHARS:],
        False,
    )


def _drain(proc: subprocess.Popen[bytes], source: bytes, limits: Limits) -> tuple[bytes, bytes, bool, bool]:
    assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
    tails = {"stdout": bytearray(), "stderr": bytearray()}
    timed_out = output_limited = False
    drained = False
    total = cursor = 0
    deadline = time.monotonic() + limits.wall_seconds
    # Four bytes per retained character cover UTF-8; decoding/slicing applies the public char limit.
    tail_bytes = 4 * MAX_OUTPUT_CHARS
    with selectors.DefaultSelector() as sel:
        for stream, name, event in (
            (proc.stdin, "stdin", selectors.EVENT_WRITE),
            (proc.stdout, "stdout", selectors.EVENT_READ),
            (proc.stderr, "stderr", selectors.EVENT_READ),
        ):
            os.set_blocking(stream.fileno(), False)
            sel.register(stream, event, name)
        try:
            while sel.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                for key, _ in sel.select(min(remaining, 0.1)):
                    name = key.data
                    stream = {"stdin": proc.stdin, "stdout": proc.stdout, "stderr": proc.stderr}[name]
                    if name == "stdin":
                        try:
                            count = os.write(stream.fileno(), source[cursor : cursor + 65536])
                            cursor += count
                        except BrokenPipeError:
                            cursor = len(source)
                        if cursor == len(source):
                            sel.unregister(stream)
                            stream.close()
                        continue
                    chunk = os.read(stream.fileno(), 65536)
                    if not chunk:
                        sel.unregister(stream)
                        stream.close()
                        continue
                    total += len(chunk)
                    tails[name].extend(chunk)
                    del tails[name][:-tail_bytes]
                    if total > limits.output_bytes:
                        output_limited = True
                        break
                if output_limited:
                    break
            drained = True
        finally:
            if drained and not timed_out and not output_limited:
                try:
                    proc.wait(timeout=max(0.001, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    timed_out = True
            # Also kill descendants when the program closed its streams but left processes alive.
            # bwrap owns the private PID namespace: killing it destroys that namespace's processes.
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                stream.close()
            proc.wait()
    return bytes(tails["stdout"]), bytes(tails["stderr"]), timed_out, output_limited


@cache
def available() -> bool:
    """True when the sandbox works here and really isolates the network and the home directory."""
    probe = (
        "import os, socket\n"
        "assert not os.path.exists('/home') and not os.path.exists('/root/.ssh')\n"
        "assert os.environ.get('HOME') is None\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 53), timeout=1)\n"
        "    raise SystemExit(3)\n"
        "except OSError:\n"
        "    pass\n"
        "print('isolated')\n"
    )
    try:
        r = run_python(probe, Limits(cpu_seconds=5, wall_seconds=10))
    except SandboxUnavailable:
        return False
    return r.ok and r.stdout.strip() == "isolated"
