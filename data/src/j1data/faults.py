"""Fault injection for recovery tests and drills (no effect unless configured).

``J1_FAULT_POINT=<name>`` arms one named point. When the current process passes that
point for the ``J1_FAULT_AFTER``-th time (default 1), ``J1_FAULT_ACTION`` decides what
happens: ``kill`` (default) sends SIGKILL to the process, so nothing after the point runs
(no ``finally``, no buffered flush); ``enospc`` raises ``OSError(ENOSPC)`` as a disk-full
write failure would. Counts are per process.
"""

from __future__ import annotations

import errno
import os
import signal

_COUNTS: dict[str, int] = {}


def point(name: str) -> None:
    if os.environ.get("J1_FAULT_POINT") != name:
        return
    _COUNTS[name] = _COUNTS.get(name, 0) + 1
    if _COUNTS[name] < int(os.environ.get("J1_FAULT_AFTER", "1")):
        return
    if os.environ.get("J1_FAULT_ACTION", "kill") == "enospc":
        raise OSError(errno.ENOSPC, f"injected at {name}: No space left on device")
    os.kill(os.getpid(), signal.SIGKILL)
