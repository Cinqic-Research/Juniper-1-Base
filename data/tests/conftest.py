"""Import ``j1data`` from this checkout's ``src``, even if the editable install is another worktree."""

import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parents[1] / "src")
if sys.path[0] != SRC:
    sys.path.insert(0, SRC)
for name in [m for m in sys.modules if m == "j1data" or m.startswith("j1data.")]:
    del sys.modules[name]
