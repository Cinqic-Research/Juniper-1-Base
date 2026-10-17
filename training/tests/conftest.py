import os

import pytest

# Deterministic cuBLAS must be configured before CUDA initializes in this process; an earlier GPU
# test would otherwise make the trainer's own setdefault too late (order-dependent failure).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


@pytest.fixture(autouse=True)
def _time_log_root(tmp_path, monkeypatch):
    # Trainer runs always write spec v0.6 section 8 time logs; keep test logs out of $HOME.
    monkeypatch.setenv("J1_TIME_LOG_ROOT", str(tmp_path / "timelogs"))
