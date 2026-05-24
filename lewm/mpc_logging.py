"""
Verbose MPC / CEM diagnostics.

Enable via:
  - ``diagnose_mpc.py --verbose`` or ``lewm_server.py --verbose``
  - env ``LEWM_MPC_VERBOSE=1`` (works without CLI flag)
"""

from __future__ import annotations

import os

MPC_VERBOSE = os.environ.get("LEWM_MPC_VERBOSE", "").lower() in ("1", "true", "yes")


def set_mpc_verbose(enabled: bool) -> None:
    global MPC_VERBOSE
    MPC_VERBOSE = bool(enabled)


def mpc_log(msg: str) -> None:
    if MPC_VERBOSE:
        print(f"[MPC] {msg}")
