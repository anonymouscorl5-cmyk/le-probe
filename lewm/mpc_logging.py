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


def mpc_shape_log(where: str, **named_tensors) -> None:
    """Always print tensor shapes (not gated on --verbose)."""
    lines = [f"[MPC:shape] {where}"]
    for name, val in named_tensors.items():
        if val is None:
            lines.append(f"  {name}: None")
        elif hasattr(val, "shape"):
            lines.append(f"  {name}: shape={tuple(val.shape)} ndim={val.ndim} dtype={getattr(val, 'dtype', '?')}")
        elif isinstance(val, (tuple, list)):
            lines.append(f"  {name}: {val}")
        else:
            lines.append(f"  {name}: {val!r}")
    print("\n".join(lines))
