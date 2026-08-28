"""Optional SUMO/TraCI traffic-physics backend.

The SimPy engine models curb competition with an analytical link performance
function. That is fast enough to run thousands of replications, which is what
calibration and optimization need, but it abstracts away lane-level dynamics:
queue spillback, the actual manoeuvre of pulling into a stall, the way a
double-parked truck interacts with the vehicles behind it.

This package swaps that layer for SUMO microsimulation while keeping every
*decision* in Python:

    Python                          SUMO (via TraCI)
    ------                          ----------------
    demand generation               car-following and lane changing
    curb choice / scoring           link travel times and congestion
    search and fallback logic       parking area occupancy
    ridehail dispatch               routing and rerouting
    experiment orchestration        vehicle movement

Everything here is optional. SUMO is not a dependency of the core model, and if
it is not installed the rest of the project runs unchanged; ``sumo_available()``
is the guard, and the SUMO tests skip themselves rather than fail.

Install with::

    pip install eclipse-sumo traci sumolib     # bundled binaries, or
    # a system SUMO install with SUMO_HOME set
"""

from __future__ import annotations

import functools
import os
import shutil
from pathlib import Path


@functools.lru_cache(maxsize=1)
def sumo_home() -> Path | None:
    """Locate a SUMO installation, preferring an explicit ``SUMO_HOME``."""
    env = os.environ.get("SUMO_HOME")
    if env and Path(env).exists():
        return Path(env)
    try:  # the pip-installed `eclipse-sumo` wheel ships the binaries
        import sumo  # type: ignore

        return Path(sumo.__file__).parent
    except Exception:
        return None


@functools.lru_cache(maxsize=8)
def sumo_binary(name: str = "sumo") -> str | None:
    """Absolute path to a SUMO executable (``sumo``, ``sumo-gui``, ``netconvert``)."""
    found = shutil.which(name)
    if found:
        return found
    home = sumo_home()
    if home is None:
        return None
    for candidate in (home / "bin" / name, home / name, home / "bin" / f"{name}.exe"):
        if candidate.exists():
            return str(candidate)
    return None


def sumo_available() -> bool:
    """True when both the SUMO binaries and the TraCI python bindings are present."""
    if sumo_binary("sumo") is None or sumo_binary("netconvert") is None:
        return False
    try:
        import sumolib  # noqa: F401
        import traci  # noqa: F401
    except ImportError:
        return False
    return True


def require_sumo() -> None:
    if not sumo_available():
        raise RuntimeError(
            "SUMO is not available. Install it with `pip install eclipse-sumo traci sumolib` "
            "or set SUMO_HOME to a SUMO installation."
        )


__all__ = ["sumo_available", "require_sumo", "sumo_binary", "sumo_home"]
