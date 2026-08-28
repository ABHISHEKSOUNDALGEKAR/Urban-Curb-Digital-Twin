"""Experiment provenance: what was run, with what code, on what machine.

A simulation result without provenance is not a result, it is an anecdote. Every
experiment writes a manifest recording the git commit, the resolved
configuration and its hash, the seed list, the platform and the wall-clock cost,
so a number in the README can always be traced back to the exact state that
produced it.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def git_commit(short: bool = True) -> str:
    """Current git commit, or ``"unknown"`` outside a repository."""
    try:
        args = ["git", "rev-parse", "--short" if short else "HEAD", "HEAD"]
        if not short:
            args = ["git", "rev-parse", "HEAD"]
        out = subprocess.run(
            args, capture_output=True, text=True, timeout=5, cwd=Path(__file__).resolve().parents[2]
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - environment dependent
        pass
    return "unknown"


def git_dirty() -> bool:
    """True if the working tree has uncommitted changes."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parents[2],
        )
        return bool(out.stdout.strip()) if out.returncode == 0 else False
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return False


def config_hash(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class Manifest:
    """Everything needed to re-run an experiment exactly."""

    experiment: str
    scenario: str
    seeds: list[int]
    config: dict[str, Any]
    config_hash: str
    git_commit: str
    git_dirty: bool
    python_version: str
    platform: str
    cpu_count: int
    started_utc: str
    finished_utc: str = ""
    wall_time_s: float = 0.0
    n_workers: int = 1
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        experiment: str,
        scenario: str,
        seeds: list[int],
        config: dict[str, Any],
        n_workers: int = 1,
        **extra: Any,
    ) -> Manifest:
        import os

        return cls(
            experiment=experiment,
            scenario=scenario,
            seeds=list(seeds),
            config=config,
            config_hash=config_hash(config),
            git_commit=git_commit(),
            git_dirty=git_dirty(),
            python_version=sys.version.split()[0],
            platform=platform.platform(),
            cpu_count=os.cpu_count() or 1,
            started_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            n_workers=n_workers,
            extra=dict(extra),
        )

    def finish(self, wall_time_s: float) -> Manifest:
        self.finished_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.wall_time_s = float(wall_time_s)
        return self

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=str), encoding="utf-8")
        return path
