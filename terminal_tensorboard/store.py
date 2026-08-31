"""Run discovery and incremental scalar storage.

A *run* is any directory under the log dir that contains tfevents files
(the same convention TensorBoard uses).  Data is kept in compact
``array`` buffers and every refresh only reads bytes appended since the
previous one, so tailing a live training run costs almost nothing.
"""

from __future__ import annotations

import os
import threading
from array import array
from typing import Dict, List, Optional, Tuple

from . import tfevents


class Series:
    """One scalar tag inside one run."""

    __slots__ = ("steps", "walls", "vals", "_monotonic")

    def __init__(self) -> None:
        self.steps = array("q")
        self.walls = array("d")
        self.vals = array("d")
        self._monotonic = True

    def __len__(self) -> int:
        return len(self.steps)

    def append(self, step: int, wall: float, val: float) -> None:
        if self.steps and step < self.steps[-1]:
            self._monotonic = False
        self.steps.append(step)
        self.walls.append(wall)
        self.vals.append(val)

    def ensure_sorted(self) -> None:
        """Sort by step (rare path: only after a restarted/overlapping run)."""
        if self._monotonic:
            return
        order = sorted(range(len(self.steps)), key=self.steps.__getitem__)
        self.steps = array("q", (self.steps[i] for i in order))
        self.walls = array("d", (self.walls[i] for i in order))
        self.vals = array("d", (self.vals[i] for i in order))
        self._monotonic = True


class _EventFile:
    __slots__ = ("path", "offset", "size", "dead", "intern")

    def __init__(self, path: str) -> None:
        self.path = path
        self.offset = 0
        self.size = 0
        self.dead = False  # set on unreadable/corrupt files
        self.intern: Dict[bytes, str] = {}


class Run:
    def __init__(self, name: str) -> None:
        self.name = name
        self.files: Dict[str, _EventFile] = {}
        self.series: Dict[str, Series] = {}
        self.first_wall: Optional[float] = None


def _is_event_file(name: str) -> bool:
    return "tfevents" in name and not name.endswith(".profile-empty")


class ScalarStore:
    """Thread-safe store filled by a background loader thread."""

    def __init__(self, logdir: str) -> None:
        self.logdir = os.path.abspath(logdir)
        self.lock = threading.Lock()
        self.runs: Dict[str, Run] = {}
        self.version = 0  # bumped on every data change
        self.total_points = 0
        self.errors: List[str] = []

    # -- discovery ---------------------------------------------------------

    def _discover(self) -> None:
        found: List[Tuple[str, str]] = []  # (run_name, file_path)
        for dirpath, dirnames, filenames in os.walk(self.logdir, followlinks=True):
            dirnames.sort()
            for fn in sorted(filenames):
                if _is_event_file(fn):
                    rel = os.path.relpath(dirpath, self.logdir)
                    run_name = "." if rel == "." else rel.replace(os.sep, "/")
                    found.append((run_name, os.path.join(dirpath, fn)))
        with self.lock:
            for run_name, path in found:
                run = self.runs.get(run_name)
                if run is None:
                    run = self.runs[run_name] = Run(run_name)
                if path not in run.files:
                    run.files[path] = _EventFile(path)

    # -- incremental reads -------------------------------------------------

    def _read_file(self, ef: _EventFile) -> Optional[Tuple[List[tfevents.ScalarPoint], int]]:
        try:
            size = os.path.getsize(ef.path)
        except OSError:
            return None
        if size <= ef.offset:
            return None
        try:
            with open(ef.path, "rb") as f:
                f.seek(ef.offset)
                data = f.read()
        except OSError as exc:
            ef.dead = True
            self.errors.append("%s: %s" % (ef.path, exc))
            return None
        try:
            points, consumed = tfevents.parse_chunk(data, ef.intern)
        except tfevents.CorruptRecord as exc:
            ef.dead = True
            self.errors.append("%s: %s" % (ef.path, exc))
            # keep whatever parsed before the corruption
            points, consumed = tfevents.parse_chunk(data[: exc.offset], ef.intern)
        return points, ef.offset + consumed

    def refresh(self) -> bool:
        """Discover new runs/files and ingest appended bytes.

        Parsing happens outside the lock; only the (fast) appends run under
        it, so the UI thread never stalls behind a large file.
        """
        self._discover()
        changed = False
        with self.lock:
            files = [
                (run, ef)
                for run in self.runs.values()
                for ef in run.files.values()
                if not ef.dead
            ]
        for run, ef in files:
            result = self._read_file(ef)
            if result is None:
                continue
            points, new_offset = result
            with self.lock:
                ef.offset = new_offset
                for tag, step, wall, val in points:
                    series = run.series.get(tag)
                    if series is None:
                        series = run.series[tag] = Series()
                    series.append(step, wall, val)
                    if wall and (run.first_wall is None or wall < run.first_wall):
                        run.first_wall = wall
                if points:
                    self.total_points += len(points)
                    self.version += 1
                    changed = True
        if changed:
            with self.lock:
                for run in self.runs.values():
                    for series in run.series.values():
                        series.ensure_sorted()
        return changed

    # -- queries (call under ``lock``) -------------------------------------

    def run_names(self) -> List[str]:
        return sorted(self.runs)

    def tags(self, enabled_runs: Optional[set] = None) -> List[str]:
        tags = set()
        for name, run in self.runs.items():
            if enabled_runs is None or name in enabled_runs:
                tags.update(run.series)
        return sorted(tags)
