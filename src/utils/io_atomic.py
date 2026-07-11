"""Atomic filesystem helpers for a checkpointed, resumable pipeline.

Every write goes to a uuid-suffixed temp path in the same directory, is
flushed and fsynced, then os.replace()d onto the destination. A reader
therefore never observes a half-written file, and a resubmitted job that
finds a `_DONE.json` marker can skip finished work as an instant no-op.
"""

import json
import os
import uuid
from typing import Any, Dict, Optional


def _tmp_path(final_path: str) -> str:
    """Return a temp path in the same directory as ``final_path``.

    Same-directory placement guarantees os.replace() is atomic (it stays
    on one filesystem).
    """
    directory = os.path.dirname(os.path.abspath(final_path))
    base = os.path.basename(final_path)
    return os.path.join(directory, ".{}.{}.tmp".format(base, uuid.uuid4().hex))


def atomic_write_text(final_path: str, text: str) -> None:
    """Write ``text`` to ``final_path`` atomically."""
    os.makedirs(os.path.dirname(os.path.abspath(final_path)), exist_ok=True)
    tmp = _tmp_path(final_path)
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, final_path)


def atomic_write_json(final_path: str, obj: Any) -> None:
    """Serialize ``obj`` to JSON and write it atomically."""
    atomic_write_text(final_path, json.dumps(obj, indent=2, sort_keys=True))


def load_json(path: str) -> Any:
    """Load and return the JSON object stored at ``path``."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def done_marker_path(unit_dir: str) -> str:
    """Return the `_DONE.json` marker path for a work-unit directory."""
    return os.path.join(unit_dir, "_DONE.json")


def is_done(unit_dir: str) -> bool:
    """Return True if the work unit at ``unit_dir`` has a done marker."""
    return os.path.isfile(done_marker_path(unit_dir))


def write_done(unit_dir: str, payload: Optional[Dict[str, Any]] = None) -> None:
    """Mark a work unit finished by writing its `_DONE.json` atomically.

    Called only after all outputs of the unit are safely on disk, so its
    presence is a reliable signal that the unit can be skipped on resume.
    """
    os.makedirs(unit_dir, exist_ok=True)
    atomic_write_json(done_marker_path(unit_dir), payload or {"status": "ok"})
