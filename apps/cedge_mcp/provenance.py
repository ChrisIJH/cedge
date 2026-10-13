"""
apps/cedge_mcp/provenance.py

Every tool result carries a provenance block: which service endpoints were
called with which arguments, when, and from which code version. This is what
lets a reviewer trace any number an LLM quotes back to its source.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List


def call_record(service: str, method: str, path: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """One HTTP call, as recorded in provenance. Large arrays are summarized,
    not copied, so the provenance block stays small."""
    compact = {}
    for key, value in args.items():
        if isinstance(value, list) and len(value) > 10:
            compact[key] = f"<list of {len(value)}>"
        elif isinstance(value, dict) and len(value) > 10:
            compact[key] = f"<dict of {len(value)} keys>"
        else:
            compact[key] = value
    return {"service": service, "method": method, "path": path, "args": compact}


def build(tool: str, calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "tool": tool,
        "calls": calls,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }