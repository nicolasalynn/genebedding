"""
Write pipeline progress to a JSON file for external monitoring (e.g. monitor script).
Set PIPELINE_STATUS_FILE or pass status_path to pipeline functions.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def write_status(
    path: Optional[Path] = None,
    *,
    phase: Optional[str] = None,
    env_profile: Optional[str] = None,
    model_key: Optional[str] = None,
    source: Optional[str] = None,
    n_total: Optional[int] = None,
    n_done: Optional[int] = None,
    tools_done: Optional[int] = None,
    tools_total: Optional[int] = None,
    sources_done_for_tool: Optional[int] = None,
    sources_total_for_tool: Optional[int] = None,
    message: Optional[str] = None,
    error: Optional[str] = None,
    **extra: Any,
) -> None:
    """Overwrite status file with given fields. Pass path from env or output_base."""
    if path is None:
        import os
        p = os.environ.get("PIPELINE_STATUS_FILE")
        path = Path(p) if p else None
    if path is None:
        return
    path = Path(path)
    payload: dict[str, Any] = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    if phase is not None:
        payload["phase"] = phase
    if env_profile is not None:
        payload["env_profile"] = env_profile
    if model_key is not None:
        payload["model_key"] = model_key
    if source is not None:
        payload["source"] = source
    if n_total is not None:
        payload["n_total"] = n_total
    if n_done is not None:
        payload["n_done"] = n_done
    if tools_done is not None:
        payload["tools_done"] = tools_done
    if tools_total is not None:
        payload["tools_total"] = tools_total
    if sources_done_for_tool is not None:
        payload["sources_done_for_tool"] = sources_done_for_tool
    if sources_total_for_tool is not None:
        payload["sources_total_for_tool"] = sources_total_for_tool
    if message is not None:
        payload["message"] = message
    if error is not None:
        payload["error"] = error
    payload.update(extra)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
    except Exception:
        pass
