"""Per-session in-memory state store."""
import uuid
from typing import Any

_sessions: dict[str, dict[str, Any]] = {}


def _new() -> dict[str, Any]:
    return {
        "data": {},          # {unit_key: DataUnit}
        "data_state": {},    # {unit_key: {field: value}} — widget state mirror
        "model": {},         # {model_key: Model}
        "model_component": {},  # {model_key: {comp_key: component}}
        "model_state": {},   # {model_key: {comp_key: {field: value}}}
        "infer": None,       # BayesInfer | MaxLikeFit | None
        "infer_state": {},   # {field: value}
    }


def get(session_id: str) -> dict[str, Any]:
    if session_id not in _sessions:
        _sessions[session_id] = _new()
    return _sessions[session_id]


def new_id() -> str:
    return str(uuid.uuid4())
