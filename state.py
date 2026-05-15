"""Per-session in-memory state store."""
import uuid
from typing import Any

_sessions: dict[str, dict[str, Any]] = {}


def _new() -> dict[str, Any]:
    return {
        # {data_key: bayspec.data.data.Data}  — Data container holds {unit_key: DataUnit}
        "data": {},
        # {data_key: {"model_binding": str|None, "units": {unit_key: {form fields, error}}}}
        "data_state": {},
        "model": {},
        "model_component": {},
        "model_state": {},
        "infer": None,
        "infer_state": {
            "pairs": [],
            "sampler": "emcee",
            "nstep": 1000,
            "discard": 100,
            "nlive": 400,
            "savepath": "./infer_out",
            "result": None,
            "error": None,
        },
        "custom_models": {},
        "editor_state": {"status": None, "status_type": "success"},
    }


def get(session_id: str) -> dict[str, Any]:
    if session_id not in _sessions:
        _sessions[session_id] = _new()
    return _sessions[session_id]


def new_id() -> str:
    return str(uuid.uuid4())
