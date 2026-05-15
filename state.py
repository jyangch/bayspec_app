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
        "infer": None,       # BayesInfer | None
        "infer_state": {     # UI + run config
            "pairs": [],
            "sampler": "emcee",
            "nstep": 1000,
            "discard": 100,
            "nlive": 400,
            "savepath": "./infer_out",
            "result": None,
            "error": None,
        },
        "custom_models": {},    # {name: cls} registered this session
        "editor_state": {"status": None, "status_type": "success"},
    }


def get(session_id: str) -> dict[str, Any]:
    if session_id not in _sessions:
        _sessions[session_id] = _new()
    return _sessions[session_id]


def new_id() -> str:
    return str(uuid.uuid4())
