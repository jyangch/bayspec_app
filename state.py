"""Per-session in-memory state store."""

from typing import Any
import uuid

_sessions: dict[str, dict[str, Any]] = {}


def _new() -> dict[str, Any]:
    return {
        # {data_key: bayspec.data.data.Data}  — Data container holds {unit_key: DataUnit}
        # {data_key: bayspec.data.data.Data}  -- Data container holds {unit_key: DataUnit}
        'data': {},
        # {data_key: {"model_binding": str|None, "units": {unit_key: {form fields, error}}}}
        'data_state': {},
        # {model_key: bayspec.model.model.Model}  -- composed/leaf model objects
        'model': {},
        # {component_key: bayspec.model.model.Model}  -- individual leaf components (pl, cpl, etc.)
        'model_component': {},
        # {model_key: {"expr": str, "components": [component_key, ...], error fields, ...}}
        'model_state': {},
        'infer': None,
        'infer_state': {
            'pairs': [],  # Auto-derived from data↔model bindings
            'nlink': 0,  # Number of parameter link groups
            'links': {},  # {idx: [par_id, ...]}
            'sampler': 'emcee',
            'nstep': 1000,
            'discard': 100,
            'nlive': 400,
            'savepath': 'output',
            'resume': False,
            'result': None,
            'stat_ic': None,
            'posterior': None,
            'error': None,
        },
        'custom_models': {},
        'editor_state': {'status': None, 'status_type': 'success'},
    }


def get(session_id: str) -> dict[str, Any]:
    if session_id not in _sessions:
        _sessions[session_id] = _new()
    return _sessions[session_id]


def new_id() -> str:
    return str(uuid.uuid4())
