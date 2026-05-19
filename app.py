import json

from st_pages import add_page_title, get_nav_from_toml
import streamlit as st

_SKIP = object()


def _strip(v):
    """Recursively coerce ``v`` to a JSON-safe structure.

    Anything that isn't ``str|int|float|bool|None|list|tuple|dict`` is
    dropped (so DataFrames, UploadedFile handles, Posterior objects etc.
    silently disappear without breaking the dump).
    """
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    if isinstance(v, (list, tuple)):
        out = []
        for x in v:
            sx = _strip(x)
            if sx is _SKIP:
                continue
            out.append(sx)
        return out
    if isinstance(v, dict):
        out = {}
        for k, x in v.items():
            sx = _strip(x)
            if sx is _SKIP:
                continue
            out[str(k)] = sx
        return out
    return _SKIP


def export_config_bytes() -> bytes:
    """Bundle the JSON-safe parts of session_state into a config blob."""
    data_state = dict(st.session_state.get('data_state', {}))
    model_state = dict(st.session_state.get('model_state', {}))
    infer_state = {
        k: v
        for k, v in st.session_state.get('infer_state', {}).items()
        # ``post`` (Posterior object) and the on-the-fly DataFrames are
        # not portable; everything else round-trips fine.
        if k not in ('post',)
    }
    cfg = {
        'version': 1,
        'data_state': _strip(data_state),
        'model_state': _strip(model_state),
        'infer_state': _strip(infer_state),
    }
    return json.dumps(cfg, indent=2).encode('utf-8')


def import_config_bytes(blob: bytes) -> tuple[bool, str]:
    """Replace the configuration dicts from a previously-exported blob.

    Returns ``(ok, message)``. On success, the page reruns so widget
    keys pick up the new values; uploaded FITS files still need to be
    re-attached because they cannot be serialised.
    """
    try:
        cfg = json.loads(blob.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return False, f'Could not parse config JSON: {exc}'

    if not isinstance(cfg, dict) or cfg.get('version') != 1:
        return False, 'Unsupported config version (expected 1).'

    for state_key in ('data_state', 'model_state', 'infer_state'):
        loaded = cfg.get(state_key)
        if not isinstance(loaded, dict):
            continue
        target = st.session_state.setdefault(state_key, {})
        target.clear()
        target.update(loaded)
        # Mirror primitives back into the top-level so widgets re-read
        # them on the next rerun.
        for k, v in loaded.items():
            if isinstance(v, (str, int, float, bool, type(None), list)):
                st.session_state[k] = v

    # Drop derived caches the import cannot re-create.
    st.session_state['data'] = {}
    st.session_state['model'] = {}
    st.session_state['model_component'] = {}
    st.session_state['infer'] = None

    return True, 'Configuration loaded — re-upload any FITS files to complete the setup.'


def init_session_state():
    if 'data' not in st.session_state:
        st.session_state.data = {}
    if 'data_state' not in st.session_state:
        st.session_state.data_state = {}
    if 'model' not in st.session_state:
        st.session_state.model = {}
    if 'model_component' not in st.session_state:
        st.session_state.model_component = {}
    if 'model_state' not in st.session_state:
        st.session_state.model_state = {}
    if 'infer' not in st.session_state:
        st.session_state.infer = None
    if 'infer_state' not in st.session_state:
        st.session_state.infer_state = {}


def _workflow_state() -> list[tuple[str, str, bool, str]]:
    """Compute the five-stage workflow progress from session_state.

    Returns a list of ``(emoji, title, done, caption)`` tuples in stage
    order, ready for the sidebar mini-stepper to render.
    """
    data = st.session_state.get('data', {})
    model = st.session_state.get('model', {})
    data_state = st.session_state.get('data_state', {})
    model_state = st.session_state.get('model_state', {})
    infer_state = st.session_state.get('infer_state', {})

    n_units = sum(
        len(getattr(d, 'data', {})) for d in data.values() if d is not None
    )
    n_models = sum(1 for m in model.values() if m is not None)
    n_pairs = 0
    for dk in data:
        mk = data_state.get(f'{dk}_model')
        if mk is None or model_state.get(f'{mk}_data') != dk:
            continue
        d = data.get(dk)
        if d is None or len(getattr(d, 'data', {})) == 0:
            continue
        if model.get(mk) is None:
            continue
        n_pairs += 1

    has_run = infer_state.get('built_hash') is not None
    has_post = 'post' in infer_state

    def caption(n: int, unit: str) -> str:
        return f'{n} {unit}{"s" if n != 1 else ""}'

    return [
        ('🔭', 'Data', n_units > 0, caption(n_units, 'unit')),
        ('🌈', 'Model', n_models > 0, caption(n_models, 'model')),
        ('🔗', 'Pairs', n_pairs > 0, caption(n_pairs, 'pair')),
        ('🚀', 'Inference', has_run, 'built' if has_run else 'pending'),
        ('📊', 'Posterior', has_post, 'ready' if has_post else 'pending'),
    ]


def render_workflow_sidebar() -> None:
    """Render the workflow mini-stepper inside ``st.sidebar``.

    The first not-yet-done stage gets the ``active`` class so the user
    sees, at a glance, what the next action is.
    """
    stages = _workflow_state()
    # First pending stage becomes the "active" one.
    active_idx = next((i for i, s in enumerate(stages) if not s[2]), -1)
    rows = []
    for i, (emoji, title, done, cap) in enumerate(stages):
        if done:
            cls = 'bsp-mini-step done'
            dot = '●'
        elif i == active_idx:
            cls = 'bsp-mini-step active'
            dot = '◉'
        else:
            cls = 'bsp-mini-step'
            dot = '○'
        rows.append(
            f'<div class="{cls}">'
            f'  <span class="bsp-mini-emoji">{emoji}</span>'
            f'  <span class="bsp-mini-body">'
            f'    <span class="bsp-mini-title">{title}</span>'
            f'    <span class="bsp-mini-caption">{cap}</span>'
            f'  </span>'
            f'  <span class="bsp-mini-dot">{dot}</span>'
            f'</div>'
        )
    st.markdown(
        '<div class="bsp-mini-stepper-head">Workflow</div>'
        '<div class="bsp-mini-stepper">' + ''.join(rows) + '</div>',
        unsafe_allow_html=True,
    )


GLOBAL_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bsp-primary: #4F46E5;
        --bsp-primary-soft: #6366F1;
        --bsp-primary-hover: #4338CA;
        --bsp-accent: #06B6D4;
        --bsp-accent-soft: #22D3EE;
        --bsp-success: #10B981;
        --bsp-warning: #F59E0B;
        --bsp-danger: #EF4444;
        --bsp-bg: #FFFFFF;
        --bsp-surface: #F8FAFC;
        --bsp-surface-2: #F1F5F9;
        --bsp-border: #E2E8F0;
        --bsp-border-soft: #EEF2F7;
        --bsp-text: #0F172A;
        --bsp-text-muted: #64748B;
        --bsp-shadow-sm: 0 1px 2px rgba(15, 23, 42, 0.04);
        --bsp-shadow-md: 0 4px 16px rgba(15, 23, 42, 0.06);
        --bsp-shadow-lg: 0 10px 30px rgba(79, 70, 229, 0.10);
        --bsp-sidebar-width: 320px;
    }

    /* Base typography — bumped up */
    html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif !important;
        font-feature-settings: "cv11", "ss01", "ss03";
        font-size: 16px;
    }
    code, pre, kbd, samp, .stCode, [data-testid="stCodeBlock"] {
        font-family: 'JetBrains Mono', 'SF Mono', Menlo, Consolas, monospace !important;
        font-size: 0.92rem;
    }
    p, li, label, .stMarkdown {
        font-size: 1rem;
        line-height: 1.6;
    }

    /* Layout */
    section.main > div.block-container {
        max-width: 82rem;
        padding-top: 2.4rem;
        padding-bottom: 4rem;
    }

    /* Headings — slightly larger throughout */
    h1, h2, h3, h4 {
        letter-spacing: -0.015em;
        color: var(--bsp-text);
    }
    h1 {
        font-weight: 800;
        font-size: 2.6rem;
        margin-bottom: 0.45rem;
    }
    h2 { font-weight: 700; font-size: 1.75rem; margin-top: 1.3rem; }
    h3 { font-weight: 600; font-size: 1.3rem; }
    h4 { font-weight: 600; font-size: 1.05rem; color: var(--bsp-text-muted); text-transform: uppercase; letter-spacing: 0.06em; }

    a, a:visited { color: var(--bsp-primary); text-decoration: none; }
    a:hover { color: var(--bsp-primary-hover); text-decoration: underline; }

    /* Expanders */
    [data-testid="stExpander"] {
        border-radius: 14px;
        border: 1px solid var(--bsp-border);
        background-color: var(--bsp-bg);
        box-shadow: var(--bsp-shadow-sm);
        margin-bottom: 0.85rem;
        transition: box-shadow 0.18s ease, border-color 0.18s ease;
    }
    [data-testid="stExpander"]:hover {
        border-color: #CBD5E1;
        box-shadow: var(--bsp-shadow-md);
    }
    [data-testid="stExpander"] details > summary {
        padding: 0.95rem 1.15rem;
        font-weight: 600;
        font-size: 1.02rem;
        color: var(--bsp-text);
    }
    [data-testid="stExpander"] details > summary:hover {
        background-color: var(--bsp-surface);
        border-radius: 14px 14px 0 0;
    }
    [data-testid="stExpander"] details[open] > summary {
        border-bottom: 1px solid var(--bsp-border-soft);
        border-radius: 14px 14px 0 0;
    }

    /* Tabs: underline indicator */
    [data-testid="stTabs"] [role="tablist"] {
        gap: 0.25rem;
        border-bottom: 1px solid var(--bsp-border);
    }
    [data-testid="stTabs"] [role="tab"] {
        padding: 0.6rem 1.05rem;
        color: var(--bsp-text-muted);
        font-weight: 500;
        font-size: 0.97rem;
        border-radius: 8px 8px 0 0;
        transition: color 0.15s ease, background-color 0.15s ease;
    }
    [data-testid="stTabs"] [role="tab"]:hover {
        color: var(--bsp-text);
        background-color: var(--bsp-surface);
    }
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: var(--bsp-primary);
        background-color: transparent;
        border-bottom: 2px solid var(--bsp-primary) !important;
    }

    /* Buttons */
    .stButton > button, .stDownloadButton > button {
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.97rem;
        border: 1px solid var(--bsp-border);
        background-color: var(--bsp-bg);
        color: var(--bsp-text);
        transition: all 0.15s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: var(--bsp-primary);
        color: var(--bsp-primary);
        transform: translateY(-1px);
        box-shadow: var(--bsp-shadow-md);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--bsp-primary) 0%, var(--bsp-primary-soft) 100%);
        color: #FFFFFF !important;
        border: none;
        box-shadow: var(--bsp-shadow-lg);
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, var(--bsp-primary-hover) 0%, var(--bsp-primary) 100%);
        color: #FFFFFF !important;
        transform: translateY(-1px);
    }

    /* Popovers */
    [data-testid="stPopover"] button {
        border-radius: 10px;
        background-color: var(--bsp-surface);
        border: 1px solid var(--bsp-border);
        font-weight: 500;
    }
    [data-testid="stPopover"] button:hover {
        background-color: var(--bsp-surface-2);
        border-color: var(--bsp-primary);
        color: var(--bsp-primary);
    }

    /* Inputs: focus ring */
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div,
    .stMultiSelect div[data-baseweb="select"] > div, .stTextArea textarea {
        border-radius: 9px !important;
        border-color: var(--bsp-border) !important;
        font-size: 0.97rem !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }
    .stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
        border-color: var(--bsp-primary) !important;
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.12) !important;
        outline: none !important;
    }

    /* Sliders */
    .stSlider [role="slider"] { box-shadow: 0 0 0 4px rgba(79, 70, 229, 0.15); }

    /* Radio + Checkbox */
    [data-testid="stRadio"] label, [data-testid="stCheckbox"] label {
        font-weight: 500;
    }
    [data-testid="stCheckbox"] [role="checkbox"][aria-checked="true"] {
        background-color: var(--bsp-primary) !important;
        border-color: var(--bsp-primary) !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
        border: 2px dashed var(--bsp-border) !important;
        border-radius: 12px !important;
        background-color: var(--bsp-surface) !important;
        transition: border-color 0.15s ease, background-color 0.15s ease;
    }
    [data-testid="stFileUploader"] section:hover,
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--bsp-primary) !important;
        background-color: rgba(79, 70, 229, 0.04) !important;
    }
    [data-testid="stFileUploader"] button {
        border-radius: 8px !important;
    }

    /* Sidebar — wider + nicer background.
       Width override is gated to aria-expanded="true" so streamlit's
       collapse animation (which uses a negative margin sized to the
       original width) can still hide the panel completely. */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FFFFFF 0%, var(--bsp-surface) 100%);
        border-right: 1px solid var(--bsp-border-soft);
    }
    section[data-testid="stSidebar"][aria-expanded="true"] {
        width: var(--bsp-sidebar-width) !important;
        min-width: var(--bsp-sidebar-width) !important;
    }
    section[data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: var(--bsp-sidebar-width) !important;
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 1rem;
    }

    section[data-testid="stSidebar"] hr {
        margin: 0.6rem 0 1rem;
        border-color: var(--bsp-border-soft);
    }

    /* Sidebar logo: let SVG be a bit larger */
    section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] img,
    section[data-testid="stSidebar"] [data-testid="stImage"] img {
        max-height: 56px;
        margin-bottom: 0.25rem;
    }
    [data-testid="stHeader"] [data-testid="stImage"] img {
        max-height: 36px;
    }

    /* Sidebar nav list (st.navigation) */
    section[data-testid="stSidebar"] [role="navigation"] a,
    section[data-testid="stSidebar"] nav a {
        border-radius: 10px;
        padding: 0.55rem 0.75rem;
        font-weight: 500;
        font-size: 1rem;
        margin: 0.1rem 0;
        transition: background-color 0.15s ease, color 0.15s ease;
    }
    section[data-testid="stSidebar"] [role="navigation"] a:hover,
    section[data-testid="stSidebar"] nav a:hover {
        background-color: rgba(79, 70, 229, 0.08);
        color: var(--bsp-primary);
        text-decoration: none;
    }
    section[data-testid="stSidebar"] [role="navigation"] a[aria-current="page"],
    section[data-testid="stSidebar"] nav a[aria-current="page"] {
        background: linear-gradient(135deg, rgba(79, 70, 229, 0.10) 0%, rgba(6, 182, 212, 0.10) 100%);
        color: var(--bsp-primary) !important;
        font-weight: 600;
    }

    section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] small {
        font-size: 0.85rem;
    }

    /* Dataframes / data editors */
    [data-testid="stDataFrame"], [data-testid="stDataEditor"] {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid var(--bsp-border-soft);
    }

    /* Alerts */
    [data-testid="stAlert"] {
        border-radius: 12px;
        border: 1px solid transparent;
        padding: 0.95rem 1.05rem;
        font-size: 0.97rem;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background-color: var(--bsp-surface);
        border: 1px solid var(--bsp-border-soft);
        border-radius: 12px;
        padding: 1rem 1.1rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: var(--bsp-shadow-md);
        border-color: var(--bsp-primary);
    }
    [data-testid="stMetricLabel"] p {
        color: var(--bsp-text-muted);
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    [data-testid="stMetricValue"] {
        font-weight: 700;
        color: var(--bsp-text);
        font-size: 1.55rem !important;
    }

    /* Plotly chart container */
    [data-testid="stPlotlyChart"] {
        border-radius: 10px;
        border: 1px solid var(--bsp-border-soft);
        padding: 0.4rem;
        background-color: var(--bsp-bg);
    }

    /* Status box */
    [data-testid="stStatusWidget"], [data-testid="stStatus"] {
        border-radius: 12px;
    }

    /* Hero gradient title (Home) */
    .bsp-gradient {
        background: linear-gradient(135deg, var(--bsp-primary) 0%, var(--bsp-accent) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .bsp-eyebrow {
        display: inline-block;
        padding: 0.3rem 0.75rem;
        background-color: rgba(79, 70, 229, 0.08);
        color: var(--bsp-primary);
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }
    .bsp-hero {
        padding: 2.6rem 2.6rem;
        border-radius: 20px;
        background:
            radial-gradient(circle at 0% 0%, rgba(79, 70, 229, 0.10), transparent 55%),
            radial-gradient(circle at 100% 100%, rgba(6, 182, 212, 0.10), transparent 55%),
            #FFFFFF;
        border: 1px solid var(--bsp-border-soft);
        box-shadow: var(--bsp-shadow-md);
        margin-bottom: 1.6rem;
    }
    .bsp-hero h1 {
        font-size: 3.2rem;
        line-height: 1.1;
        margin-bottom: 0.85rem;
    }
    .bsp-hero p.lead {
        color: var(--bsp-text-muted);
        font-size: 1.1rem;
        max-width: 46rem;
        margin-bottom: 1.2rem;
    }

    /* Page subtitle right below the auto page title */
    .bsp-subtitle {
        color: var(--bsp-text-muted);
        font-size: 1rem;
        margin-top: -0.6rem;
        margin-bottom: 1.4rem;
    }

    /* Sidebar tagline block */
    .bsp-tagline {
        font-size: 0.85rem;
        color: var(--bsp-text-muted);
        line-height: 1.5;
        padding: 0.2rem 0.1rem 0.4rem;
    }

    /* Empty-state card */
    .bsp-empty {
        text-align: center;
        padding: 1.6rem;
        border-radius: 14px;
        border: 1px dashed var(--bsp-border);
        background-color: var(--bsp-surface);
        color: var(--bsp-text-muted);
    }
    .bsp-empty .bsp-empty-icon { font-size: 1.8rem; margin-bottom: 0.35rem; }
    .bsp-empty .bsp-empty-title { color: var(--bsp-text); font-weight: 600; margin-bottom: 0.25rem; font-size: 1.02rem; }

    /* Inference stepper */
    .bsp-stepper {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.6rem;
        margin: 0.4rem 0 1.4rem;
    }
    .bsp-step {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        padding: 0.7rem 0.95rem;
        border-radius: 12px;
        border: 1px solid var(--bsp-border);
        background-color: var(--bsp-surface);
        color: var(--bsp-text-muted);
        transition: border-color 0.2s ease, background-color 0.2s ease, color 0.2s ease;
    }
    .bsp-step.active {
        border-color: var(--bsp-primary);
        background:
            linear-gradient(135deg, rgba(79, 70, 229, 0.10) 0%, rgba(6, 182, 212, 0.08) 100%),
            #FFFFFF;
        color: var(--bsp-text);
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.12);
    }
    .bsp-step.done {
        border-color: var(--bsp-success);
        background-color: rgba(16, 185, 129, 0.06);
        color: var(--bsp-text);
    }
    .bsp-step-num {
        flex-shrink: 0;
        width: 1.85rem;
        height: 1.85rem;
        border-radius: 999px;
        background-color: var(--bsp-bg);
        border: 1px solid var(--bsp-border);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.95rem;
        color: var(--bsp-text-muted);
    }
    .bsp-step.active .bsp-step-num {
        background: linear-gradient(135deg, var(--bsp-primary) 0%, var(--bsp-primary-soft) 100%);
        color: #FFFFFF;
        border-color: var(--bsp-primary);
    }
    .bsp-step.done .bsp-step-num {
        background-color: var(--bsp-success);
        color: #FFFFFF;
        border-color: var(--bsp-success);
    }
    .bsp-step-body { display: flex; flex-direction: column; gap: 0.05rem; min-width: 0; }
    .bsp-step-tag {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--bsp-text-muted);
    }
    .bsp-step.active .bsp-step-tag, .bsp-step.done .bsp-step-tag {
        color: var(--bsp-primary);
    }
    .bsp-step-title {
        font-size: 0.95rem;
        font-weight: 600;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* Pair badges in "Pairs detected" card */
    .bsp-pair-row {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        padding: 0.55rem 0.75rem;
        margin: 0.35rem 0;
        background-color: var(--bsp-surface);
        border: 1px solid var(--bsp-border-soft);
        border-radius: 10px;
    }
    .bsp-data-badge, .bsp-model-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        font-weight: 600;
        padding: 0.2rem 0.6rem;
        border-radius: 7px;
    }
    .bsp-data-badge {
        background-color: rgba(79, 70, 229, 0.10);
        color: var(--bsp-primary);
    }
    .bsp-model-badge {
        background-color: rgba(6, 182, 212, 0.10);
        color: var(--bsp-accent);
    }
    .bsp-pair-arrow {
        font-size: 1.1rem;
        color: var(--bsp-text-muted);
        font-weight: 700;
    }

    /* Sidebar workflow mini-stepper */
    .bsp-mini-stepper-head {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        color: var(--bsp-text-muted);
        margin: 0.4rem 0.1rem 0.5rem;
    }
    .bsp-mini-stepper {
        display: flex;
        flex-direction: column;
        gap: 0.35rem;
        margin-bottom: 0.6rem;
    }
    .bsp-mini-step {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        padding: 0.45rem 0.6rem;
        border-radius: 10px;
        border: 1px solid var(--bsp-border-soft);
        background-color: var(--bsp-bg);
        transition: border-color 0.15s ease, background-color 0.15s ease;
    }
    .bsp-mini-step.done {
        border-color: rgba(16, 185, 129, 0.30);
        background-color: rgba(16, 185, 129, 0.05);
    }
    .bsp-mini-step.active {
        border-color: var(--bsp-primary);
        background:
            linear-gradient(135deg, rgba(79, 70, 229, 0.08) 0%, rgba(6, 182, 212, 0.06) 100%),
            #FFFFFF;
        animation: bsp-mini-pulse 1.8s ease-in-out infinite;
    }
    @keyframes bsp-mini-pulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(79, 70, 229, 0.30); }
        50%      { box-shadow: 0 0 0 6px rgba(79, 70, 229, 0.00); }
    }
    .bsp-mini-emoji { font-size: 1.05rem; line-height: 1; }
    .bsp-mini-body {
        display: flex;
        flex-direction: column;
        gap: 0.05rem;
        flex: 1;
        min-width: 0;
    }
    .bsp-mini-title {
        font-size: 0.86rem;
        font-weight: 600;
        color: var(--bsp-text);
        line-height: 1.2;
    }
    .bsp-mini-caption {
        font-size: 0.72rem;
        color: var(--bsp-text-muted);
        line-height: 1.2;
        font-family: 'JetBrains Mono', monospace;
    }
    .bsp-mini-dot {
        font-size: 0.95rem;
        color: var(--bsp-border);
    }
    .bsp-mini-step.done .bsp-mini-dot { color: var(--bsp-success); }
    .bsp-mini-step.active .bsp-mini-dot { color: var(--bsp-primary); }
    .bsp-mini-step.active .bsp-mini-title { color: var(--bsp-primary); }

    /* Page header block (eyebrow + subtitle pair used on Data/Model/Infer) */
    .bsp-page-eyebrow {
        display: inline-block;
        padding: 0.22rem 0.7rem;
        background-color: rgba(79, 70, 229, 0.08);
        color: var(--bsp-primary);
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        margin: -0.4rem 0 0.55rem;
    }

    /* Small count pill, used by section heads */
    .bsp-count-pill {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 1.7rem;
        height: 1.7rem;
        padding: 0 0.55rem;
        border-radius: 999px;
        background: linear-gradient(135deg, var(--bsp-primary) 0%, var(--bsp-accent) 100%);
        color: #FFFFFF;
        font-weight: 700;
        font-size: 0.85rem;
        line-height: 1;
        margin-left: 0.3rem;
    }
</style>
"""


st.set_page_config(
    page_title='BaySpec',
    page_icon='.streamlit/logo_mark.svg',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

st.logo(
    '.streamlit/logo.svg',
    icon_image='.streamlit/logo_mark.svg',
    size='large',
)

# Initialise session state before any page or sidebar reads it.
init_session_state()

with st.sidebar:
    st.markdown(
        '<div class="bsp-tagline">Bayesian spectral fitting workbench '
        'for high-energy astrophysics.</div>',
        unsafe_allow_html=True,
    )
    render_workflow_sidebar()
    st.divider()

    st.markdown(
        '<div class="bsp-mini-stepper-head">Configuration</div>',
        unsafe_allow_html=True,
    )
    st.download_button(
        '⬇️  Save config (.json)',
        data=export_config_bytes(),
        file_name='bayspec_config.json',
        mime='application/json',
        use_container_width=True,
        key='cfg_download',
        help='Bundle every UI choice (notc, stat, grpg, priors, sampler '
        'settings…) into a JSON file. FITS uploads are not included; '
        'attach them again after loading.',
    )
    uploaded_cfg = st.file_uploader(
        'Load config (.json)',
        type=['json'],
        key='cfg_upload',
        label_visibility='collapsed',
    )
    if uploaded_cfg is not None and st.session_state.get('_last_cfg') != uploaded_cfg.name:
        ok, msg = import_config_bytes(uploaded_cfg.getvalue())
        st.session_state['_last_cfg'] = uploaded_cfg.name
        if ok:
            st.success(msg, icon='✅')
            st.rerun()
        else:
            st.error(msg, icon='🚨')

    st.divider()

nav = get_nav_from_toml('.streamlit/pages.toml')

pg = st.navigation(nav)

add_page_title(pg)

pg.run()
