import asyncio
import re
import threading
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import state

app = FastAPI(title="BaySpec")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
templates.env.globals["zip"] = zip

from bayspec.model.local import local_models as _local_models

templates.env.globals["local_model_names"] = list(_local_models.keys())


def _label(text: str) -> str:
    """``UNIT_META`` / ``CountsSpectrum NCHAN`` → Title Case for display."""
    out = text.replace("_", " ")
    out = re.sub(r"([a-z])([A-Z])", r"\1 \2", out)
    return " ".join(w.capitalize() for w in out.split())


templates.env.globals["_label"] = _label

SESSION_COOKIE = "bsp_session"
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

_tasks: dict[str, dict] = {}  # task_id → {status, messages, result_html, error}


def _session(request: Request) -> tuple[str, dict, bool]:
    """Return (session_id, session_dict, is_new)."""
    sid = request.cookies.get(SESSION_COOKIE)
    is_new = sid is None
    if is_new:
        sid = state.new_id()
    return sid, state.get(sid), is_new


def _render(name: str, request: Request, **ctx):
    sid, s, is_new = _session(request)
    resp = templates.TemplateResponse(
        request=request, name=name, context={"session_id": sid, "s": s, **ctx}
    )
    if is_new:
        resp.set_cookie(SESSION_COOKIE, sid, httponly=True, samesite="lax")
    return resp


def _partial(name: str, request: Request, **ctx):
    sid, s, is_new = _session(request)
    resp = templates.TemplateResponse(
        request=request, name=name, context={"s": s, **ctx}
    )
    if is_new:
        resp.set_cookie(SESSION_COOKIE, sid, httponly=True, samesite="lax")
    return resp


def _safe_key(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", key)[:64]


def _parse_notc(notc_str: str):
    """Parse Streamlit-style ``"8-30;40-1000"`` → list of [lo, hi]."""
    s = notc_str.strip()
    if not s:
        return None
    windows = [w.strip() for w in s.split(";") if w.strip()]
    parsed = []
    for w in windows:
        rng = [x.strip() for x in w.split("-")]
        if len(rng) != 2:
            raise ValueError(f"Each window must be 'lo-hi', got: {w!r}")
        parsed.append([float(rng[0]), float(rng[1])])
    return parsed[0] if len(parsed) == 1 else parsed


def _parse_optional_int(val: Optional[str]) -> Optional[int]:
    if val is None or val == "":
        return None
    return int(float(val))


def _parse_optional_float(val: Optional[str]) -> Optional[float]:
    if val is None or val == "":
        return None
    return float(val)


def _build_grpg_rebn(min_evt, min_sigma, max_bin) -> Optional[dict]:
    if min_evt is None and min_sigma is None and max_bin is None:
        return None
    return {"min_evt": min_evt, "min_sigma": min_sigma, "max_bin": max_bin}


def _classify_spec_file(filename: str) -> Optional[str]:
    """Map a filename to one of src/bkg/rsp/rmf/arf based on substring match."""
    n = filename.lower()
    if "rmf" in n:
        return "rmf"
    if "arf" in n:
        return "arf"
    if "rsp" in n or "resp" in n:
        return "rsp"
    if "bkg" in n or "bak" in n:
        return "bkg"
    if "src" in n or "pha" in n:
        return "src"
    return None


def _counts_plot_div(unit) -> str:
    """Counts spectrum (CE style): src + bkg + net as point + x/y error bars,
    matching bayspec's Plot.dataunit visual."""
    import plotly.graph_objects as go
    import plotly.offline as pyo

    x = unit.rsp_chbin_mean.astype(float)
    half_w = unit.rsp_chbin_width.astype(float) / 2

    def _err_x():
        return dict(
            type="data",
            symmetric=False,
            array=half_w,
            arrayminus=half_w,
            thickness=1.2,
            width=0,
        )

    def _err_y(arr):
        return dict(type="data", array=arr, thickness=1.2, width=0)

    fig = go.Figure()

    src_y = unit.src_ctsspec.astype(float)
    src_e = unit.src_ctsspec_error.astype(float)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=src_y,
            mode="markers",
            name="Source",
            error_x=_err_x(),
            error_y=_err_y(src_e),
            marker=dict(symbol="circle", size=3, color="#4F46E5"),
        )
    )

    try:
        bkg_y = unit.bkg_ctsspec.astype(float)
        bkg_e = unit.bkg_ctsspec_error.astype(float)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=bkg_y,
                mode="markers",
                name="Background",
                error_x=_err_x(),
                error_y=_err_y(bkg_e),
                marker=dict(symbol="circle", size=3, color="#94A3B8"),
            )
        )
    except Exception:
        pass

    try:
        net_y = unit.net_ctsspec.astype(float)
        net_e = unit.net_ctsspec_error.astype(float)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=net_y,
                mode="markers",
                name="Net",
                error_x=_err_x(),
                error_y=_err_y(net_e),
                marker=dict(symbol="circle", size=3, color="#10B981"),
            )
        )
    except Exception:
        pass

    fig.update_layout(
        xaxis=dict(
            title="Energy (keV)", type="log", showgrid=True, gridcolor="#F1F5F9"
        ),
        yaxis=dict(
            title="Counts s⁻¹ keV⁻¹", type="log", showgrid=True, gridcolor="#F1F5F9"
        ),
        template="simple_white",
        margin=dict(l=65, r=20, t=20, b=50),
        height=460,
        showlegend=True,
        legend=dict(x=0.98, y=0.98, xanchor="right", yanchor="top"),
        font=dict(family="Inter, system-ui, sans-serif", size=12, color="#0F172A"),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
    )

    return pyo.plot(fig, output_type="div", include_plotlyjs=False)


# ── Page routes ────────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return _render("home.html", request)


@app.get("/data", response_class=HTMLResponse)
async def data_page(request: Request):
    return _render("data.html", request)


@app.get("/model", response_class=HTMLResponse)
async def model_page(request: Request):
    return _render("model.html", request)


@app.get("/infer", response_class=HTMLResponse)
async def infer_page(request: Request):
    return _render("infer.html", request)


@app.get("/editor", response_class=HTMLResponse)
async def editor_page(request: Request):
    return _render("editor.html", request)


# ── Data helpers ───────────────────────────────────────────────────────────────


def _render_container_list_s(s: dict, request: Request):
    return templates.TemplateResponse(
        request=request,
        name="partials/data_container_list.html",
        context={"s": s},
    )


def _render_container_s(data_key: str, s: dict, request: Request):
    return templates.TemplateResponse(
        request=request,
        name="partials/data_container.html",
        context={"s": s, "data_key": data_key},
    )


# ── Data API routes ────────────────────────────────────────────────────────────


@app.post("/data/containers", response_class=HTMLResponse)
async def create_container(request: Request, data_key: str = Form("")):
    sid, s, is_new = _session(request)
    requested = _safe_key(data_key.strip())
    if not requested:
        n = len(s["data"]) + 1
        while f"Data{n}" in s["data"]:
            n += 1
        requested = f"Data{n}"
    if requested in s["data"]:
        # Idempotent: silently ignore duplicate creation
        resp = _render_container_list_s(s, request)
    else:
        from bayspec.data.data import Data

        d = Data()
        d.data = (
            d.data
        )  # trigger _update to set .names/.srcs/… (bayspec 0.3.11 init nuance)
        s["data"][requested] = d
        s["data_state"][requested] = {"model_binding": None, "units": {}}
        resp = _render_container_list_s(s, request)
    if is_new:
        resp.set_cookie(SESSION_COOKIE, sid, httponly=True, samesite="lax")
    return resp


@app.delete("/data/containers/{data_key}", response_class=HTMLResponse)
async def delete_container(data_key: str, request: Request):
    sid, s, _ = _session(request)
    s["data"].pop(data_key, None)
    s["data_state"].pop(data_key, None)

    import shutil

    container_dir = UPLOAD_DIR / sid / data_key
    if container_dir.exists():
        shutil.rmtree(container_dir, ignore_errors=True)

    return _render_container_list_s(s, request)


@app.post("/data/containers/{data_key}/bind", response_class=HTMLResponse)
async def bind_model(data_key: str, request: Request, model_key: str = Form("")):
    _, s, _ = _session(request)
    if data_key not in s["data_state"]:
        s["data_state"][data_key] = {"model_binding": None, "units": {}}
    s["data_state"][data_key]["model_binding"] = model_key or None
    return _render_container_s(data_key, s, request)


@app.post("/data/containers/{data_key}/units", response_class=HTMLResponse)
async def add_unit_to_container(
    data_key: str,
    request: Request,
    unit_key: str = Form(""),
    spec_files: list[UploadFile] = File([]),
    src: Optional[UploadFile] = File(None),
    bkg: Optional[UploadFile] = File(None),
    rsp: Optional[UploadFile] = File(None),
    rmf: Optional[UploadFile] = File(None),
    arf: Optional[UploadFile] = File(None),
    stat: str = Form("pgstat"),
    notc_str: str = Form(""),
    grpg_min_evt: Optional[str] = Form(None),
    grpg_min_sigma: Optional[str] = Form(None),
    grpg_max_bin: Optional[str] = Form(None),
    rebn_min_evt: Optional[str] = Form(None),
    rebn_min_sigma: Optional[str] = Form(None),
    rebn_max_bin: Optional[str] = Form(None),
    time: Optional[str] = Form(None),
):
    sid, s, _ = _session(request)
    if data_key not in s["data"]:
        return HTMLResponse("<p class='alert alert-warning'>Container not found.</p>")
    container = s["data"][data_key]
    dst = s["data_state"].setdefault(data_key, {"model_binding": None, "units": {}})

    requested = _safe_key(unit_key.strip())
    if not requested:
        n = len(container.data) + 1
        while f"unit{n}" in container.data:
            n += 1
        requested = f"unit{n}"

    unit_dir = UPLOAD_DIR / sid / data_key / requested
    unit_dir.mkdir(parents=True, exist_ok=True)

    async def _save(upload: Optional[UploadFile]) -> Optional[str]:
        if upload is None or not upload.filename:
            return None
        path = unit_dir / upload.filename
        path.write_bytes(await upload.read())
        return str(path)

    paths: dict[str, Optional[str]] = {
        "src": None,
        "bkg": None,
        "rsp": None,
        "rmf": None,
        "arf": None,
    }

    # Batch upload: classify by filename
    for f in spec_files or []:
        if not f or not f.filename:
            continue
        kind = _classify_spec_file(f.filename)
        if kind and paths[kind] is None:
            paths[kind] = await _save(f)

    # Per-slot uploads override / fill remaining
    for kind, upload in (
        ("src", src),
        ("bkg", bkg),
        ("rsp", rsp),
        ("rmf", rmf),
        ("arf", arf),
    ):
        saved = await _save(upload)
        if saved is not None:
            paths[kind] = saved

    # Build form-state mirror (so the UI can show what was submitted)
    form_state = {
        "src_path": paths["src"],
        "bkg_path": paths["bkg"],
        "rsp_path": paths["rsp"],
        "rmf_path": paths["rmf"],
        "arf_path": paths["arf"],
        "stat": stat,
        "notc_str": notc_str,
        "grpg_min_evt": grpg_min_evt,
        "grpg_min_sigma": grpg_min_sigma,
        "grpg_max_bin": grpg_max_bin,
        "rebn_min_evt": rebn_min_evt,
        "rebn_min_sigma": rebn_min_sigma,
        "rebn_max_bin": rebn_max_bin,
        "time": time,
        "error": None,
    }

    if paths["src"] is None:
        form_state["error"] = "Source file (src) is required."
        dst["units"][requested] = form_state
        return _render_container_s(data_key, s, request)

    try:
        notc = _parse_notc(notc_str)
        grpg = _build_grpg_rebn(
            _parse_optional_int(grpg_min_evt),
            _parse_optional_float(grpg_min_sigma),
            _parse_optional_int(grpg_max_bin),
        )
        rebn = _build_grpg_rebn(
            _parse_optional_int(rebn_min_evt),
            _parse_optional_float(rebn_min_sigma),
            _parse_optional_int(rebn_max_bin),
        )
        time_val = _parse_optional_float(time)

        from bayspec.data.data import DataUnit

        du = DataUnit(
            src=paths["src"],
            bkg=paths["bkg"],
            rsp=paths["rsp"],
            rmf=paths["rmf"],
            arf=paths["arf"],
            stat=stat,
            notc=notc,
            grpg=grpg,
            rebn=rebn,
            time=time_val,
        )
        du.name = requested
        container[requested] = du
    except Exception as exc:
        form_state["error"] = str(exc)

    dst["units"][requested] = form_state
    return _render_container_s(data_key, s, request)


@app.delete("/data/containers/{data_key}/units/{unit_key}", response_class=HTMLResponse)
async def delete_unit_from_container(data_key: str, unit_key: str, request: Request):
    sid, s, _ = _session(request)
    container = s["data"].get(data_key)
    if container is not None and unit_key in container:
        del container[unit_key]
    s["data_state"].get(data_key, {}).get("units", {}).pop(unit_key, None)

    import shutil

    unit_dir = UPLOAD_DIR / sid / data_key / unit_key
    if unit_dir.exists():
        shutil.rmtree(unit_dir, ignore_errors=True)

    return _render_container_s(data_key, s, request)


@app.get(
    "/data/containers/{data_key}/units/{unit_key}/plot", response_class=HTMLResponse
)
async def unit_plot(data_key: str, unit_key: str, request: Request):
    _, s, _ = _session(request)
    container = s["data"].get(data_key)
    du = container.data.get(unit_key) if container is not None else None
    if du is None:
        return HTMLResponse("<p class='alert alert-warning'>Unit not found.</p>")
    try:
        return HTMLResponse(_counts_plot_div(du))
    except Exception as exc:
        return HTMLResponse(f"<p class='alert alert-danger'>Plot error: {exc}</p>")


# ── Model helpers ──────────────────────────────────────────────────────────────


def _parse_prior_str(s: str):
    """Return (prior_obj, frozen). frozen=True if s=='frozen'."""
    from bayspec.util.prior import all_priors

    s = s.strip()
    if s == "frozen":
        return None, True
    m = re.match(r"^(\w+)\((.+)\)$", s)
    if not m:
        raise ValueError(f"Cannot parse prior: {s!r}")
    name, args_str = m.group(1), m.group(2)
    if name not in all_priors:
        raise ValueError(f"Unknown prior kind: {name!r}")
    args = [float(x.strip()) for x in args_str.split(",")]
    return all_priors[name](*args), False


_PLOT_STYLE_LABEL = {
    "vFv": "E² N(E)  (keV² photons s⁻¹ cm⁻² keV⁻¹)",
    "Fv": "E N(E)  (keV photons s⁻¹ cm⁻² keV⁻¹)",
    "NE": "N(E)  (photons s⁻¹ cm⁻² keV⁻¹)",
    "NoU": "func(E)",
}


def _apply_app_layout(
    fig, *, style: str, height: int = 320, showlegend: bool = True
) -> None:
    """Re-skin a bayspec.Plot figure to match the rest of the app."""
    fig.update_xaxes(showgrid=True, gridcolor="#F1F5F9")
    fig.update_yaxes(
        title_text=_PLOT_STYLE_LABEL.get(style, "func(E)"),
        showgrid=True,
        gridcolor="#F1F5F9",
    )
    fig.update_layout(
        template="simple_white",
        margin=dict(l=70, r=20, t=20, b=50),
        height=height,
        width=None,
        showlegend=showlegend,
        legend=dict(x=0.98, y=0.98, xanchor="right", yanchor="top"),
        font=dict(family="Inter, system-ui, sans-serif", size=12, color="#0F172A"),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
    )


def _build_model_plot(
    components: list,
    style: str,
    e_lo: float,
    e_hi: float,
    tarr_val: Optional[float],
    *,
    post: bool = False,
    height: int = 320,
) -> str:
    """Render one or more components as a ``ModelPlot`` div."""
    import numpy as np
    import plotly.offline as pyo
    from bayspec.util.plot import Plot

    E = np.logspace(float(e_lo), float(e_hi), 300)

    mp = Plot.model(style=style, post=post)
    for comp in components:
        T = (
            (tarr_val * np.ones_like(E))
            if (tarr_val is not None and getattr(comp, "type", None) == "add")
            else None
        )
        mp.add_model(comp, E, T)

    fig = mp.get_fig().fig
    _apply_app_layout(fig, style=style, height=height, showlegend=True)
    return pyo.plot(fig, output_type="div", include_plotlyjs=False)


def _render_model_card(mkey: str, s: dict, request: Request):
    return templates.TemplateResponse(
        request=request,
        name="partials/model_card.html",
        context={"s": s, "mkey": mkey},
    )


def _render_model_list(s: dict, request: Request):
    return templates.TemplateResponse(
        request=request,
        name="partials/model_list.html",
        context={"s": s},
    )


def _get_library_models(library: str, s: dict) -> tuple[dict, str]:
    """Return ({model_name: model_class}, error_message)."""
    if library == "local":
        from bayspec.model.local import local_models

        return local_models, ""
    if library == "astro":
        try:
            from bayspec.model.astro import astro_models

            return astro_models, ""
        except Exception as exc:  # noqa: BLE001
            return {}, f"Astromodels unavailable ({exc.__class__.__name__})."
    if library == "xspec":
        try:
            from bayspec.model.xspec import xspec_models

            return xspec_models, ""
        except Exception as exc:  # noqa: BLE001
            return {}, f"XSPEC unavailable ({exc.__class__.__name__})."
    if library == "user":
        return s.get("custom_models", {}), ""
    return {}, f"Unknown library: {library!r}"


def _component_plot_div(
    comp,
    style: str = "vFv",
    e_lo: float = 0.0,
    e_hi: float = 4.0,
    tarr_val: Optional[float] = None,
) -> str:
    """Spectrum of a single component on a log E grid, in the requested style."""
    return _build_model_plot(
        [comp], style, e_lo, e_hi, tarr_val, post=False, height=320
    )


# ── Model API routes ───────────────────────────────────────────────────────────


@app.post("/model/models", response_class=HTMLResponse)
async def create_model(request: Request, model_key: str = Form("")):
    sid, s, is_new = _session(request)
    requested = _safe_key(model_key.strip())
    if not requested:
        n = len(s["model_component"]) + 1
        while f"Model{n}" in s["model_component"]:
            n += 1
        requested = f"Model{n}"
    if requested not in s["model_component"]:
        s["model_component"][requested] = {}
        s["model_state"][requested] = {"expression": "", "error": None}
    resp = _render_model_list(s, request)
    if is_new:
        resp.set_cookie(SESSION_COOKIE, sid, httponly=True, samesite="lax")
    return resp


@app.delete("/model/models/{mkey}", response_class=HTMLResponse)
async def delete_model(mkey: str, request: Request):
    sid, s, is_new = _session(request)
    s["model_component"].pop(mkey, None)
    s["model_state"].pop(mkey, None)
    s["model"].pop(mkey, None)
    resp = _render_model_list(s, request)
    if is_new:
        resp.set_cookie(SESSION_COOKIE, sid, httponly=True, samesite="lax")
    return resp


XSPEC_ABUND_CHOICES = ["angr", "aspl", "feld", "aneb", "grsa", "wilm", "lodd", "lpgp"]
XSPEC_XSECT_CHOICES = ["bcmc", "obcm", "vern"]


@app.post("/model/models/{mkey}/components", response_class=HTMLResponse)
async def add_component(
    mkey: str,
    request: Request,
    library: str = Form("local"),
    comp_type: str = Form(...),
    comp_key: str = Form(""),
    xspec_abund: str = Form(""),
    xspec_xsect: str = Form(""),
):
    _, s, _ = _session(request)
    ckey = _safe_key(comp_key.strip()) if comp_key.strip() else comp_type

    lib_dict, lib_err = _get_library_models(library, s)
    if lib_err:
        s["model_state"].setdefault(mkey, {})["error"] = lib_err
        return _render_model_card(mkey, s, request)
    if comp_type not in lib_dict:
        s["model_state"].setdefault(mkey, {})["error"] = (
            f"Unknown model {comp_type!r} in library {library!r}"
        )
        return _render_model_card(mkey, s, request)

    # Apply XSPEC global state before instantiating the component.
    if library == "xspec":
        try:
            from bayspec.model.xspec import abund as _abund
            from bayspec.model.xspec import xsect as _xsect

            if xspec_abund in XSPEC_ABUND_CHOICES:
                _abund(xspec_abund)
                s["model_state"].setdefault(mkey, {})["xspec_abund"] = xspec_abund
            if xspec_xsect in XSPEC_XSECT_CHOICES:
                _xsect(xspec_xsect)
                s["model_state"].setdefault(mkey, {})["xspec_xsect"] = xspec_xsect
        except Exception as exc:
            s["model_state"].setdefault(mkey, {})["error"] = (
                f"XSPEC config error: {exc}"
            )
            return _render_model_card(mkey, s, request)

    try:
        comp = lib_dict[comp_type]()
    except Exception as exc:
        s["model_state"].setdefault(mkey, {})["error"] = (
            f"Failed to instantiate {comp_type}: {exc}"
        )
        return _render_model_card(mkey, s, request)

    s["model_component"].setdefault(mkey, {})[ckey] = comp
    s["model_state"].setdefault(mkey, {})["error"] = None
    return _render_model_card(mkey, s, request)


@app.get("/model/libraries/{library}/options", response_class=HTMLResponse)
async def library_options(library: str, request: Request):
    _, s, _ = _session(request)
    lib_dict, err = _get_library_models(library, s)
    return templates.TemplateResponse(
        request=request,
        name="partials/library_options.html",
        context={"library": library, "models": lib_dict, "error": err},
    )


@app.get("/model/xspec_options", response_class=HTMLResponse)
async def xspec_options(request: Request):
    """Return abund + xsect dropdowns (visible only when library==xspec)."""
    return templates.TemplateResponse(
        request=request,
        name="partials/xspec_options.html",
        context={
            "abund_choices": XSPEC_ABUND_CHOICES,
            "xsect_choices": XSPEC_XSECT_CHOICES,
        },
    )


@app.get("/model/xspec_options/empty", response_class=HTMLResponse)
async def xspec_options_empty():
    return HTMLResponse("")


@app.post("/model/models/{mkey}/bind", response_class=HTMLResponse)
async def bind_data(mkey: str, request: Request, data_key: str = Form("")):
    _, s, _ = _session(request)
    s["model_state"].setdefault(mkey, {})["data_binding"] = data_key or None
    return _render_model_card(mkey, s, request)


@app.get("/model/models/{mkey}/components/{ckey}/plot", response_class=HTMLResponse)
async def component_plot(
    mkey: str,
    ckey: str,
    request: Request,
    style: str = "vFv",
    e_lo: float = 0.0,
    e_hi: float = 4.0,
    time: str = "",
):
    _, s, _ = _session(request)
    comp = s["model_component"].get(mkey, {}).get(ckey)
    if comp is None:
        return HTMLResponse("<p class='alert alert-warning'>Component not found.</p>")
    try:
        tarr_val = _parse_optional_float(time)
        return HTMLResponse(_component_plot_div(comp, style, e_lo, e_hi, tarr_val))
    except Exception as exc:
        return HTMLResponse(f"<p class='alert alert-danger'>Plot error: {exc}</p>")


@app.get("/model/models/{mkey}/components/spectra", response_class=HTMLResponse)
async def model_components_spectra(
    mkey: str,
    request: Request,
    style: str = "vFv",
    e_lo: float = 0.0,
    e_hi: float = 4.0,
    comps: str = "",
    time: str = "",
):
    """Multi-component combined spectra preview (pre-inference)."""
    _, s, _ = _session(request)
    comp_keys = [c.strip() for c in comps.split(",") if c.strip()]
    components = s["model_component"].get(mkey, {})
    selected = [components[ck] for ck in comp_keys if ck in components]
    if not selected:
        return HTMLResponse(
            "<p class='alert alert-warning'>Select at least one component.</p>"
        )
    try:
        tarr_val = _parse_optional_float(time)
        return HTMLResponse(
            _build_model_plot(
                selected,
                style,
                e_lo,
                e_hi,
                tarr_val,
                post=False,
                height=380,
            )
        )
    except Exception as exc:
        return HTMLResponse(f"<p class='alert alert-danger'>Spectra error: {exc}</p>")


@app.delete("/model/models/{mkey}/components/{ckey}", response_class=HTMLResponse)
async def delete_component(mkey: str, ckey: str, request: Request):
    _, s, _ = _session(request)
    s["model_component"].get(mkey, {}).pop(ckey, None)
    s["model"].pop(mkey, None)  # invalidate composed model
    return _render_model_card(mkey, s, request)


@app.post("/model/models/{mkey}/components/{ckey}/update", response_class=HTMLResponse)
async def update_component(mkey: str, ckey: str, request: Request):
    form = await request.form()
    _, s, _ = _session(request)
    comp = s["model_component"].get(mkey, {}).get(ckey)
    if comp is None:
        return _render_model_card(mkey, s, request)

    cfg_dict = comp.cfg_info.data_dict
    for idx, param, orig_val in zip(
        cfg_dict["cfg#"], cfg_dict["Parameter"], cfg_dict["Value"]
    ):
        field = f"cfg_{idx}"
        if field not in form:
            continue
        raw = form[field]
        try:
            if isinstance(orig_val, bool):
                new_val = raw == "true"
            elif isinstance(orig_val, int):
                new_val = int(float(raw))
            else:
                new_val = float(raw)
            comp.config[param]._val = new_val
        except (ValueError, TypeError):
            pass

    par_dict = comp.par_info.data_dict
    for idx, param, orig_prior in zip(
        par_dict["par#"], par_dict["Parameter"], par_dict["Prior"]
    ):
        val_field = f"par_val_{idx}"
        prior_field = f"par_prior_{idx}"
        if val_field in form:
            try:
                comp.params[param].val = float(form[val_field])
            except (ValueError, TypeError):
                pass
        if prior_field in form:
            new_prior = form[prior_field].strip()
            if new_prior and new_prior != orig_prior:
                try:
                    prior_obj, frozen = _parse_prior_str(new_prior)
                    comp.params[param].frozen = frozen
                    if not frozen:
                        comp.params[param].prior = prior_obj
                except (ValueError, KeyError):
                    pass

    return _render_model_card(mkey, s, request)


@app.post("/model/models/{mkey}/compose", response_class=HTMLResponse)
async def compose_model(
    mkey: str,
    request: Request,
    expression: str = Form(...),
):
    _, s, _ = _session(request)
    components = s["model_component"].get(mkey, {})
    s["model_state"].setdefault(mkey, {})["expression"] = expression
    error = None
    try:
        composed = eval(expression, {"__builtins__": {}}, dict(components))  # noqa: S307
        s["model"][mkey] = composed
        s["model_state"][mkey]["error"] = None
    except Exception as exc:
        error = str(exc)
        s["model_state"][mkey]["error"] = error
    return _render_model_card(mkey, s, request)


@app.get("/model/models/{mkey}/plot", response_class=HTMLResponse)
async def model_plot(
    mkey: str,
    request: Request,
    style: str = "vFv",
    comps: str = "",
    e_lo: float = 0.0,
    e_hi: float = 4.0,
    time: str = "",
):
    _, s, _ = _session(request)
    if s["model"].get(mkey) is None:
        return HTMLResponse(
            "<p class='alert alert-warning'>Compose the model first.</p>"
        )

    all_comps = s["model_component"].get(mkey, {})
    selected = [c.strip() for c in comps.split(",") if c.strip()]
    plot_composed = "*" in selected
    if plot_composed:
        selected = list(all_comps.keys())
    if not selected:
        return HTMLResponse(
            "<p class='alert alert-warning'>Pick at least one component to plot.</p>"
        )

    add_styles = {"vFv", "Fv", "NE"}
    nou_styles = {"NoU"}

    def _style_ok(comp) -> bool:
        t = getattr(comp, "type", None)
        if style in add_styles:
            return t == "add"
        if style in nou_styles:
            return t in ("mul", "math")
        return True

    chosen = [
        all_comps[c] for c in selected if c in all_comps and _style_ok(all_comps[c])
    ]
    if plot_composed:
        composed_model = s["model"].get(mkey)
        if composed_model and _style_ok(composed_model):
            chosen.insert(0, composed_model)
    if not chosen:
        return HTMLResponse(
            f"<p class='alert alert-warning'>No components compatible with style '{style}'. "
            "vFv / Fv / NE need additive components; NoU needs mul / math.</p>"
        )

    try:
        tarr_val = _parse_optional_float(time)
        return HTMLResponse(
            _build_model_plot(
                chosen, style, e_lo, e_hi, tarr_val, post=False, height=380
            )
        )
    except Exception as exc:
        return HTMLResponse(f"<p class='alert alert-danger'>Plot error: {exc}</p>")


# ── Inference helpers ──────────────────────────────────────────────────────────


def _derived_pairs(s: dict) -> list[dict]:
    """Scan data↔model bindings (unidirectional is sufficient) to derive pairs."""
    pairs = []
    seen = set()
    # Direction 1: data → model
    for dk, dst in s.get("data_state", {}).items():
        mk = dst.get("model_binding")
        if mk and mk in s.get("model", {}):
            key = (dk, mk)
            if key not in seen:
                seen.add(key)
                pairs.append({"data": dk, "model": mk})
    # Direction 2: model → data (catch any not covered above)
    for mk, mst in s.get("model_state", {}).items():
        dk = mst.get("data_binding")
        if dk and dk in s.get("data", {}) and mk in s.get("model", {}):
            key = (dk, mk)
            if key not in seen:
                seen.add(key)
                pairs.append({"data": dk, "model": mk})
    return pairs


def _posterior_html(post) -> str:
    """Return HTML fragment with parameter CI table + stat + IC tables."""
    def _fmt(v, fmt=".4g"):
        return format(v, fmt) if v is not None else "—"

    fp = post.free_par_info.data_dict
    par_rows = "".join(
        f"<tr><td class='param-name'>{par}</td>"
        f"<td>{_fmt(best)}</td>"
        f"<td><code>{ci}</code></td>"
        f"<td>{_fmt(mean)}</td>"
        f"<td>{_fmt(med)}</td></tr>"
        for par, best, ci, mean, med in zip(
            fp["Parameter"], fp["Best"], fp["1sigma CI"], fp["Mean"], fp["Median"]
        )
    )
    par_html = (
        "<div class='posterior-main'>"
        "<div class='param-section-label' style='margin-bottom:.4rem'>Parameters</div>"
        "<table class='param-table'>"
        "<thead><tr><th>Parameter</th><th>Best</th><th>1σ CI</th>"
        "<th>Mean</th><th>Median</th></tr></thead>"
        f"<tbody>{par_rows}</tbody></table>"
        "</div>"
    )

    si = post.stat_info.data_dict
    stat_rows = "".join(
        f"<tr><td class='param-name'>{d}</td><td>{m}</td>"
        f"<td>{stat}</td><td>{v}</td><td>{b}</td></tr>"
        for d, m, stat, v, b in zip(
            si["Data"], si["Model"], si["Statistic"], si["Value"], si["Bins"]
        )
    )
    stat_html = (
        "<div>"
        "<div class='param-section-label' style='margin-bottom:.4rem'>Statistics</div>"
        "<table class='param-table'>"
        "<thead><tr><th>Data</th><th>Model</th><th>Statistic</th>"
        "<th>Value</th><th>Bins</th></tr></thead>"
        f"<tbody>{stat_rows}</tbody></table>"
        "</div>"
    )

    ic = post.IC_info.data_dict
    ic_rows = "".join(
        f"<tr><td class='param-name'>{k}</td><td>{_fmt(ic[k][0])}</td></tr>" for k in ic
    )
    ic_html = (
        "<div>"
        "<div class='param-section-label' style='margin-bottom:.4rem'>Information Criteria</div>"
        "<table class='param-table'>"
        "<thead><tr><th>Criteria</th><th>Value</th></tr></thead>"
        f"<tbody>{ic_rows}</tbody></table>"
        "</div>"
    )

    return par_html + f"<div class='posterior-side'>{stat_html}{ic_html}</div>"


def _render_infer_panel(s: dict, request: Request):
    return templates.TemplateResponse(
        request=request,
        name="partials/infer_panel.html",
        context={"s": s},
    )


def _render_infer_panel_str(s: dict) -> str:
    return templates.env.get_template("partials/infer_panel.html").render(s=s)


def _model_spectra_div(
    post,
    mkey: str,
    style: str,
    comp_keys: list[str],
    s: dict,
    e_lo: float = 0.0,
    e_hi: float = 4.0,
    tarr_val: Optional[float] = None,
) -> str:
    """Post-fit component spectra rendered via ``Plot.model(post=True)``."""
    post.at_par(post.par_best)
    components = s["model_component"].get(mkey, {})

    comps = []
    # Non-string entries pass through directly (e.g. composed model object)
    for ck in comp_keys:
        if isinstance(ck, str):
            if ck in components:
                comps.append(components[ck])
        else:
            comps.append(ck)

    if not comps:
        return "<p class='alert alert-warning'>No matching components.</p>"
    return _build_model_plot(comps, style, e_lo, e_hi, tarr_val, post=True, height=380)


def _corner_plot_div(post) -> str:
    import numpy as np
    import plotly.graph_objects as go
    import plotly.offline as pyo

    samples = post.param_sample  # (n_samples, n_free_params)
    param_names = post.free_par_info.data_dict["Parameter"]
    n = len(param_names)
    if n == 0:
        return "<p class='alert alert-warning'>No free parameters for corner plot.</p>"

    if len(samples) > 2000:
        idx = np.random.default_rng(0).choice(len(samples), 2000, replace=False)
        samples = samples[idx]

    dims = [
        {"label": name, "values": samples[:, i].tolist()}
        for i, name in enumerate(param_names)
    ]
    fig = go.Figure(
        go.Splom(
            dimensions=dims,
            showupperhalf=False,
            diagonal_visible=True,
            marker=dict(size=2, color="rgba(79,70,229,0.3)"),
        )
    )
    fig.update_layout(
        height=max(380, 160 * n),
        margin=dict(l=60, r=20, t=20, b=60),
        template="simple_white",
        font=dict(family="Inter, system-ui, sans-serif", size=11, color="#0F172A"),
        paper_bgcolor="#FFFFFF",
    )
    return pyo.plot(fig, output_type="div", include_plotlyjs=False)


def _spectra_plot_div(post) -> str:
    import plotly.graph_objects as go
    import plotly.offline as pyo

    post.at_par(post.par_best)

    colors = ["#4F46E5", "#06B6D4", "#10B981", "#F59E0B", "#EF4444"]
    fig = go.Figure()

    for i, pair in enumerate(post.Pair):
        color = colors[i % len(colors)]
        x_list = pair.data.rsp_re_chbin_mean
        y_data_list = pair.data.net_re_ctsspec
        y_err_list = pair.data.net_re_ctsspec_error
        y_model_list = pair.conv_re_ctsspec

        for j, (x, y_d, y_e, y_m) in enumerate(
            zip(x_list, y_data_list, y_err_list, y_model_list)
        ):
            suffix = f" [{i + 1}]" if len(post.Pair) > 1 or len(x_list) > 1 else ""
            if len(x_list) > 1:
                suffix += f".{j + 1}"
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y_d,
                    error_y=dict(
                        type="data", array=y_e, visible=True, thickness=1, width=0
                    ),
                    mode="lines",
                    name=f"data{suffix}",
                    line=dict(color=color, width=1.5),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y_m,
                    mode="lines",
                    name=f"model{suffix}",
                    line=dict(color=color, width=2, dash="dot"),
                )
            )

    fig.update_layout(
        xaxis=dict(
            title="Energy (keV)", type="log", showgrid=True, gridcolor="#F1F5F9"
        ),
        yaxis=dict(
            title="Counts s⁻¹ keV⁻¹", type="log", showgrid=True, gridcolor="#F1F5F9"
        ),
        template="simple_white",
        margin=dict(l=65, r=20, t=20, b=50),
        height=350,
        showlegend=True,
        legend=dict(x=0.02, y=0.98, xanchor="left", yanchor="top"),
        font=dict(family="Inter, system-ui, sans-serif", size=12, color="#0F172A"),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
    )
    return pyo.plot(fig, output_type="div", include_plotlyjs=False)


# ── Inference API routes ───────────────────────────────────────────────────────


@app.post("/infer/build", response_class=HTMLResponse)
async def build_infer(request: Request):
    """Auto-derive pairs from data↔model bindings and build BayesInfer."""
    _, s, _ = _session(request)
    ist = s["infer_state"]

    pairs = _derived_pairs(s)
    ist["pairs"] = pairs
    ist["links"] = {}
    ist["nlink"] = 0
    s["infer"] = None
    ist["result"] = None
    ist["posterior"] = None
    ist["error"] = None

    if not pairs:
        ist["error"] = (
            "No bidirectional data↔model bindings found. "
            "Go to the Data or Model page and bind containers to each other."
        )
        return _render_infer_panel(s, request)

    try:
        from bayspec.infer.infer import BayesInfer

        infer_pairs = []
        for p in pairs:
            dc = s["data"].get(p["data"])
            m = s["model"].get(p["model"])
            if dc is None or m is None:
                continue
            infer_pairs.append((dc, m))

        if not infer_pairs:
            ist["error"] = "No valid pairs could be constructed."
            return _render_infer_panel(s, request)

        # Check each Data container has at least one DataUnit
        for dc, _ in infer_pairs:
            if not dc.data:
                ist["error"] = (
                    "Data container has no units. "
                    "Upload spectral files on the Data page first."
                )
                return _render_infer_panel(s, request)

        s["infer"] = BayesInfer(pairs=infer_pairs)
    except Exception as exc:
        ist["error"] = str(exc)

    return _render_infer_panel(s, request)


@app.post("/infer/link", response_class=HTMLResponse)
async def link_params(request: Request, nlink: int = Form(0)):
    _, s, _ = _session(request)
    ist = s["infer_state"]
    infer = s.get("infer")
    if infer is None:
        ist["error"] = "Build inference pairs first."
        return _render_infer_panel(s, request)

    for pid in list(infer.par.keys()):
        infer.unlink(pid)

    ist["links"] = {}
    body = await request.form()
    for i in range(nlink):
        key = f"link_{i}"
        raw = body.getlist(key)
        pids = [int(r[4:]) for r in raw if r.startswith("par#")]
        if len(pids) > 1:
            infer.link(pids)
            ist["links"][i] = pids

    ist["nlink"] = nlink
    ist["error"] = None
    return _render_infer_panel(s, request)


@app.post("/infer/manual", response_class=HTMLResponse)
async def manual_fit(request: Request):
    _, s, _ = _session(request)
    ist = s["infer_state"]
    infer = s.get("infer")
    if infer is None:
        ist["error"] = "Build inference pairs first."
        return _render_infer_panel(s, request)

    form = await request.form()
    now_par = []
    for j, (_, par) in enumerate(infer.free_par.items(), start=1):
        field = f"par_val_{j}"
        raw = form.get(field)
        if raw is not None and raw != "":
            try:
                par.val = float(raw)
            except (ValueError, TypeError):
                pass
        now_par.append(par.val)
    infer.at_par(now_par)

    sd = infer.stat_info.data_dict
    tiles = "".join(
        f"<div class='stat-tile'>"
        f"<div class='stat-tile-label'>{d} → {m}</div>"
        f"<div class='stat-tile-value'>{v}</div>"
        f"<div class='stat-tile-meta'>{st} · {b} bins</div>"
        f"</div>"
        for d, m, st, v, b in zip(
            sd["Data"], sd["Model"], sd["Statistic"], sd["Value"], sd["Bins"]
        )
    )
    return HTMLResponse(f"<div class='stat-tiles'>{tiles}</div>")


@app.get("/infer/manual/plot", response_class=HTMLResponse)
async def manual_fit_plot(request: Request):
    _, s, _ = _session(request)
    infer = s.get("infer")
    if infer is None:
        return HTMLResponse("<p class='alert alert-warning'>No inference built.</p>")
    try:
        from bayspec.util.plot import Plot

        fig = Plot.infer(infer, style="CE")
        import plotly.offline as pyo

        return HTMLResponse(
            pyo.plot(fig.fig, output_type="div", include_plotlyjs=False)
        )
    except Exception as exc:
        return HTMLResponse(f"<p class='alert alert-danger'>Plot error: {exc}</p>")


@app.post("/infer/run", response_class=HTMLResponse)
async def run_infer(
    request: Request,
    sampler: str = Form("emcee"),
    nstep: int = Form(1000),
    discard: int = Form(100),
    nlive: int = Form(400),
    savepath: str = Form("./infer_out"),
    resume: str = Form("No"),
):
    _, s, _ = _session(request)
    ist = s["infer_state"]
    ist.update(
        {
            "sampler": sampler,
            "nstep": nstep,
            "discard": discard,
            "nlive": nlive,
            "savepath": savepath,
            "result": None,
            "error": None,
            "resume": resume == "Yes",
        }
    )

    do_resume = ist.get("resume", False)

    pairs = ist.get("pairs", [])
    if not pairs:
        ist["error"] = "No pairs — click Build inference first."
        return _render_infer_panel(s, request)

    # Validate all pairs and build BayesInfer before starting the thread
    try:
        from bayspec.infer.infer import BayesInfer, MaxLikeFit

        infer_pairs = []
        for p in pairs:
            dc = s["data"].get(p["data"])
            m = s["model"].get(p["model"])
            if dc is None or m is None:
                continue
            infer_pairs.append((dc, m))

        is_bayesian = sampler in ("emcee", "multinest")
        if is_bayesian:
            s["infer"] = BayesInfer(pairs=infer_pairs)
        else:
            s["infer"] = MaxLikeFit(pairs=infer_pairs)
        Path(savepath).mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        ist["error"] = str(exc)
        return _render_infer_panel(s, request)

    # Create task and launch sampler in background thread
    task_id = str(uuid.uuid4())
    _tasks[task_id] = {
        "status": "running",
        "messages": [],
        "result_html": None,
        "error": None,
    }

    def _worker():
        task = _tasks[task_id]
        try:
            n = s["infer"].free_nparams
            task["messages"].append(f"Ready — {n} free parameter(s)")
            if is_bayesian:
                bi = s["infer"]
                if sampler == "emcee":
                    task["messages"].append(
                        f"emcee: nstep={nstep}, discard={discard}, resume={do_resume}"
                    )
                    post = bi.emcee(
                        nstep=nstep,
                        discard=discard,
                        resume=do_resume,
                        savepath=savepath,
                    )
                else:
                    task["messages"].append(
                        f"multinest: nlive={nlive}, resume={do_resume}"
                    )
                    post = bi.multinest(
                        nlive=nlive, resume=do_resume, savepath=savepath
                    )
            else:
                fit = s["infer"]
                task["messages"].append(f"Optimizer: {sampler}")
                if sampler == "lmfit":
                    post = fit.lmfit(savepath=savepath)
                else:
                    post = fit.iminuit(savepath=savepath)

            result = _posterior_html(post)
            ist["posterior"] = post
            ist["result"] = result
            ist["error"] = None
            task["result_html"] = result
            task["status"] = "done"
            task["messages"].append("Complete.")
        except Exception as exc:
            ist["error"] = str(exc)
            task["error"] = str(exc)
            task["status"] = "error"
            task["messages"].append(f"Error: {exc}")

    threading.Thread(target=_worker, daemon=True).start()

    running_html = (
        f'<div id="infer-panel">'
        f'<div class="card">'
        f'<h3 style="margin-top:0;margin-bottom:.75rem">Running\u2026</h3>'
        f'<div id="run-log" class="run-log"></div>'
        f'<div class="run-status-row">'
        f'<span class="spinner"></span>'
        f'<span id="run-status-text">Connecting\u2026</span>'
        f"</div>"
        f"</div>"
        f"</div>"
        f"<script>"
        f"(function(){{"
        f'const log=document.getElementById("run-log");'
        f'const st=document.getElementById("run-status-text");'
        f'const es=new EventSource("/infer/stream/{task_id}");'
        f'es.onmessage=function(e){{log.insertAdjacentHTML("beforeend",e.data);log.scrollTop=log.scrollHeight;st.textContent="Running\u2026";}};'
        f'es.addEventListener("done",function(e){{es.close();document.getElementById("infer-panel").outerHTML=e.data;}});'
        f'es.onerror=function(){{es.close();st.textContent="Stream error \u2014 refresh to see results.";}};'
        f"}})();"
        f"</script>"
    )
    return HTMLResponse(running_html)


@app.get("/infer/stream/{task_id}")
async def infer_stream(task_id: str, request: Request):
    _, s, _ = _session(request)

    async def generate():
        sent = 0
        while True:
            if await request.is_disconnected():
                break
            task = _tasks.get(task_id)
            if task is None:
                panel = '<div id="infer-panel"><div class="alert alert-warning">Task not found.</div></div>'
                yield f"event: done\ndata: {panel}\n\n"
                break

            messages = task["messages"]
            while sent < len(messages):
                safe = (
                    messages[sent]
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                )
                sent += 1
                yield f"data: <div class='log-line'>{safe}</div>\n\n"

            if task["status"] in ("done", "error"):
                panel_html = _render_infer_panel_str(s)
                lines = panel_html.replace("\r\n", "\n").split("\n")
                data_block = "\n".join(f"data: {line}" for line in lines)
                yield f"event: done\n{data_block}\n\n"
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/infer/plots/corner", response_class=HTMLResponse)
async def infer_corner_plot(request: Request):
    _, s, _ = _session(request)
    post = s["infer_state"].get("posterior")
    if post is None:
        return HTMLResponse(
            "<p class='alert alert-warning'>No posterior available.</p>"
        )
    try:
        return HTMLResponse(_corner_plot_div(post))
    except Exception as exc:
        return HTMLResponse(
            f"<p class='alert alert-danger'>Corner plot error: {exc}</p>"
        )


@app.get("/infer/plots/spectra", response_class=HTMLResponse)
async def infer_spectra_plot(request: Request):
    _, s, _ = _session(request)
    post = s["infer_state"].get("posterior")
    if post is None:
        return HTMLResponse(
            "<p class='alert alert-warning'>No posterior available.</p>"
        )
    try:
        return HTMLResponse(_spectra_plot_div(post))
    except Exception as exc:
        return HTMLResponse(
            f"<p class='alert alert-danger'>Spectra plot error: {exc}</p>"
        )


@app.get("/infer/samples.csv")
async def infer_samples_csv(request: Request):
    """Download posterior samples (free parameters) as CSV."""
    _, s, _ = _session(request)
    post = s["infer_state"].get("posterior")
    if post is None:
        return PlainTextResponse(
            "No posterior available — run inference first.", status_code=404
        )
    try:
        import io

        import numpy as np

        samples = np.asarray(post.param_sample)
        names = post.free_par_info.data_dict["Parameter"]
        buf = io.StringIO()
        buf.write(",".join(str(n) for n in names) + "\n")
        np.savetxt(buf, samples, delimiter=",", fmt="%.8g")
        return PlainTextResponse(
            buf.getvalue(),
            media_type="text/csv",
            headers={
                "Content-Disposition": 'attachment; filename="posterior_samples.csv"'
            },
        )
    except Exception as exc:
        return PlainTextResponse(f"CSV export error: {exc}", status_code=500)


@app.get("/infer/plots/model", response_class=HTMLResponse)
async def infer_model_plot(
    request: Request,
    mkey: str = "",
    style: str = "vFv",
    comps: str = "",
    e_lo: float = 0.0,
    e_hi: float = 4.0,
    time: str = "",
):
    _, s, _ = _session(request)
    post = s["infer_state"].get("posterior")
    if post is None:
        return HTMLResponse(
            "<p class='alert alert-warning'>No posterior available.</p>"
        )

    all_comps = s["model_component"].get(mkey, {})
    selected = [c.strip() for c in comps.split(",") if c.strip()]
    plot_composed = "*" in selected
    if plot_composed:
        selected = list(all_comps.keys())
    if not selected:
        return HTMLResponse(
            "<p class='alert alert-warning'>Select at least one component.</p>"
        )

    add_styles = {"vFv", "Fv", "NE"}
    nou_styles = {"NoU"}

    def _style_ok(comp) -> bool:
        t = getattr(comp, "type", None)
        if style in add_styles:
            return t == "add"
        if style in nou_styles:
            return t in ("mul", "math")
        return True

    keep = [c for c in selected if c in all_comps and _style_ok(all_comps[c])]
    if plot_composed:
        composed_model = s["model"].get(mkey)
        if composed_model and _style_ok(composed_model):
            keep.insert(0, composed_model)
    if not keep:
        return HTMLResponse(
            f"<p class='alert alert-warning'>No components compatible with style '{style}'. "
            "vFv / Fv / NE need additive components; NoU needs mul / math.</p>"
        )

    try:
        tarr_val = _parse_optional_float(time)
        return HTMLResponse(
            _model_spectra_div(post, mkey, style, keep, s, e_lo, e_hi, tarr_val)
        )
    except Exception as exc:
        return HTMLResponse(
            f"<p class='alert alert-danger'>Model spectra error: {exc}</p>"
        )


# ── Editor helpers ─────────────────────────────────────────────────────────────


def _render_editor_panel(request: Request):
    _, s, _ = _session(request)
    return templates.TemplateResponse(
        request=request,
        name="partials/editor_panel.html",
        context={"s": s},
    )


# ── Editor API routes ──────────────────────────────────────────────────────────


@app.post("/editor/register", response_class=HTMLResponse)
async def register_model(request: Request, code: str = Form(...)):
    _, s, _ = _session(request)
    est = s["editor_state"]

    from bayspec.model.model import Model

    namespace: dict = {}
    try:
        exec(compile(code, "<user-model>", "exec"), namespace)  # noqa: S102
    except SyntaxError as exc:
        est.update({"status": f"Syntax error: {exc}", "status_type": "danger"})
        return _render_editor_panel(request)
    except Exception as exc:
        est.update({"status": f"Runtime error: {exc}", "status_type": "danger"})
        return _render_editor_panel(request)

    new_classes = {
        name: cls
        for name, cls in namespace.items()
        if isinstance(cls, type)
        and issubclass(cls, Model)
        and name not in ("Model", "Additive", "Multiplicative", "Mathematic")
    }
    if not new_classes:
        est.update(
            {"status": "No Model subclass found in the code.", "status_type": "warning"}
        )
        return _render_editor_panel(request)

    _local_models.update(new_classes)
    templates.env.globals["local_model_names"] = list(_local_models.keys())
    s["custom_models"].update(new_classes)

    names = ", ".join(new_classes.keys())
    est.update({"status": f"Registered: {names}", "status_type": "success"})
    return _render_editor_panel(request)
