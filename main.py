import re
from pathlib import Path
from typing import Optional

import state
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="BaySpec")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
templates.env.globals["zip"] = zip

from bayspec.model.local import local_models as _local_models
templates.env.globals["local_model_names"] = list(_local_models.keys())

SESSION_COOKIE = "bsp_session"
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


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
    s = notc_str.strip()
    if not s:
        return None
    windows = [w.strip() for w in s.split("|") if w.strip()]
    parsed = []
    for w in windows:
        vals = [float(x) for x in w.split(",")]
        if len(vals) != 2:
            raise ValueError(f"Each window must be 'lo, hi', got: {w!r}")
        parsed.append(vals)
    return parsed[0] if len(parsed) == 1 else parsed


def _counts_plot_div(unit) -> str:
    import plotly.graph_objects as go
    import plotly.offline as pyo

    x = unit.rsp_re_chbin_mean
    y = unit.src_re_ctsspec
    err = unit.src_re_ctsspec_error

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y,
        error_y=dict(type="data", array=err, visible=True, thickness=1, width=0),
        mode="lines",
        name="Source",
        line=dict(color="#4F46E5", width=1.5),
    ))

    try:
        yb = unit.bkg_re_ctsspec
        errb = unit.bkg_re_ctsspec_error
        fig.add_trace(go.Scatter(
            x=x, y=yb,
            error_y=dict(type="data", array=errb, visible=True, thickness=1, width=0),
            mode="lines",
            name="Background",
            line=dict(color="#94A3B8", width=1, dash="dot"),
        ))
        fig.update_layout(
            showlegend=True,
            legend=dict(x=0.02, y=0.98, xanchor="left", yanchor="top"),
        )
    except Exception:
        pass

    fig.update_layout(
        xaxis=dict(title="Energy (keV)", type="log", showgrid=True, gridcolor="#F1F5F9"),
        yaxis=dict(title="Counts s⁻¹ keV⁻¹", type="log", showgrid=True, gridcolor="#F1F5F9"),
        template="simple_white",
        margin=dict(l=60, r=20, t=20, b=50),
        height=300,
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


# ── Data API routes ────────────────────────────────────────────────────────────

@app.post("/data/units", response_class=HTMLResponse)
async def add_unit(
    request: Request,
    unit_key: str = Form(...),
    src: UploadFile = File(...),
    bkg: Optional[UploadFile] = File(None),
    rsp: Optional[UploadFile] = File(None),
    rmf: Optional[UploadFile] = File(None),
    arf: Optional[UploadFile] = File(None),
    stat: str = Form("pgstat"),
    notc_str: str = Form(""),
    min_sigma: Optional[float] = Form(None),
    max_bin: Optional[int] = Form(None),
):
    sid, s, is_new = _session(request)
    key = _safe_key(unit_key) or "unit"

    unit_dir = UPLOAD_DIR / sid / key
    unit_dir.mkdir(parents=True, exist_ok=True)

    async def _save(upload: Optional[UploadFile]) -> Optional[str]:
        if upload is None or not upload.filename:
            return None
        path = unit_dir / upload.filename
        path.write_bytes(await upload.read())
        return str(path)

    paths = {
        "src": await _save(src),
        "bkg": await _save(bkg),
        "rsp": await _save(rsp),
        "rmf": await _save(rmf),
        "arf": await _save(arf),
    }

    error = None
    if paths["src"]:
        try:
            notc = _parse_notc(notc_str)
            rebn = {}
            if min_sigma is not None:
                rebn["min_sigma"] = min_sigma
            if max_bin is not None:
                rebn["max_bin"] = int(max_bin)

            from bayspec.data.data import DataUnit
            du = DataUnit(
                src=paths["src"],
                bkg=paths["bkg"],
                rsp=paths["rsp"],
                rmf=paths["rmf"],
                arf=paths["arf"],
                stat=stat,
                notc=notc,
                rebn=rebn or None,
            )
            s["data"][key] = du
        except Exception as exc:
            error = str(exc)
    else:
        error = "Source file is required."

    s["data_state"][key] = {
        "stat": stat,
        "notc_str": notc_str,
        "min_sigma": min_sigma,
        "max_bin": max_bin,
        "error": error,
    }

    resp = templates.TemplateResponse(
        request=request, name="partials/unit_list.html", context={"s": s}
    )
    if is_new:
        resp.set_cookie(SESSION_COOKIE, sid, httponly=True, samesite="lax")
    return resp


@app.delete("/data/units/{key}", response_class=HTMLResponse)
async def delete_unit(key: str, request: Request):
    sid, s, is_new = _session(request)
    s["data"].pop(key, None)
    s["data_state"].pop(key, None)

    import shutil
    unit_dir = UPLOAD_DIR / sid / key
    if unit_dir.exists():
        shutil.rmtree(unit_dir)

    resp = templates.TemplateResponse(
        request=request, name="partials/unit_list.html", context={"s": s}
    )
    if is_new:
        resp.set_cookie(SESSION_COOKIE, sid, httponly=True, samesite="lax")
    return resp


@app.get("/data/units/{key}/plot", response_class=HTMLResponse)
async def unit_plot(key: str, request: Request):
    sid, s, _ = _session(request)
    du = s["data"].get(key)
    if du is None:
        return HTMLResponse("<p class='alert alert-warning'>Unit not found.</p>")
    try:
        div = _counts_plot_div(du)
        return HTMLResponse(div)
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


def _model_plot_div(model) -> str:
    import numpy as np
    import plotly.graph_objects as go
    import plotly.offline as pyo

    E = np.logspace(1, 3, 150)
    try:
        NE = model.func(E)
        vFv = E ** 2 * NE
    except Exception as exc:
        return f"<p class='alert alert-danger'>Evaluation error: {exc}</p>"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=E, y=vFv, mode="lines",
        line=dict(color="#4F46E5", width=2),
        name=str(getattr(model, "expr", "model")),
    ))
    fig.update_layout(
        xaxis=dict(title="Energy (keV)", type="log", showgrid=True, gridcolor="#F1F5F9"),
        yaxis=dict(title="E² N(E)  (keV photons s⁻¹ cm⁻²)", type="log",
                   showgrid=True, gridcolor="#F1F5F9"),
        template="simple_white",
        margin=dict(l=65, r=20, t=20, b=50),
        height=280,
        font=dict(family="Inter, system-ui, sans-serif", size=12, color="#0F172A"),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
    )
    return pyo.plot(fig, output_type="div", include_plotlyjs=False)


def _render_model_card(mkey: str, request: Request):
    _, s, _ = _session(request)
    return templates.TemplateResponse(
        request=request,
        name="partials/model_card.html",
        context={"s": s, "mkey": mkey},
    )


def _render_model_list(request: Request):
    _, s, _ = _session(request)
    return templates.TemplateResponse(
        request=request,
        name="partials/model_list.html",
        context={"s": s},
    )


# ── Model API routes ───────────────────────────────────────────────────────────

@app.post("/model/models", response_class=HTMLResponse)
async def create_model(request: Request, model_key: str = Form(...)):
    sid, s, is_new = _session(request)
    key = _safe_key(model_key) or "model"
    if key not in s["model_component"]:
        s["model_component"][key] = {}
        s["model_state"][key] = {"expression": "", "error": None}
    resp = _render_model_list(request)
    if is_new:
        resp.set_cookie(SESSION_COOKIE, sid, httponly=True, samesite="lax")
    return resp


@app.delete("/model/models/{mkey}", response_class=HTMLResponse)
async def delete_model(mkey: str, request: Request):
    sid, s, is_new = _session(request)
    s["model_component"].pop(mkey, None)
    s["model_state"].pop(mkey, None)
    s["model"].pop(mkey, None)
    resp = _render_model_list(request)
    if is_new:
        resp.set_cookie(SESSION_COOKIE, sid, httponly=True, samesite="lax")
    return resp


@app.post("/model/models/{mkey}/components", response_class=HTMLResponse)
async def add_component(
    mkey: str,
    request: Request,
    comp_type: str = Form(...),
    comp_key: str = Form(""),
):
    _, s, _ = _session(request)
    ckey = _safe_key(comp_key.strip()) if comp_key.strip() else comp_type
    from bayspec.model.local import local_models as lm
    if comp_type not in lm:
        s["model_state"].setdefault(mkey, {})["error"] = f"Unknown model: {comp_type!r}"
        return _render_model_card(mkey, request)
    comp = lm[comp_type]()
    s["model_component"].setdefault(mkey, {})[ckey] = comp
    s["model_state"].setdefault(mkey, {})["error"] = None
    return _render_model_card(mkey, request)


@app.delete("/model/models/{mkey}/components/{ckey}", response_class=HTMLResponse)
async def delete_component(mkey: str, ckey: str, request: Request):
    _, s, _ = _session(request)
    s["model_component"].get(mkey, {}).pop(ckey, None)
    s["model"].pop(mkey, None)  # invalidate composed model
    return _render_model_card(mkey, request)


@app.post("/model/models/{mkey}/components/{ckey}/update", response_class=HTMLResponse)
async def update_component(mkey: str, ckey: str, request: Request):
    form = await request.form()
    _, s, _ = _session(request)
    comp = s["model_component"].get(mkey, {}).get(ckey)
    if comp is None:
        return _render_model_card(mkey, request)

    cfg_dict = comp.cfg_info.data_dict
    for idx, param, orig_val in zip(cfg_dict["cfg#"], cfg_dict["Parameter"], cfg_dict["Value"]):
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
    for idx, param, orig_prior in zip(par_dict["par#"], par_dict["Parameter"], par_dict["Prior"]):
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

    return _render_model_card(mkey, request)


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
    return _render_model_card(mkey, request)


@app.get("/model/models/{mkey}/plot", response_class=HTMLResponse)
async def model_plot(mkey: str, request: Request):
    _, s, _ = _session(request)
    model = s["model"].get(mkey)
    if model is None:
        return HTMLResponse("<p class='alert alert-warning'>Compose the model first.</p>")
    return HTMLResponse(_model_plot_div(model))
