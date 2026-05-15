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
