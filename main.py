import asyncio
import re
import threading
import uuid
from pathlib import Path
from typing import Optional

import state
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
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


@app.get("/editor", response_class=HTMLResponse)
async def editor_page(request: Request):
    return _render("editor.html", request)


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


# ── Inference helpers ──────────────────────────────────────────────────────────

def _posterior_html(post) -> str:
    """Return HTML fragment with parameter CI table + stat table."""
    fp = post.free_par_info.data_dict
    par_rows = "".join(
        f"<tr><td class='param-name'>{par}</td>"
        f"<td>{best:.4g}</td>"
        f"<td><code>{ci}</code></td>"
        f"<td>{mean:.4g}</td>"
        f"<td>{med:.4g}</td></tr>"
        for par, best, ci, mean, med in zip(
            fp["Parameter"], fp["Best"], fp["1sigma CI"], fp["Mean"], fp["Median"]
        )
    )
    par_html = (
        "<div class='param-section-label' style='margin-bottom:.4rem'>Parameters</div>"
        "<table class='param-table'>"
        "<thead><tr><th>Parameter</th><th>Best</th><th>1σ CI</th>"
        "<th>Mean</th><th>Median</th></tr></thead>"
        f"<tbody>{par_rows}</tbody></table>"
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
        "<div class='param-section-label' style='margin:.75rem 0 .4rem'>Statistics</div>"
        "<table class='param-table'>"
        "<thead><tr><th>Data</th><th>Model</th><th>Statistic</th>"
        "<th>Value</th><th>Bins</th></tr></thead>"
        f"<tbody>{stat_rows}</tbody></table>"
    )

    lnz_html = ""
    try:
        lnz = post.lnZ
        if lnz is not None:
            lnz_html = f"<p class='caption' style='margin-top:.6rem'>ln Z = {lnz:.3f}</p>"
    except Exception:
        pass

    return par_html + stat_html + lnz_html


def _render_infer_panel(request: Request):
    _, s, _ = _session(request)
    return templates.TemplateResponse(
        request=request,
        name="partials/infer_panel.html",
        context={"s": s},
    )


def _render_infer_panel_str(s: dict) -> str:
    return templates.env.get_template("partials/infer_panel.html").render(s=s)


# ── Inference API routes ───────────────────────────────────────────────────────

@app.post("/infer/pairs", response_class=HTMLResponse)
async def add_infer_pair(
    request: Request,
    data_key: str = Form(...),
    model_key: str = Form(...),
):
    _, s, _ = _session(request)
    ist = s["infer_state"]
    ist.setdefault("pairs", [])
    ist["pairs"].append({"data": data_key, "model": model_key})
    return _render_infer_panel(request)


@app.delete("/infer/pairs/{idx}", response_class=HTMLResponse)
async def delete_infer_pair(idx: int, request: Request):
    _, s, _ = _session(request)
    pairs = s["infer_state"].get("pairs", [])
    if 0 <= idx < len(pairs):
        pairs.pop(idx)
    return _render_infer_panel(request)


@app.post("/infer/run", response_class=HTMLResponse)
async def run_infer(
    request: Request,
    sampler: str = Form("emcee"),
    nstep: int = Form(1000),
    discard: int = Form(100),
    nlive: int = Form(400),
    savepath: str = Form("./infer_out"),
):
    _, s, _ = _session(request)
    ist = s["infer_state"]
    ist.update({
        "sampler": sampler, "nstep": nstep, "discard": discard,
        "nlive": nlive, "savepath": savepath, "result": None, "error": None,
    })

    pairs = ist.get("pairs", [])
    if not pairs:
        ist["error"] = "Add at least one (data, model) pair before running."
        return _render_infer_panel(request)

    # Validate all pairs and build BayesInfer before starting the thread
    try:
        from bayspec.data.data import Data
        from bayspec.infer.infer import BayesInfer

        infer_pairs = []
        for p in pairs:
            du = s["data"].get(p["data"])
            m = s["model"].get(p["model"])
            if du is None:
                raise ValueError(f"Data unit '{p['data']}' not found.")
            if m is None:
                raise ValueError(f"Model '{p['model']}' not yet composed.")
            infer_pairs.append((Data(data=[(p["data"], du)]), m))

        bi = BayesInfer(pairs=infer_pairs)
        s["infer"] = bi
        Path(savepath).mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        ist["error"] = str(exc)
        return _render_infer_panel(request)

    # Create task and launch sampler in background thread
    task_id = str(uuid.uuid4())
    _tasks[task_id] = {"status": "running", "messages": [], "result_html": None, "error": None}

    def _worker():
        task = _tasks[task_id]
        try:
            task["messages"].append(
                f"BayesInfer ready — {len(infer_pairs)} pair(s), {bi.free_nparams} free parameter(s)"
            )
            if sampler == "emcee":
                task["messages"].append(f"emcee: nstep={nstep}, discard={discard}")
                post = bi.emcee(nstep=nstep, discard=discard, savepath=savepath)
            else:
                task["messages"].append(f"multinest: nlive={nlive}")
                post = bi.multinest(nlive=nlive, savepath=savepath)
            result = _posterior_html(post)
            ist["result"] = result
            ist["error"] = None
            task["result_html"] = result
            task["status"] = "done"
            task["messages"].append("Sampling complete.")
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
        f'</div>'
        f'</div>'
        f'</div>'
        f'<script>'
        f'(function(){{'
        f'const log=document.getElementById("run-log");'
        f'const st=document.getElementById("run-status-text");'
        f'const es=new EventSource("/infer/stream/{task_id}");'
        f'es.onmessage=function(e){{log.insertAdjacentHTML("beforeend",e.data);log.scrollTop=log.scrollHeight;st.textContent="Running\u2026";}};'
        f'es.addEventListener("done",function(e){{es.close();document.getElementById("infer-panel").outerHTML=e.data;}});'
        f'es.onerror=function(){{es.close();st.textContent="Stream error \u2014 refresh to see results.";}};'
        f'}})();'
        f'</script>'
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
                safe = (messages[sent]
                        .replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;"))
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
        est.update({"status": "No Model subclass found in the code.", "status_type": "warning"})
        return _render_editor_panel(request)

    _local_models.update(new_classes)
    templates.env.globals["local_model_names"] = list(_local_models.keys())
    s["custom_models"].update(new_classes)

    names = ", ".join(new_classes.keys())
    est.update({"status": f"Registered: {names}", "status_type": "success"})
    return _render_editor_panel(request)
