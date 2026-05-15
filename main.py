import state
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="BaySpec")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

SESSION_COOKIE = "bsp_session"


def _session_id(request: Request, response: Response) -> str:
    sid = request.cookies.get(SESSION_COOKIE)
    if not sid:
        sid = state.new_id()
        response.set_cookie(SESSION_COOKIE, sid, httponly=True, samesite="lax")
    return sid


def _ctx(request: Request, response: Response, **kwargs) -> dict:
    sid = _session_id(request, response)
    return {"session_id": sid, "s": state.get(sid), **kwargs}


def _render(name: str, request: Request, response: Response, **kwargs):
    return templates.TemplateResponse(
        request=request, name=name, context=_ctx(request, response, **kwargs)
    )


@app.get("/", response_class=HTMLResponse)
async def home(request: Request, response: Response):
    return _render("home.html", request, response)


@app.get("/data", response_class=HTMLResponse)
async def data_page(request: Request, response: Response):
    return _render("data.html", request, response)


@app.get("/model", response_class=HTMLResponse)
async def model_page(request: Request, response: Response):
    return _render("model.html", request, response)


@app.get("/infer", response_class=HTMLResponse)
async def infer_page(request: Request, response: Response):
    return _render("infer.html", request, response)
