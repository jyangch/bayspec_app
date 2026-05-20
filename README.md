# BaySpec App

A Bayesian spectral fitting workbench for high-energy astrophysics — a web
front-end for the [`bayspec`](https://github.com/jyangch/bayspec) library.
Load OGIP spectra, assemble spectral models from a few component libraries,
bind them to data, and run a Bayesian sampler or a maximum-likelihood
optimizer to obtain posterior estimates — all in a single browser session,
no notebook glue required.

## Features

- **Data containers** — group source / background / response triplets into
  `Data` objects, one per unit. FITS uploads are parsed in place; metadata,
  counts, exposure, and effective area are all surfaced.
- **Models from four libraries** — local (`bayspec`), `astromodels`, XSPEC
  (where available), and user-defined Python sources registered through the
  Model page. Compose components with an algebraic expression like
  `tbabs * cpl`.
- **Auto-paired inference** — every `Data` ↔ `Model` binding becomes a
  fitting pair. Tweak parameters, freeze them, or link them across pairs.
- **Bayesian samplers** — `emcee` (default) and ultranest, with a
  maximum-likelihood optimizer for quick checks. Progress streams over SSE.
- **Run history (last 3)** — switch between recent runs from a chip strip in
  the Inference panel, with a stale-binding warning if the displayed run no
  longer matches the current setup.
- **Compare runs** — side-by-side best-fit ± 1σ table across history
  entries, CSV-exportable.
- **Save / Load configuration** — bundle every UI choice and parameter into
  a JSON file. FITS uploads aren't included (not portable); re-attach them
  after import.
- **Reset session** — wipe Data, Models, Pairs, and Inference back to an
  initial state in one click. Custom-model registrations are preserved.
- **Live workflow indicator** — five-stage Data → Model → Pairs →
  Inference → Analyzer progress bar in the sidebar, updated from session
  state.

## Quickstart

Requires Python 3.12+.

```sh
git clone https://github.com/jyangch/bayspec_app.git
cd bayspec_app

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

uvicorn main:app --reload
```

Open <http://127.0.0.1:8000>. Follow the workflow: **Data** (upload
spectra) → **Model** (pick or compose components, bind to a Data
container) → **Inference** (configure sampler, run, inspect posterior).

### XSPEC components (optional)

The XSPEC library tab is populated only if `xspec` is importable in your
Python environment. The rest of the app works without it.

## Run on Hugging Face Spaces

A live, public deployment is available at
<https://huggingface.co/spaces/jyangch/bayspec>. The repo doubles as a
Docker-SDK Space — the YAML header at the top of this README and the
`Dockerfile` are everything HF needs to build and serve the app.

The HF free tier resets the filesystem on container restart, so uploaded
FITS files and exported fits are lost between sleeps. Enable Persistent
Storage in the Space settings if that matters.

## Project layout

```
main.py              FastAPI app — every route, helper, and SSE worker
state.py             Per-session in-memory state store (cookie-keyed)
templates/           Jinja2 templates (base.html + one per page + partials)
static/style.css     UI styles (design tokens, components, layout)
static/app.js        Tiny client-side glue (HTMX is the main driver)
docs/                Internal conventions (docstring standard, etc.)
```

The app is a single FastAPI process with no JavaScript framework — HTMX
swaps server-rendered fragments into the page. All session data lives in
a per-process dict keyed by a cookie; restarting the server clears every
session.

## Citation and upstream

This app is a front-end. The fitting machinery lives in the
[`bayspec`](https://pypi.org/project/bayspec/) library; please cite that
project when publishing results.

## License

GPL-3.0 — see [`LICENSE`](LICENSE).
