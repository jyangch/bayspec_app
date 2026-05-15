import streamlit as st

st.markdown(
    """
    <div class="bsp-hero">
        <span class="bsp-eyebrow">v0.3 · Bayesian spectral fitting</span>
        <h1>Welcome to <span class="bsp-gradient">BaySpec</span></h1>
        <p class="lead">
            A modern, browser-based fitting workbench for high-energy astrophysical spectra.
            Combine local, Astromodels, XSPEC and user-defined components, then sample with
            <b>emcee</b>, <b>multinest</b>, <b>lmfit</b> or <b>iminuit</b> — all from a single,
            reproducible interface.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '[![PyPI](https://img.shields.io/pypi/v/bayspec?color=4F46E5&logo=PyPI&logoColor=white&style=for-the-badge)]'
    '(https://pypi.org/project/bayspec/) '
    '[![Docs](https://img.shields.io/badge/docs-readthedocs-06B6D4?style=for-the-badge&logo=readthedocs&logoColor=white)]'
    '(https://bayspec.readthedocs.io) '
    '[![License: GPL v3](https://img.shields.io/badge/license-GPL--3.0-0F172A?style=for-the-badge)]'
    '(https://www.gnu.org/licenses/gpl-3.0)'
)

st.write('')
a, b, c, d = st.columns(4)
a.metric('Inference', '4 backends', help='emcee · multinest · lmfit · iminuit')
b.metric('Model libraries', '4', help='local · Astromodels · XSPEC · user-defined')
c.metric('Spectra', 'Multi-wavelength', help='joint multi-instrument fits')
d.metric('Stack', 'Streamlit', help='browser-based, no install on user side')

st.write('')

st.markdown('#### What you can do')
f1, f2, f3 = st.columns(3)
with f1.container(border=True):
    st.markdown('##### 🔭 Bring your data')
    st.write(
        'Upload OGIP-format `src` / `bkg` / `rsp` (or `rmf` + `arf`); '
        'pick statistic, noticing, grouping, rebinning per unit.'
    )
with f2.container(border=True):
    st.markdown('##### 🌈 Compose your model')
    st.write(
        'Select components from local / Astromodels / XSPEC, '
        'or paste your own Python class. Combine with `+ - * /`.'
    )
with f3.container(border=True):
    st.markdown('##### 📝 Run inference')
    st.write(
        'Pair Data ↔ Model, manually fit, then run a Bayesian sampler '
        'or maximum-likelihood optimizer with one click.'
    )

st.write('')

with st.expander('Installation', expanded=False):
    st.markdown(
        """
**BaySpec** is on PyPI:
```bash
pip install bayspec
```

#### Optional sampler / model backends

| Backend | Install | Use |
|--|--|--|
| `emcee` | `pip install emcee` | MCMC sampler |
| `multinest` | see [PyMultiNest docs](https://johannesbuchner.github.io/PyMultiNest/) | Nested sampling |
| `lmfit` | `pip install lmfit` | Maximum-likelihood fit |
| `iminuit` | `pip install iminuit` | Maximum-likelihood fit |
| `astromodels` | `pip install astromodels` | Bridge to Astromodels |
| `xspec_models_cxc` | requires HEASoft + Xspec ≥ 12.12.1, see [`xspec-models-cxc`](https://github.com/cxcsds/xspec-models-cxc) | Bridge to XSPEC |
        """
    )

with st.expander('Documentation & resources', expanded=False):
    st.markdown(
        """
- 📘 [Documentation](https://bayspec.readthedocs.io)
- 🧪 [Examples](https://github.com/jyangch/bayspec/tree/main/examples)
- 💻 [Source code](https://github.com/jyangch/bayspec)
- 🌐 [App on Streamlit Cloud](https://bayspec.streamlit.app)
        """
    )

st.divider()
st.caption(
    '_BaySpec_ is distributed under the terms of the '
    '[GPL-3.0](https://www.gnu.org/licenses/gpl-3.0-standalone.html) license.'
)
