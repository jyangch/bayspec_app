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

st.markdown('#### Quickstart')
st.markdown(
    '<div class="bsp-quickstart">'
    '  <div class="bsp-qs-card">'
    '    <div class="bsp-qs-num">1</div>'
    '    <div class="bsp-qs-emoji">🔭</div>'
    '    <div class="bsp-qs-title">Bring your data</div>'
    '    <div class="bsp-qs-body">Upload OGIP-format <code>src</code> / '
    '<code>bkg</code> / <code>rsp</code> (or <code>rmf</code> + <code>arf</code>); '
    'pick statistic, noticing, grouping per unit.</div>'
    '    <a href="/data" target="_self" class="bsp-qs-link">Go to Data →</a>'
    '  </div>'
    '  <div class="bsp-qs-arrow">›</div>'
    '  <div class="bsp-qs-card">'
    '    <div class="bsp-qs-num">2</div>'
    '    <div class="bsp-qs-emoji">🌈</div>'
    '    <div class="bsp-qs-title">Compose your model</div>'
    '    <div class="bsp-qs-body">Pick components from local / Astromodels / '
    'XSPEC, or paste your own Python class. Combine with '
    '<code>+ − * /</code>.</div>'
    '    <a href="/model" target="_self" class="bsp-qs-link">Go to Model →</a>'
    '  </div>'
    '  <div class="bsp-qs-arrow">›</div>'
    '  <div class="bsp-qs-card">'
    '    <div class="bsp-qs-num">3</div>'
    '    <div class="bsp-qs-emoji">📝</div>'
    '    <div class="bsp-qs-title">Run inference</div>'
    '    <div class="bsp-qs-body">Pair Data ↔ Model, manually fit, then run '
    'a Bayesian sampler or max-likelihood optimizer with one click.</div>'
    '    <a href="/infer" target="_self" class="bsp-qs-link">Go to Inference →</a>'
    '  </div>'
    '</div>',
    unsafe_allow_html=True,
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
