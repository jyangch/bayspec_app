import os
from pathlib import Path
import time

from bayspec.infer.infer import BayesInfer, MaxLikeFit
from bayspec.util.plot import Plot
import numpy as np
import pandas as pd
import streamlit as st


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


init_session_state()

st.markdown(
    '<p class="bsp-subtitle">Pair Data ↔ Model, do a manual fit, then run a Bayesian '
    'sampler or maximum-likelihood optimizer and inspect the posterior.</p>',
    unsafe_allow_html=True,
)


def empty_card(icon, title, body):
    st.markdown(
        f'<div class="bsp-empty">'
        f'  <div class="bsp-empty-icon">{icon}</div>'
        f'  <div class="bsp-empty-title">{title}</div>'
        f'  <div>{body}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def set_ini(key, ini=None):
    if key not in st.session_state.infer_state:
        st.session_state.infer_state[key] = ini


def get_val(key):
    if key in st.session_state:
        st.session_state.infer_state[key] = st.session_state[key]
    return st.session_state.infer_state[key]


def get_data(key):
    if key in st.session_state:
        for row, edited in st.session_state[key]['edited_rows'].items():
            for col, value in edited.items():
                st.session_state.infer_state[key].loc[int(row), col] = value
    return st.session_state.infer_state[key]


def get_idx(key, options):
    if key in st.session_state:
        st.session_state.infer_state[key] = st.session_state[key]
    value = st.session_state.infer_state[key]
    if (value is None) or (value not in options):
        return None
    else:
        return options.index(value)


def get_download_folder():
    if os.name == 'nt':
        return Path(os.getenv('USERPROFILE')) / 'Downloads'
    else:
        return Path.home() / 'Downloads'


def render_stepper(step1_done: bool, step2_done: bool, step3_done: bool) -> None:
    """4-step stepper at the top of the Infer page.

    Step 4 (Analyzer) is "active" iff step 3 is done.
    """
    def cls(done: bool, active: bool) -> str:
        if done:
            return 'bsp-step done'
        if active:
            return 'bsp-step active'
        return 'bsp-step'

    steps = [
        (cls(step1_done, len(all_pairs) > 0 and not step1_done), '1', 'Pair', 'Data ↔ Model'),
        (cls(step2_done, step1_done and not step2_done), '2', 'Check', 'Configs & Params'),
        (cls(step3_done, step2_done and not step3_done), '3', 'Fit', 'Manual + Inference'),
        (cls(False, step3_done), '4', 'Analyze', 'Posterior summary'),
    ]
    html = '<div class="bsp-stepper">'
    for state_cls, num, tag, title in steps:
        html += (
            f'<div class="{state_cls}">'
            f'  <span class="bsp-step-num">{num}</span>'
            f'  <span class="bsp-step-body">'
            f'    <span class="bsp-step-tag">Step {num}</span>'
            f'    <span class="bsp-step-title">{tag} — {title}</span>'
            f'  </span>'
            f'</div>'
        )
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


with st.sidebar:
    st.markdown('##### 📝 Fitting workflow')
    st.caption(
        '1. Pair Data ↔ Model · 2. Check configs · 3. Fit & inference · 4. Analyze posterior'
    )


# ---- Auto-derive pairs from Data ↔ Model bindings -----------------------
all_pairs: dict[str, list[str]] = {}
for data_key in st.session_state.data:
    model_key = st.session_state.data_state.get(f'{data_key}_model')
    if (
        model_key is not None
        and st.session_state.model_state.get(f'{model_key}_data') == data_key
    ):
        all_pairs[f'{data_key} 🔗 {model_key}'] = [data_key, model_key]

pair_hash = tuple(sorted(all_pairs.keys()))
ist = st.session_state.infer_state
built = ist.get('built_hash') == pair_hash and len(all_pairs) > 0
confirmed = built and ist.get('confirmed_hash') == pair_hash
has_post = 'post' in ist

render_stepper(step1_done=built, step2_done=confirmed, step3_done=has_post)

# Reset the live infer object every rerun; we rebuild it when the user
# is past the Build gate so that edits on Data/Model pages flow through.
st.session_state.infer = None
pair_list: list[list] = []
infer = None
if built:
    for pkey in all_pairs:
        data_key, model_key = all_pairs[pkey]
        pair_list.append(
            [
                st.session_state.data[data_key],
                st.session_state.model[model_key],
            ]
        )
    try:
        infer = BayesInfer(pair_list)
        st.session_state.infer = infer
    except Exception as exc:
        st.error(f'Could not build inference object: {exc}', icon='🚨')
        built = confirmed = False
        ist.pop('built_hash', None)
        ist.pop('confirmed_hash', None)


# ---- Step 1: Build inference -------------------------------------------
if not all_pairs:
    empty_card(
        '🔗',
        'No fitting pairs yet',
        'Pairs are auto-derived from Data ↔ Model bindings. '
        'Go to the <a href="/data">Data</a> or <a href="/model">Model</a> page '
        'to bind a Data container to a Model, then return here.',
    )
    st.stop()

if not built:
    with st.container(border=True):
        head_col, count_col = st.columns([6, 1])
        with head_col:
            st.markdown('##### 🔗 Pairs detected')
        with count_col:
            st.markdown(
                f'<div class="bsp-count-pill">{len(all_pairs)}</div>',
                unsafe_allow_html=True,
            )

        for data_key, model_key in all_pairs.values():
            st.markdown(
                f'<div class="bsp-pair-row">'
                f'  <span class="bsp-data-badge">{data_key}</span>'
                f'  <span class="bsp-pair-arrow">↔</span>'
                f'  <span class="bsp-model-badge">{model_key}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        if st.button(
            '🛠️  Build inference',
            type='primary',
            use_container_width=True,
            help='Initialise the BayesInfer object from the detected pairs.',
        ):
            ist['built_hash'] = pair_hash
            ist.pop('confirmed_hash', None)
            st.rerun()
    st.stop()


# ---- Step 2: Check configs & params ------------------------------------
if not confirmed:
    assert infer is not None  # past the Build gate
    with st.container(border=True):
        st.markdown('##### 🔎 Review configs & parameters')
        st.caption(
            'Inspect the configuration values and free parameters that the '
            'sampler will see. Confirm to proceed, or Recheck to drop back '
            'after editing your Data or Model.'
        )

        cfg_col, _, par_col = st.columns([4.9, 0.2, 4.9])
        with cfg_col:
            st.markdown('**Configurations**')
            cfg_df = pd.DataFrame(infer.cfg_info.data_dict)
            st.dataframe(cfg_df, use_container_width=True, hide_index=True)
        with par_col:
            st.markdown('**Parameters**')
            par_df = pd.DataFrame(infer.notable_par_info.data_dict)
            st.dataframe(par_df, use_container_width=True, hide_index=True)

        confirm_col, recheck_col = st.columns(2)
        with confirm_col:
            if st.button(
                '✅  Confirm — proceed to fitting',
                type='primary',
                use_container_width=True,
            ):
                ist['confirmed_hash'] = pair_hash
                st.rerun()
        with recheck_col:
            if st.button('↩️  Recheck — drop back', use_container_width=True):
                ist.pop('built_hash', None)
                ist.pop('confirmed_hash', None)
                st.rerun()
    st.stop()


# ---- Step 3+4: parameters / manual fit / inference / posterior ---------
assert infer is not None  # past the build/confirm gates above

with st.expander('***Parameter links***', expanded=False):
    key = 'infer_nlink'
    set_ini(key, 0)
    nlink = st.number_input(
        'Number of linking groups',
        min_value=0,
        value=get_val(key),
        key=key,
        help='Tie multiple free parameters together so they share a value during fitting.',
    )

    for idx in range(nlink):
        key = f'infer_link_{idx}'
        set_ini(key, [])
        options = [f'par#{p}' for p in list(infer.par.keys())]
        default = [p for p in get_val(key) if p in options]
        pids = st.multiselect(
            f'Link group {idx + 1}',
            options=options,
            default=default,
            key=key,
        )
        if len(pids) > 1:
            infer.link([int(pi[4:]) for pi in pids])

    par_df = pd.DataFrame(infer.notable_par_info.data_dict)
    st.dataframe(par_df, use_container_width=True, hide_index=True)

with st.expander('***Manual fitting***', expanded=True):
    free_par_df = pd.DataFrame(infer.free_par_info.data_dict)
    key = 'manual_free_par'
    set_ini(key, free_par_df)
    # If the parameter set has changed (e.g. via a link group edit), reset.
    cached = st.session_state.infer_state[key]
    if (
        not isinstance(cached, pd.DataFrame)
        or list(cached.columns) != list(free_par_df.columns)
        or len(cached) != len(free_par_df)
    ):
        st.session_state.infer_state[key] = free_par_df
    free_par_df = st.data_editor(
        get_data(key),
        use_container_width=True,
        num_rows='fixed',
        disabled=['par#', 'Class', 'Expression', 'Component', 'Parameter', 'Prior'],
        hide_index=True,
        key=key,
    )
    now_par = [row['Value'] for _, row in free_par_df.to_dict('index').items()]
    infer.at_par(now_par)

    stat_col, _, plot_col = st.columns([4.9, 0.2, 4.9])

    with stat_col:
        stat_df = pd.DataFrame(infer.stat_info.data_dict)
        st.dataframe(stat_df, use_container_width=True, hide_index=True)

    with plot_col:
        fig = Plot.infer(infer, style='CE')
        st.plotly_chart(
            fig.fig,
            theme='streamlit',
            use_container_width=True,
            key='manual_ctsspec_fig',
        )

with st.expander('***Inference***', expanded=True):
    run_col, _, post_col = st.columns([4.9, 0.2, 4.9])

    with run_col, st.popover('🛠️  Method settings', use_container_width=True):
        key = 'infer_method'
        ini = 'emcee'
        set_ini(key, ini)
        sampler_options = ['emcee', 'multinest']
        optimizer_options = ['lmfit', 'iminuit']
        options = sampler_options + optimizer_options
        sampler = st.selectbox(
            'Choose Bayesian sampler or maximum-likelihood optimizer',
            options,
            index=get_idx(key, options),
            key=key,
            format_func=lambda m: (
                f'{m} (Bayesian sampler)'
                if m in sampler_options
                else f'{m} (max-likelihood optimizer)'
            ),
        )

        sampler_exist = True
        multinest_nlive = 300
        emcee_nstep = 2000
        emcee_discard = 100

        if sampler == 'multinest':
            try:
                import pymultinest  # noqa: F401
            except ImportError:
                sampler_exist = False
                st.warning(
                    'To utilize Multinest for Bayesian inference, ensure Multinest is installed!',
                    icon='⚠️',
                )

            key = 'infer_multinest_nlive'
            set_ini(key, 300)
            multinest_nlive = st.slider(
                'Select the number of live points',
                50,
                1000,
                value=get_val(key),
                step=50,
                key=key,
            )

        if sampler == 'emcee':
            try:
                import emcee  # noqa: F401
            except ImportError:
                sampler_exist = False
                st.warning(
                    'To utilize Emcee for Bayesian inference, ensure Emcee is installed!',
                    icon='⚠️',
                )

            key = 'infer_emcee_nstep'
            set_ini(key, 2000)
            emcee_nstep = st.slider(
                'Select the number of steps',
                0,
                10000,
                value=get_val(key),
                step=1000,
                key=key,
            )

            key = 'infer_emcee_discard'
            set_ini(key, 100)
            emcee_discard = st.slider(
                'Select the discard steps',
                0,
                2000,
                value=get_val(key),
                step=100,
                key=key,
            )

        if sampler == 'lmfit':
            try:
                import lmfit  # noqa: F401
            except ImportError:
                sampler_exist = False
                st.warning(
                    'To utilize lmfit for Maximum Likelihood Estimation, ensure lmfit is installed!',
                    icon='⚠️',
                )

        if sampler == 'iminuit':
            try:
                import iminuit  # noqa: F401
            except ImportError:
                sampler_exist = False
                st.warning(
                    'To utilize iminuit for Maximum Likelihood Estimation, ensure iminuit is installed!',
                    icon='⚠️',
                )

    with run_col:
        if sampler in sampler_options:
            key = 'infer_resume'
            set_ini(key, 'Yes')
            resume_options = ['Yes', 'No']
            resume = st.selectbox(
                'Resume from previous run (samplers only)',
                resume_options,
                index=get_idx(key, resume_options),
                key=key,
            )
            resume = resume == 'Yes'
        else:
            resume = False

        key = 'infer_savepath'
        set_ini(key, '/home/appuser/Downloads/bsp')
        savepath = st.text_input(
            'Save results to',
            value=get_val(key),
            placeholder='/home/appuser/Downloads/bsp',
            key=key,
        )
        if not savepath:
            dirpath = get_download_folder()
            folder = f'bsp_{int(np.random.uniform() * 1e10)}'
            savepath = f'{dirpath}/{folder}'
        st.info(f'savepath: {savepath}')
        if os.path.exists(savepath):
            st.info('Note: this folder already exists — output will be merged in.')

        run = st.button(
            '🚀  Run inference',
            key='infer_run',
            type='primary',
            help='Launch the selected sampler / optimizer with the current settings.',
            use_container_width=True,
        )

        if run:
            if not sampler_exist:
                st.warning('Selected method backend is not installed!', icon='⚠️')
            else:
                run_panel = st.container(border=True)
                with run_panel, st.status('Running…', expanded=True) as status:
                    st.write(
                        f'Start: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}'
                    )
                    if not os.path.exists(savepath):
                        os.makedirs(savepath)

                    if sampler == 'multinest':
                        post = infer.multinest(
                            nlive=multinest_nlive,
                            resume=resume,
                            savepath=savepath,
                        )
                    elif sampler == 'emcee':
                        post = infer.emcee(
                            nstep=emcee_nstep,
                            discard=emcee_discard,
                            resume=resume,
                            savepath=savepath,
                        )
                    elif sampler in ('lmfit', 'iminuit'):
                        fit = MaxLikeFit(pair_list)
                        post = (
                            fit.lmfit(savepath=savepath)
                            if sampler == 'lmfit'
                            else fit.iminuit(savepath=savepath)
                        )
                    else:
                        post = None

                    st.write(
                        f'Stop: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}'
                    )
                    if post is not None:
                        st.session_state.infer_state['post'] = post
                    status.update(label='Run complete!', state='complete', expanded=False)
                st.rerun()

    post = st.session_state.infer_state.get('post')

    with post_col:
        with st.popover('📊  Posterior summary', use_container_width=True):
            if post is None:
                empty_card(
                    '🚀',
                    'No inference results yet',
                    'Run a sampler or optimizer to see results here.',
                )
            else:
                st.markdown('**Free parameters**')
                st.dataframe(
                    pd.DataFrame(post.free_par_info.data_dict),
                    use_container_width=True,
                    hide_index=True,
                )
                st.markdown('**Statistic**')
                st.dataframe(
                    pd.DataFrame(post.stat_info.data_dict),
                    use_container_width=True,
                    hide_index=True,
                )
                st.markdown('**Information criteria**')
                st.dataframe(
                    pd.DataFrame(post.IC_info.data_dict),
                    use_container_width=True,
                    hide_index=True,
                )

        with st.popover('🎯  Corner plot', use_container_width=True):
            if post is None:
                empty_card(
                    '🚀',
                    'No inference results yet',
                    'Run a Bayesian sampler to see the corner plot.',
                )
            else:
                fig = Plot.post_corner(post)
                st.plotly_chart(
                    fig.fig,
                    theme='streamlit',
                    use_container_width=True,
                    key='infer_corner_fig',
                )

        with st.popover('📈  Counts spectra', use_container_width=True):
            if post is None:
                empty_card(
                    '🚀',
                    'No inference results yet',
                    'Run a sampler or optimizer to see the fitted counts spectra.',
                )
            else:
                fig = Plot.infer(post, style='CE')
                st.plotly_chart(
                    fig.fig,
                    theme='streamlit',
                    use_container_width=True,
                    key='infer_ctsspec_fig',
                )

        with st.popover('🌊  Model spectra', use_container_width=True):
            if post is None:
                empty_card(
                    '🚀',
                    'No inference results yet',
                    'Run a sampler or optimizer to see model spectra.',
                )
            else:
                style_options = ['Fv', 'NE', 'vFv', 'NoU']
                key = 'post_model_style'
                set_ini(key, 'NE')
                style = st.selectbox(
                    'Spectral style',
                    style_options,
                    index=get_idx(key, style_options) or 0,
                    key=key,
                )
                if style is None:
                    style = 'NE'

                all_comps: dict = {}
                for cdict in st.session_state.model_component.values():
                    all_comps.update(cdict)

                nou_comps = {k: c for k, c in all_comps.items() if c.type in ('mul', 'math')}
                you_comps = {k: c for k, c in all_comps.items() if c.type == 'add'}

                if style in ('Fv', 'NE', 'vFv'):
                    comp_options = list(you_comps.keys())
                else:
                    comp_options = list(nou_comps.keys())

                key = 'post_model_comps'
                set_ini(key, [])
                default = [c for c in get_val(key) if c in comp_options]
                comp_keys = st.multiselect(
                    'Components to display',
                    options=comp_options,
                    default=default,
                    key=key,
                )

                if comp_keys:
                    modelplot = Plot.model(style=style, post=True)
                    comp_tabs = st.tabs([str(comp) for comp in comp_keys])
                    for comp_key, comp_tab in zip(comp_keys, comp_tabs, strict=False):
                        comp = all_comps[comp_key]
                        with comp_tab:
                            er_key = f'post_{comp_key}_erange'
                            set_ini(er_key, (0, 4))
                            erange = st.slider(
                                'Energy range (log10 keV)',
                                -1,
                                5,
                                value=get_val(er_key),
                                key=er_key,
                            )
                            earr = np.logspace(erange[0], erange[1], 300)

                            if comp.type == 'add':
                                ep_key = f'post_{comp_key}_epoch'
                                set_ini(ep_key, '')
                                epoch_str = st.text_input(
                                    'Spectral time (optional)',
                                    value=get_val(ep_key),
                                    placeholder='leave blank if time-independent',
                                    key=ep_key,
                                )
                                tarr = None
                                if epoch_str:
                                    try:
                                        tarr = float(epoch_str) * np.ones_like(earr)
                                    except (ValueError, TypeError):
                                        st.error(
                                            'Spectral time must be a number.',
                                            icon='🚨',
                                        )
                            else:
                                tarr = None

                            modelplot.add_model(comp, earr, tarr)

                    fig = modelplot.get_fig()
                    st.plotly_chart(
                        fig.fig,
                        theme='streamlit',
                        use_container_width=True,
                        key='infer_model_fig',
                    )
