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


with st.sidebar:
    st.markdown('##### 📝 Fitting workflow')
    st.caption(
        '1. Pair Data ↔ Model · 2. Manual fit · 3. Run inference · 4. Inspect posterior'
    )

st.session_state.infer = None
st.session_state.infer_state['infer_pair_flag'] = False

all_pairs = {}
for data_key in st.session_state.data:
    model_key = st.session_state.data_state[f'{data_key}_model']
    if (
        model_key is not None
        and st.session_state.model_state[f'{model_key}_data'] == data_key
    ):
        all_pairs[f'{data_key}🔗{model_key}'] = [data_key, model_key]

with st.expander('***Set fitting pairs***', expanded=True):
    key = 'infer_pairs'
    ini = list(all_pairs.keys())
    set_ini(key, ini)
    options = list(all_pairs.keys())
    pairs = st.multiselect('Select infer pairs', options=options, default=get_val(key), key=key)

    if len(pairs) > 0:
        st.session_state.infer_state['infer_pair_flag'] = True

        pair_list = list()
        for pair in pairs:
            data_key, model_key = all_pairs[pair]
            pair_list.append([st.session_state.data[data_key], st.session_state.model[model_key]])

        infer = BayesInfer(pair_list)
        st.session_state.infer = infer

    cfg_col, _, par_col = st.columns([4.9, 0.2, 4.9])

    with cfg_col, st.popover('⚙️  Configurations', use_container_width=True):
        if not st.session_state.infer_state['infer_pair_flag']:
            empty_card('🔗', 'No fitting pair selected', 'Pick at least one Data ↔ Model pair above to continue.')
        else:
            cfg_df = pd.DataFrame(infer.cfg_info.data_dict)
            key = 'infer_cfg'
            cfg_df = st.data_editor(
                cfg_df,
                use_container_width=True,
                num_rows='fixed',
                disabled=True,
                hide_index=True,
                key=key,
            )

    with par_col, st.popover('🔗  Parameters & links', use_container_width=True):
        if not st.session_state.infer_state['infer_pair_flag']:
            empty_card('🔗', 'No fitting pair selected', 'Pick at least one Data ↔ Model pair above to continue.')
        else:
            key = 'infer_nlink'
            ini = 'min'
            set_ini(key, ini)
            nlink = st.number_input(
                'Input the number of linking parameters',
                min_value=0,
                value=ini,
                key=key,
            )

            for idx in range(nlink):
                key = f'infer_link_{idx}'
                ini = None
                set_ini(key, ini)
                options = [f'par#{p}' for p in list(infer.par.keys())]
                pids = st.multiselect(
                    'Select the parameters to link',
                    options=options,
                    default=ini,
                    key=key,
                )
                if len(pids) > 1:
                    infer.link([int(pi[4:]) for pi in pids])

            par_df = pd.DataFrame(infer.notable_par_info.data_dict)
            key = 'notable_infer_par'
            par_df = st.data_editor(
                par_df,
                use_container_width=True,
                num_rows='fixed',
                disabled=True,
                hide_index=True,
                key=key,
            )

with st.expander('***Manual fitting***', expanded=False):
    if not st.session_state.infer_state['infer_pair_flag']:
        empty_card('🔗', 'No fitting pair selected', 'Pick at least one Data ↔ Model pair above to continue.')
    else:
        free_par_df = pd.DataFrame(infer.free_par_info.data_dict)
        key = 'manual_free_par'
        ini = free_par_df
        set_ini(key, ini)
        free_par_df = st.data_editor(
            get_data(key),
            use_container_width=True,
            num_rows='fixed',
            disabled=['par#', 'Class', 'Expression', 'Component', 'Parameter', 'Prior'],
            hide_index=True,
            key=key,
        )
        now_par = list()
        for _, row in free_par_df.to_dict('index').items():
            now_par.append(row['Value'])
        infer.at_par(now_par)

        stat_col, _, plot_col = st.columns([4.9, 0.2, 4.9])

        with stat_col:
            stat_df = pd.DataFrame(infer.stat_info.data_dict)
            key = 'manual_stat'
            stat_df = st.data_editor(
                stat_df,
                use_container_width=True,
                num_rows='fixed',
                disabled=True,
                hide_index=True,
                key=key,
            )

        with plot_col:
            fig = Plot.infer(infer, style='CE')

            key = 'manual_ctsspec_fig'
            st.plotly_chart(fig.fig, theme='streamlit', use_container_width=True, key=key)

with st.expander('***Inference***', expanded=False):
    run_col, _, post_col = st.columns([4.9, 0.2, 4.9])

    with run_col:
        with st.popover('🛠️  Method settings', use_container_width=True):
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

            if sampler == 'multinest':
                try:
                    import pymultinest  # noqa: F401
                except ImportError:
                    sampler_exist = False
                    st.warning(
                        'To utilize Multinest for Bayesian inference, ensure Multinest is installed!',
                        icon='⚠️',
                    )
                else:
                    sampler_exist = True

                key = 'infer_multinest_nlive'
                ini = 300
                set_ini(key, ini)
                multinest_nlive = st.slider(
                    'Select the number of live point',
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
                else:
                    sampler_exist = True

                key = 'infer_emcee_nstep'
                ini = 2000
                set_ini(key, ini)
                emcee_nstep = st.slider(
                    'Select the number of steps',
                    0,
                    10000,
                    value=get_val(key),
                    step=1000,
                    key=key,
                )

                key = 'infer_emcee_discard'
                ini = 100
                set_ini(key, ini)
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
                else:
                    sampler_exist = True

            if sampler == 'iminuit':
                try:
                    import iminuit  # noqa: F401
                except ImportError:
                    sampler_exist = False
                    st.warning(
                        'To utilize iminuit for Maximum Likelihood Estimation, ensure iminuit is installed!',
                        icon='⚠️',
                    )
                else:
                    sampler_exist = True

        if sampler in sampler_options:
            key = 'infer_resume'
            ini = 'Yes'
            set_ini(key, ini)
            options = ['Yes', 'No']
            resume = st.selectbox(
                'Choose to resume or not (samplers only)',
                options,
                index=get_idx(key, options),
                key=key,
            )
            resume = resume == 'Yes'
        else:
            resume = False

        key = 'infer_savepath'
        ini = '/home/appuser/Downloads/bsp'
        set_ini(key, ini)
        savepath = st.text_input(
            'Input the path to save the results',
            value=get_val(key),
            placeholder='/home/appuser/Downloads/bsp',
            key=key,
        )
        if savepath == '' or savepath is None:
            dirpath = get_download_folder()
            folder = f'bsp_{int(np.random.uniform() * 1e10)}'
            savepath = f'{dirpath}/{folder}'
        st.info(f'savepath: {savepath}')

        if os.path.exists(savepath):
            st.info('Note: the folder of results has already existed!')

        key = 'infer_run'
        run = st.button(
            '🚀  Run inference',
            key=key,
            type='primary',
            help='Launch the selected sampler / optimizer with the current settings.',
            use_container_width=True,
        )

        if run:
            if not st.session_state.infer_state['infer_pair_flag']:
                empty_card('🔗', 'No fitting pair selected', 'Pick at least one Data ↔ Model pair above to continue.')
            elif not sampler_exist:
                st.warning('Selected method backend is not installed!', icon='⚠️')
            else:
                with st.sidebar.status('Running...', expanded=True) as status:
                    st.write(f'Start: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')

                    if not os.path.exists(savepath):
                        os.makedirs(savepath)

                    if sampler == 'multinest':
                        post = infer.multinest(
                            nlive=multinest_nlive, resume=resume, savepath=savepath
                        )
                    elif sampler == 'emcee':
                        post = infer.emcee(nstep=emcee_nstep, resume=resume, savepath=savepath)
                    elif sampler in ('lmfit', 'iminuit'):
                        fit = MaxLikeFit(pair_list)
                        if sampler == 'lmfit':
                            post = fit.lmfit(savepath=savepath)
                        else:
                            post = fit.iminuit(savepath=savepath)

                    st.write(f'Stop: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
                    st.session_state.infer_state['post'] = post
                    status.update(label='Run complete!', state='complete', expanded=False)
        else:
            if 'post' in st.session_state.infer_state:
                post = st.session_state.infer_state['post']

    with post_col:
        with st.popover('📊  Posterior summary', use_container_width=True):
            if 'post' not in st.session_state.infer_state:
                empty_card('🚀', 'No inference results yet', 'Run a sampler or optimizer above to see results here.')
            else:
                free_par_df = pd.DataFrame(post.free_par_info.data_dict)
                key = 'post_free_par'
                free_par_df = st.data_editor(
                    free_par_df,
                    use_container_width=True,
                    num_rows='fixed',
                    disabled=True,
                    hide_index=True,
                    key=key,
                )

                stat_df = pd.DataFrame(post.stat_info.data_dict)
                key = 'post_stat'
                stat_df = st.data_editor(
                    stat_df,
                    use_container_width=True,
                    num_rows='fixed',
                    disabled=True,
                    hide_index=True,
                    key=key,
                )

                IC_df = pd.DataFrame(post.IC_info.data_dict)
                key = 'post_IC'
                IC_df = st.data_editor(
                    IC_df,
                    use_container_width=True,
                    num_rows='fixed',
                    disabled=True,
                    hide_index=True,
                    key=key,
                )

        with st.popover('🎯  Corner plot', use_container_width=True):
            if 'post' not in st.session_state.infer_state:
                empty_card('🚀', 'No inference results yet', 'Run a sampler or optimizer above to see results here.')
            else:
                fig = Plot.post_corner(post)

                key = 'infer_corner_fig'
                st.plotly_chart(fig.fig, theme='streamlit', use_container_width=True, key=key)

        with st.popover('📈  Counts spectra', use_container_width=True):
            if 'post' not in st.session_state.infer_state:
                empty_card('🚀', 'No inference results yet', 'Run a sampler or optimizer above to see results here.')
            else:
                fig = Plot.infer(post, style='CE')

                key = 'infer_ctsspec_fig'
                st.plotly_chart(fig.fig, theme='streamlit', use_container_width=True, key=key)

        with st.popover('🌊  Model spectra', use_container_width=True):
            if 'post' not in st.session_state.infer_state:
                empty_card('🚀', 'No inference results yet', 'Run a sampler or optimizer above to see results here.')
            else:
                key = 'post_model_style'
                ini = None
                set_ini(key, ini)
                options = ['Fv', 'NE', 'vFv', 'NoU']
                style = st.selectbox(
                    'Select spectral style to display', options, index=ini, key=key
                )

                all_comps = dict()
                for cdict in st.session_state.model_component.values():
                    all_comps.update(cdict)

                nou_comps = dict()
                you_comps = dict()
                for key, comp in all_comps.items():
                    if comp.type in ['mul', 'math']:
                        nou_comps[key] = comp
                    if comp.type == 'add':
                        you_comps[key] = comp

                if style in ['Fv', 'NE', 'vFv']:
                    options = list(you_comps.keys())
                elif style in ['NoU']:
                    options = list(nou_comps.keys())
                else:
                    options = []

                key = 'post_model_comps'
                ini = None
                set_ini(key, ini)
                comp_keys = st.multiselect(
                    'Select model components to display',
                    options=options,
                    default=ini,
                    key=key,
                )

                if len(comp_keys) > 0:
                    modelplot = Plot.model(style=style, post=True)

                    comp_tabs = st.tabs([str(comp) for comp in comp_keys])
                    for comp_key, comp_tab in zip(comp_keys, comp_tabs, strict=False):
                        comp = all_comps[comp_key]
                        with comp_tab:
                            key = f'post_{comp_key}_erange'
                            ini = (0, 4)
                            set_ini(key, ini)
                            erange = st.slider(
                                'Select energy range in logspace',
                                -1,
                                5,
                                value=(0, 4),
                                key=key,
                            )
                            earr = np.logspace(erange[0], erange[1], 300)

                            if comp.type == 'add':
                                key = f'post_{comp_key}_epoch'
                                ini = None
                                set_ini(key, ini)
                                epoch = st.text_input(
                                    'Input spectral time (optional)',
                                    value=ini,
                                    placeholder='leave blank if time-independent',
                                    key=key,
                                )
                                if epoch == '' or epoch is None:
                                    tarr = None
                                else:
                                    try:
                                        epoch = float(epoch)
                                    except (ValueError, TypeError):
                                        st.error(
                                            'The input value should be int or float!',
                                            icon='🚨',
                                        )
                                        tarr = None
                                    else:
                                        tarr = epoch * np.ones_like(earr)
                            else:
                                tarr = None

                        modelplot.add_model(comp, earr, tarr)

                    fig = modelplot.get_fig()

                    key = 'infer_model_fig'
                    st.plotly_chart(fig.fig, theme='streamlit', use_container_width=True, key=key)
