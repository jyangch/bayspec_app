import importlib
import json
import os
import re

from bayspec.util.plot import Plot
from bayspec.util.prior import all_priors
from code_editor import code_editor
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
    '<p class="bsp-subtitle">Compose spectral models from local, Astromodels, XSPEC '
    'or your own components, then bind each to a Data container.</p>',
    unsafe_allow_html=True,
)


def set_ini(key, ini=None):
    if key not in st.session_state.model_state:
        st.session_state.model_state[key] = ini


def get_val(key):
    if key in st.session_state:
        st.session_state.model_state[key] = st.session_state[key]
    return st.session_state.model_state[key]


def get_data(key):
    if key in st.session_state:
        for row, edited in st.session_state[key]['edited_rows'].items():
            for col, value in edited.items():
                st.session_state.model_state[key].loc[int(row), col] = value
    return st.session_state.model_state[key]


def get_resp(key):
    if (
        key in st.session_state
        and st.session_state[key] is not None
        and st.session_state[key]['type'] in ['submit', 'saved']
    ):
        st.session_state.model_state[key] = st.session_state[key]['text']
    return st.session_state.model_state[key]


def get_idx(key, options):
    if key in st.session_state:
        st.session_state.model_state[key] = st.session_state[key]
    value = st.session_state.model_state[key]
    if (value is None) or (value not in options):
        return None
    else:
        return options.index(value)


def reset_model():
    st.session_state.model = {}
    st.session_state.model_component = {}


def pop_key(keys):
    for key in keys:
        if key in st.session_state:
            _ = st.session_state.pop(key)
        if key in st.session_state.model_state:
            _ = st.session_state.model_state.pop(key)
        if key in st.session_state.infer_state:
            _ = st.session_state.infer_state.pop(key)


with st.sidebar:
    st.markdown('##### 🌈 Model setup')
    key = 'nmodel'
    ini = 'min'
    set_ini(key, ini)
    nmodel = st.number_input(
        'Number of Models',
        min_value=1,
        value=get_val(key),
        key=key,
        on_change=reset_model,
        help='Each Model is a composite of one or more spectral components.',
    )

for i in range(nmodel):
    st.session_state.model[f'Model{i + 1}'] = None
for i in range(nmodel):
    st.session_state.model_component[f'Model{i + 1}'] = {}

for mi, model_key in enumerate(st.session_state.model.keys()):
    with st.expander(f'***Configure the {model_key}***', expanded=False):
        ncomponent_col, _, fit_col = st.columns([4.9, 0.2, 4.9])

        with ncomponent_col:
            key = f'{model_key}_ncomponent'
            ini = 'min'
            set_ini(key, ini)
            ncomponent = st.number_input(
                'Input the number of components of model',
                min_value=1,
                value=get_val(key),
                key=key,
                on_change=pop_key,
                args=([f'{model_key}_expression'],),
            )

        with fit_col:
            key = f'{model_key}_data'
            ini = None
            set_ini(key, ini)
            options = list(st.session_state.data.keys())
            data_key = st.selectbox(
                'Choose a Data to fit with this Model',
                options,
                index=get_idx(key, options),
                key=key,
            )
            st.session_state.data_state[f'{data_key}_model'] = model_key

        component_keys = [f'component{mi + 1}-{i + 1}' for i in range(ncomponent)]
        expression_key = 'model expression'
        all_tabs = st.tabs([*component_keys, expression_key])
        component_tabs = all_tabs[:-1]
        expression_tab = all_tabs[-1]

        for component_key, component_tab in zip(component_keys, component_tabs, strict=False):
            with component_tab:
                set_col, _, info_col = st.columns([4.9, 0.2, 4.9])

                with set_col:
                    key = f'{model_key}_{component_key}_library'
                    ini = None
                    set_ini(key, ini)
                    options = ['local', 'astro', 'xspec', 'user']
                    library = st.selectbox(
                        'Choose model library',
                        options,
                        index=get_idx(key, options),
                        key=key,
                        on_change=pop_key,
                        args=(
                            [
                                f'{model_key}_{component_key}_name',
                                f'{model_key}_{component_key}_expr',
                                f'{model_key}_{component_key}_cfg',
                                f'{model_key}_{component_key}_par',
                                f'{model_key}_expression',
                            ],
                        ),
                    )

                    if library is None:
                        library_keys = []
                    elif library == 'local':
                        from bayspec.model.local import local_models

                        library_dict = local_models
                        library_keys = list(local_models.keys())
                    elif library == 'astro':
                        try:
                            from bayspec.model.astro import astro_models
                        except ImportError:
                            library_keys = []
                            st.warning(
                                'To utilize models from Astromodels, ensure Astromodels is installed!',
                                icon='⚠️',
                            )
                        else:
                            library_dict = astro_models
                            library_keys = list(astro_models.keys())
                    elif library == 'xspec':
                        try:
                            from bayspec.model.xspec import abund, xsect, xspec_models
                        except ImportError:
                            library_keys = []
                            st.warning(
                                'To utilize models from Xspec, ensure HEASoft and Xspec are installed!',
                                icon='⚠️',
                            )
                        else:
                            library_dict = xspec_models
                            library_keys = list(xspec_models.keys())
                    elif library == 'user':
                        library_keys = []
                    else:
                        pass

                    key = f'{model_key}_{component_key}_name'
                    ini = None
                    set_ini(key, ini)
                    name = st.selectbox(
                        'Choose model component',
                        library_keys,
                        index=get_idx(key, library_keys),
                        key=key,
                        on_change=pop_key,
                        args=(
                            [
                                f'{model_key}_{component_key}_expr',
                                f'{model_key}_{component_key}_cfg',
                                f'{model_key}_{component_key}_par',
                                f'{model_key}_expression',
                            ],
                        ),
                    )

                    if library is None:
                        expr = component_key
                        component = None

                    elif library == 'user':
                        info = """**Note: Please make sure to back up yourself defined model,
                        as this APP will not save it. If you want to use it as a build-in model
                        of this APP, please contact the APP author (jyang@smail.nju.edu.cn).**"""
                        st.info(info)

                        with open('.streamlit/custom_buttons_bar_alt.json') as json_button_file_alt:
                            custom_buttons_alt = json.load(json_button_file_alt)
                        with open('.streamlit/info_bar.json') as json_info_file:
                            info_bar = json.load(json_info_file)
                        with open('.streamlit/code_editor_css.scss') as css_file:
                            css_text = css_file.read()

                        comp_props = {
                            'css': css_text,
                            'globalCSS': ':root {\n  --streamlit-dark-font-family: monospace;\n}',
                        }
                        ace_props = {'style': {'borderRadius': '0px 0px 8px 8px'}}

                        user_dir = os.path.dirname(os.path.abspath(__file__)) + '/model'
                        with open(user_dir + '/user.py') as file_obj:
                            model_format = file_obj.read()

                        key = f'{model_key}_{component_key}_user_model'
                        ini = model_format
                        set_ini(key, ini)
                        response_dict = code_editor(
                            get_resp(key),
                            height=[30],
                            lang='python',
                            theme='default',
                            shortcuts='vscode',
                            focus=False,
                            buttons=custom_buttons_alt,
                            info=info_bar,
                            component_props=comp_props,
                            props=ace_props,
                            options={'wrap': True},
                            key=key,
                        )

                        if response_dict['type'] == 'submit' and len(response_dict['id']) != 0:
                            st.info('Note: you have submitted you model!')

                            key = f'{model_key}_{component_key}_user_fname'
                            ini = f'user_{model_key}_{component_key}'
                            set_ini(key, ini)
                            user_fname = get_val(key)
                            with open(user_dir + f'/{user_fname}.py', 'w') as file_obj:
                                file_obj.write(response_dict['text'])

                            component = importlib.import_module(
                                f'bayspec.model.user.{user_fname}'
                            ).user()
                            expr = component.expr
                        else:
                            expr = component_key
                            component = None

                    else:
                        if name is None:
                            expr = component_key
                            component = None
                        else:
                            if library == 'xspec':
                                options = [
                                    'angr',
                                    'aspl',
                                    'feld',
                                    'aneb',
                                    'grsa',
                                    'wilm',
                                    'lodd',
                                    'lpgp',
                                ]
                                key = f'{model_key}_{component_key}_abund'
                                ini = None
                                set_ini(key, ini)
                                abundance = st.selectbox(
                                    'Choose xspec abundance',
                                    options,
                                    index=get_idx(key, options),
                                    key=key,
                                )
                                if abundance is None:
                                    abundance = 'wilm'
                                if abundance is not None:
                                    abund(abundance)

                                options = ['bcmc', 'obcm', 'vern']
                                key = f'{model_key}_{component_key}_xsect'
                                ini = None
                                set_ini(key, ini)
                                section = st.selectbox(
                                    'Choose xspec cross-section',
                                    options,
                                    index=get_idx(key, options),
                                    key=key,
                                )
                                if section is None:
                                    section = 'vern'
                                if section is not None:
                                    xsect(section)

                            component = library_dict[name]()

                            key = f'{model_key}_{component_key}_expr'
                            ini = component.expr
                            set_ini(key, ini)
                            expr = st.text_input(
                                'Input model component name',
                                value=get_val(key),
                                placeholder=component.expr,
                                key=key,
                            )
                            if expr is None or expr == '':
                                expr = component.expr
                            if expr in st.session_state.model_component[model_key]:
                                st.warning(
                                    'Sorry for prohibiting the use of the same component name',
                                    icon='⚠️',
                                )
                            component.expr = expr

                            cfg_data = dict(component.cfg_info.data_dict)
                            cfg_data['Value'] = [float(v) for v in cfg_data['Value']]
                            cfg_df = pd.DataFrame(cfg_data)
                            key = f'{model_key}_{component_key}_cfg'
                            ini = cfg_df
                            set_ini(key, ini)
                            cfg_df = st.data_editor(
                                get_data(key),
                                column_config={
                                    'Value': st.column_config.NumberColumn(format='%g'),
                                },
                                use_container_width=True,
                                num_rows='fixed',
                                disabled=['cfg#', 'Component', 'Parameter'],
                                hide_index=True,
                                key=key,
                            )

                            for _, row in cfg_df.to_dict('index').items():
                                cfg_obj = component.cfg[int(row['cfg#'])]
                                orig = cfg_obj.val
                                try:
                                    if isinstance(orig, bool):
                                        new_val = bool(row['Value'])
                                    elif isinstance(orig, int):
                                        new_val = int(row['Value'])
                                    else:
                                        new_val = float(row['Value'])
                                except (ValueError, TypeError):
                                    st.error(
                                        f'Invalid cfg value for {row["Parameter"]}: {row["Value"]!r}',
                                        icon='🚨',
                                    )
                                    continue
                                cfg_obj.val = new_val

                            par_df = pd.DataFrame(component.par_info.data_dict)
                            plabels_sig = ','.join(component.par_info.data_dict['Parameter'])
                            key = f'{model_key}_{component_key}_par|{plabels_sig}'
                            ini = par_df
                            set_ini(key, ini)
                            par_df = st.data_editor(
                                get_data(key),
                                use_container_width=True,
                                num_rows='fixed',
                                disabled=['par#', 'Component', 'Parameter'],
                                hide_index=True,
                                key=key,
                            )

                            for _, row in par_df.to_dict('index').items():
                                par_obj = component.par[int(row['par#'])]
                                par_obj.val = row['Value']

                                prior_str = str(row['Prior']).strip()
                                if prior_str == 'frozen':
                                    par_obj.frozen = True
                                else:
                                    par_obj.frozen = False
                                    prior_info = [s.strip() for s in re.split(r'[(,)]', prior_str)]
                                    prior = prior_info[0]
                                    args = [float(s) for s in prior_info[1:-1]]
                                    if prior not in all_priors:
                                        st.error(f'{prior} is not one of priors!', icon='🚨')
                                    else:
                                        par_obj.prior = all_priors[prior](*args)

                    st.session_state.model_component[model_key][expr] = component

                with info_col:
                    st.write('')
                    st.write('')

                    key = f'{model_key}_{component_key}_info'
                    ini = False
                    set_ini(key, ini)
                    if st.checkbox('Show model component infomation', value=ini, key=key):
                        if component is None:
                            if library == 'user':
                                st.warning(
                                    'The user-defined model component has not been submitted!',
                                    icon='⚠️',
                                )
                            else:
                                st.warning('The model component has not been set!', icon='⚠️')
                        else:
                            st.info(f'{component.expr} [{component.type}]')
                            st.info(component.comment)

                    with st.popover('📈  Display model spectra', use_container_width=True):
                        if component is None:
                            if library == 'user':
                                st.warning(
                                    'The user-defined model component has not been submitted!',
                                    icon='⚠️',
                                )
                            else:
                                st.warning('The model component has not been set!', icon='⚠️')
                        else:
                            if component.type in ['mul', 'math']:
                                options = ['NoU']
                            elif component.type == 'add':
                                options = ['Fv', 'NE', 'vFv']
                            else:
                                options = []

                            key = f'{model_key}_{component_key}_style'
                            ini = None
                            set_ini(key, ini)
                            style = st.selectbox(
                                'Select spectral style to display',
                                options,
                                index=ini,
                                key=key,
                            )

                            key = f'{model_key}_{component_key}_erange'
                            ini = (0, 4)
                            set_ini(key, ini)
                            erange = st.slider(
                                'Select energy range in logspace',
                                -1,
                                5,
                                value=ini,
                                key=key,
                            )
                            earr = np.logspace(erange[0], erange[1], 300)

                            if component.type == 'add':
                                key = f'{model_key}_{component_key}_epoch'
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

                            if style is not None:
                                modelplot = Plot.model(style=style, post=False)
                                modelplot.add_model(component, earr, tarr)
                                fig = modelplot.get_fig()

                                key = f'{model_key}_{component_key}_fig'
                                st.plotly_chart(
                                    fig.fig,
                                    theme='streamlit',
                                    use_container_width=True,
                                    key=key,
                                )

        with expression_tab:
            set_col, _, info_col = st.columns([4.9, 0.2, 4.9])

            with set_col:
                info = """**Note: The model expression defines a combined model
                involved with multiple components, which is also the model used
                in the fitting.**"""
                st.info(info)

                key = f'{model_key}_expression'
                ini = None
                set_ini(key, ini)
                placeholder = '+'.join(st.session_state.model_component[model_key].keys())
                expression = st.text_input(
                    'Input model expression',
                    value=get_val(key),
                    placeholder=placeholder,
                    key=key,
                )
                if expression == '':
                    expression = None

                if expression is not None:
                    expression = re.sub(r'\s*', '', expression)
                    expression_sp = re.split(r'[(+\-*/)]', expression)
                    expression_sp = [ex for ex in expression_sp if ex != '']
                    if len(set(expression_sp)) < len(expression_sp):
                        st.warning(
                            'Sorry for prohibiting the use of the same component name!',
                            icon='⚠️',
                        )
                    elif not (
                        set(expression_sp)
                        <= set(st.session_state.model_component[model_key].keys())
                    ):
                        st.warning(
                            'The model expression include invalid component name!',
                            icon='⚠️',
                        )
                    elif None in [
                        st.session_state.model_component[model_key][ex] for ex in expression_sp
                    ]:
                        st.warning('Some model components have not been set!', icon='⚠️')
                    else:
                        model = eval(expression, {}, st.session_state.model_component[model_key])
                        st.session_state.model[model_key] = model
                        st.session_state.model_component[model_key][expression] = model

                        cfg_df = pd.DataFrame(model.cfg_info.data_dict)
                        key = f'{model_key}_cfg'
                        cfg_df = st.data_editor(
                            cfg_df,
                            use_container_width=True,
                            num_rows='fixed',
                            disabled=True,
                            hide_index=True,
                            key=key,
                        )

                        par_df = pd.DataFrame(model.par_info.data_dict)
                        key = f'{model_key}_par'
                        par_df = st.data_editor(
                            par_df,
                            use_container_width=True,
                            num_rows='fixed',
                            disabled=True,
                            hide_index=True,
                            key=key,
                        )

            with info_col:
                st.write('')
                st.write('')

                key = f'{model_key}_info'
                ini = False
                set_ini(key, ini)
                if st.checkbox('Show model infomation', value=ini, key=key):
                    if st.session_state.model[model_key] is None:
                        st.warning('The model has not been set!', icon='⚠️')
                    else:
                        st.info(f'{model.expr} [{model.type}]')
                        for comment in model.comment.split('\n'):
                            st.info(comment)

                with st.popover('📈  Display model spectra', use_container_width=True):
                    if st.session_state.model[model_key] is None:
                        st.warning('The model has not been set!', icon='⚠️')
                    elif None in list(st.session_state.model_component[model_key].values()):
                        st.warning('Some model components have not been set!', icon='⚠️')
                    else:
                        key = f'{model_key}_style'
                        ini = None
                        set_ini(key, ini)
                        options = ['Fv', 'NE', 'vFv', 'NoU']
                        style = st.selectbox(
                            'Select spectral style to display',
                            options,
                            index=ini,
                            key=key,
                        )

                        all_comps = st.session_state.model_component[model_key]

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

                        key = f'{model_key}_comps'
                        ini = None
                        set_ini(key, ini)
                        comp_keys = st.multiselect(
                            'Select the model components to display',
                            options=options,
                            default=ini,
                            key=key,
                        )

                        if len(comp_keys) > 0:
                            modelplot = Plot.model(style=style, post=False)

                            comp_tabs = st.tabs([str(comp) for comp in comp_keys])
                            for comp_key, comp_tab in zip(comp_keys, comp_tabs, strict=False):
                                comp = all_comps[comp_key]
                                with comp_tab:
                                    key = f'{model_key}_{comp_key}_erange'
                                    ini = (0, 4)
                                    set_ini(key, ini)
                                    erange = st.slider(
                                        'Select energy range in logspace',
                                        -1,
                                        5,
                                        value=ini,
                                        key=key,
                                    )
                                    earr = np.logspace(erange[0], erange[1], 300)

                                    if comp.type == 'add':
                                        key = f'{model_key}_{comp_key}_epoch'
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

                            key = f'{model_key}_fig'
                            st.plotly_chart(
                                fig.fig,
                                theme='streamlit',
                                use_container_width=True,
                                key=key,
                            )
