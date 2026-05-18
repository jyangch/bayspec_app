from bayspec.data.data import Data, DataUnit
from bayspec.util.plot import Plot
import pandas as pd
import streamlit as st


def init_session_state():
    if "data" not in st.session_state:
        st.session_state.data = {}
    if "data_state" not in st.session_state:
        st.session_state.data_state = {}
    if "model" not in st.session_state:
        st.session_state.model = {}
    if "model_component" not in st.session_state:
        st.session_state.model_component = {}
    if "model_state" not in st.session_state:
        st.session_state.model_state = {}
    if "infer" not in st.session_state:
        st.session_state.infer = None
    if "infer_state" not in st.session_state:
        st.session_state.infer_state = {}


init_session_state()

st.markdown(
    '<span class="bsp-page-eyebrow">Stage 1 · Spectral data</span>'
    '<p class="bsp-subtitle">Upload OGIP spectra and configure noticing, grouping, '
    'and rebinning per data unit.</p>',
    unsafe_allow_html=True,
)


def set_ini(key, ini=None):
    if key not in st.session_state.data_state:
        st.session_state.data_state[key] = ini


def get_val(key):
    if key in st.session_state:
        st.session_state.data_state[key] = st.session_state[key]
    return st.session_state.data_state[key]


def get_idx(key, options):
    if key in st.session_state:
        st.session_state.data_state[key] = st.session_state[key]
    value = st.session_state.data_state[key]
    if (value is None) or (value not in options):
        return None
    else:
        return options.index(value)


def get_file(key, accept_multiple_files=False):
    if accept_multiple_files:
        if st.session_state[key] != []:
            st.session_state.data_state[key] = st.session_state[key]
        else:
            if st.session_state.data_state[key] != []:
                for file in st.session_state.data_state[key]:
                    st.write("📄", file.name)
    else:
        if st.session_state[key] is not None:
            st.session_state.data_state[key] = st.session_state[key]
        else:
            if st.session_state.data_state[key] is not None:
                st.write("📄", st.session_state.data_state[key].name)
    return st.session_state.data_state[key]


def reset_data():
    st.session_state.data = {}


with st.sidebar:
    st.markdown("##### 🔭 Data setup")
    key = "ndata"
    ini = "min"
    set_ini(key, ini)
    ndata = st.number_input(
        "Number of Data containers",
        min_value=1,
        value=get_val(key),
        key=key,
        on_change=reset_data,
        help="Each Data container groups one or more spectral units (DataUnits).",
    )

for i in range(ndata):
    st.session_state.data[f"Data{i + 1}"] = Data()


# ---- Summary header: containers / units / bindings --------------------
def _binding_for(dkey: str) -> str | None:
    """Return the model key bound to ``dkey``, or None."""
    mkey = st.session_state.data_state.get(f"{dkey}_model")
    if mkey is None:
        return None
    # Cross-check that the model side agrees, mirroring the Infer page rule.
    if st.session_state.model_state.get(f"{mkey}_data") != dkey:
        return None
    return mkey


total_units = sum(
    len(getattr(st.session_state.data.get(k, Data()), "data", {}))
    for k in st.session_state.data
)
bound_count = sum(
    1 for k in st.session_state.data if _binding_for(k) is not None
)

stat_a, stat_b, stat_c = st.columns(3)
stat_a.metric("Data containers", ndata)
stat_b.metric("DataUnits", total_units)
stat_c.metric(
    "Bound to a model",
    f"{bound_count} / {ndata}",
    help="Pairs ready to feed into the Inference page.",
)
st.write("")

for di, data_key in enumerate(st.session_state.data.keys()):
    bound_model = _binding_for(data_key)
    # The Data container is re-instantiated at the top of each rerun and
    # repopulated further down, so its .data dict is empty here. Read the
    # user-configured unit count from data_state for the expander label.
    n_units_here = int(
        st.session_state.data_state.get(f"{data_key}_nunit", 1) or 1
    )
    unit_tag = f"{n_units_here} unit{'s' if n_units_here != 1 else ''}"

    if bound_model:
        expander_title = f"**{data_key}** · {unit_tag} · ↔ **{bound_model}**"
        badge_html = (
            f'<div class="bsp-pair-row" style="margin:0 0 .85rem">'
            f'<span class="bsp-data-badge">{data_key}</span>'
            f'<span class="bsp-pair-arrow">↔</span>'
            f'<span class="bsp-model-badge">{bound_model}</span>'
            f'<span style="margin-left:auto;color:var(--bsp-success);'
            f'font-weight:600;font-size:.85rem">● bound</span>'
            f'</div>'
        )
    else:
        expander_title = f"**{data_key}** · {unit_tag} · — unbound —"
        badge_html = (
            f'<div class="bsp-pair-row" style="margin:0 0 .85rem">'
            f'<span class="bsp-data-badge">{data_key}</span>'
            f'<span class="bsp-pair-arrow">↔</span>'
            f'<span style="font-family:JetBrains Mono,monospace;font-size:.9rem;'
            f'color:var(--bsp-text-muted)">— unbound —</span>'
            f'<span style="margin-left:auto;color:var(--bsp-text-muted);'
            f'font-weight:600;font-size:.85rem">○ no model</span>'
            f'</div>'
        )

    with st.expander(expander_title, expanded=False):
        st.markdown(badge_html, unsafe_allow_html=True)
        nunit_col, _, fit_col = st.columns([4.9, 0.2, 4.9])

        with nunit_col:
            key = f"{data_key}_nunit"
            ini = "min"
            set_ini(key, ini)
            nunit = st.number_input(
                "Input the number of units of Data",
                min_value=1,
                value=get_val(key),
                key=key,
            )
        with fit_col:
            key = f"{data_key}_model"
            ini = None
            set_ini(key, ini)
            options = list(st.session_state.model.keys())
            model_key = st.selectbox(
                "Choose a Model to fit this Data",
                options,
                index=get_idx(key, options),
                key=key,
            )
            st.session_state.model_state[f"{model_key}_data"] = data_key

        unit_keys = [f"unit{di + 1}-{i + 1}" for i in range(nunit)]
        unit_tabs = st.tabs(unit_keys)

        for unit_key, unit_tab in zip(unit_keys, unit_tabs, strict=False):
            with unit_tab:
                set_col, _, info_col = st.columns([4.9, 0.2, 4.9])

                with set_col:
                    key = f"{data_key}_{unit_key}_expr"
                    ini = unit_key
                    set_ini(key, ini)
                    expr = st.text_input(
                        "Input dataunit expression",
                        value=get_val(key),
                        placeholder=unit_key,
                        key=key,
                    )
                    if expr is None or expr == "":
                        expr = unit_key

                    if expr in st.session_state.data[data_key]:
                        st.warning(
                            "Sorry for prohibiting the use of the same dataunit name",
                            icon="⚠️",
                        )

                    src = bkg = rsp = rmf = arf = None

                    key = f"{data_key}_{unit_key}_spec"
                    ini = []
                    set_ini(key, ini)
                    spec_files = st.file_uploader(
                        "Upload spectral files: src, bkg, rsp (or rmf and arf)",
                        accept_multiple_files=True,
                        key=key,
                    )
                    spec_files = get_file(key, True)
                    if spec_files is not None:
                        for speci in spec_files:
                            if "src" in speci.name or "pha" in speci.name:
                                src = speci
                            if "bkg" in speci.name or "bak" in speci.name:
                                bkg = speci
                            if "rsp" in speci.name or "resp" in speci.name:
                                rsp = speci
                            if "rmf" in speci.name:
                                rmf = speci
                            if "arf" in speci.name:
                                arf = speci

                    key = f"{data_key}_{unit_key}_src"
                    ini = None
                    set_ini(key, ini)
                    _ = st.file_uploader("Upload source spectrum: src", key=key)
                    if get_file(key) is not None:
                        src = get_file(key)

                    key = f"{data_key}_{unit_key}_bkg"
                    ini = None
                    set_ini(key, ini)
                    _ = st.file_uploader("Upload background spectrum: bkg", key=key)
                    if get_file(key) is not None:
                        bkg = get_file(key)

                    key = f"{data_key}_{unit_key}_rsp"
                    ini = None
                    set_ini(key, ini)
                    _ = st.file_uploader("Upload response matrix: rsp", key=key)
                    if get_file(key) is not None:
                        rsp = get_file(key)

                    key = f"{data_key}_{unit_key}_rmf"
                    ini = None
                    set_ini(key, ini)
                    _ = st.file_uploader("Upload redistribution matrix: rmf", key=key)
                    if get_file(key) is not None:
                        rmf = get_file(key)

                    key = f"{data_key}_{unit_key}_arf"
                    ini = None
                    set_ini(key, ini)
                    _ = st.file_uploader(
                        "Upload auxiliary response matrix: arf", key=key
                    )
                    if get_file(key) is not None:
                        arf = get_file(key)

                    key = f"{data_key}_{unit_key}_stat"
                    ini = "pgstat"
                    set_ini(key, ini)
                    options = [
                        "gstat",
                        "chi2",
                        "pstat",
                        "ppstat",
                        "cstat",
                        "pgstat",
                        "Xppstat",
                        "Xcstat",
                        "Xpgstat",
                        "ULppstat",
                        "ULpgstat",
                    ]
                    stat = st.selectbox(
                        "Choose fitting statistic metric: stat",
                        options,
                        index=get_idx(key, options),
                        key=key,
                    )

                    key = f"{data_key}_{unit_key}_notc"
                    ini = None
                    set_ini(key, ini)
                    notc = st.text_input(
                        "Input notice energy: notc",
                        value=get_val(key),
                        placeholder="8-30;40-1000 (defaults to None)",
                        key=key,
                    )
                    if notc == "":
                        notc = None
                    if notc is not None:
                        notc_list = notc.split(";")
                        notc = []
                        for notc_str in notc_list:
                            notc_range = notc_str.split("-")
                            if len(notc_range) == 2:
                                try:
                                    nt1 = float(notc_range[0].strip())
                                except (ValueError, TypeError):
                                    st.error(
                                        "The input value should be int or float!",
                                        icon="🚨",
                                    )
                                try:
                                    nt2 = float(notc_range[1].strip())
                                except (ValueError, TypeError):
                                    st.error(
                                        "The input value should be int or float!",
                                        icon="🚨",
                                    )
                                notc.append([nt1, nt2])
                            else:
                                st.error(
                                    "The input value is in a wrong format!", icon="🚨"
                                )

                    key = f"{data_key}_{unit_key}_grpg_evt"
                    ini = None
                    set_ini(key, ini)
                    grpg_min_evt = st.text_input(
                        "Input grouping minimum events: grpg_min_evt",
                        value=get_val(key),
                        placeholder="5 (defaults to None)",
                        key=key,
                    )
                    if grpg_min_evt == "":
                        grpg_min_evt = None
                    if grpg_min_evt is not None:
                        try:
                            grpg_min_evt = int(grpg_min_evt)
                        except (ValueError, TypeError):
                            st.error("The input value should be int!", icon="🚨")

                    key = f"{data_key}_{unit_key}_grpg_sig"
                    ini = None
                    set_ini(key, ini)
                    grpg_min_sigma = st.text_input(
                        "Set grouping minimum sigma: grpg_min_sigma",
                        value=get_val(key),
                        placeholder="3 (defaults to None)",
                        key=key,
                    )
                    if grpg_min_sigma == "":
                        grpg_min_sigma = None
                    if grpg_min_sigma is not None:
                        try:
                            grpg_min_sigma = float(grpg_min_sigma)
                        except (ValueError, TypeError):
                            st.error(
                                "The input value should be int or float!", icon="🚨"
                            )

                    key = f"{data_key}_{unit_key}_grpg_bin"
                    ini = None
                    set_ini(key, ini)
                    grpg_max_bin = st.text_input(
                        "Input grouping maximum bins: grpg_max_bin",
                        value=get_val(key),
                        placeholder="20 (defaults to None)",
                        key=key,
                    )
                    if grpg_max_bin == "":
                        grpg_max_bin = None
                    if grpg_max_bin is not None:
                        try:
                            grpg_max_bin = int(grpg_max_bin)
                        except (ValueError, TypeError):
                            st.error("The input value should be int!", icon="🚨")

                    if (
                        grpg_min_evt is None
                        and grpg_min_sigma is None
                        and grpg_max_bin is None
                    ):
                        grpg = None
                    else:
                        grpg = {
                            "min_evt": grpg_min_evt,
                            "min_sigma": grpg_min_sigma,
                            "max_bin": grpg_max_bin,
                        }

                    key = f"{data_key}_{unit_key}_rebn_evt"
                    ini = None
                    set_ini(key, ini)
                    rebn_min_evt = st.text_input(
                        "Input rebining minimum events: rebn_min_evt",
                        value=get_val(key),
                        placeholder="5 (defaults to None)",
                        key=key,
                    )
                    if rebn_min_evt == "":
                        rebn_min_evt = None
                    if rebn_min_evt is not None:
                        try:
                            rebn_min_evt = int(rebn_min_evt)
                        except (ValueError, TypeError):
                            st.error("The input value should be int!", icon="🚨")

                    key = f"{data_key}_{unit_key}_rebn_sig"
                    ini = None
                    set_ini(key, ini)
                    rebn_min_sigma = st.text_input(
                        "Set rebining minimum sigma: rebn_min_sigma",
                        value=get_val(key),
                        placeholder="3 (defaults to None)",
                        key=key,
                    )
                    if rebn_min_sigma == "":
                        rebn_min_sigma = None
                    if rebn_min_sigma is not None:
                        try:
                            rebn_min_sigma = float(rebn_min_sigma)
                        except (ValueError, TypeError):
                            st.error(
                                "The input value should be int or float!", icon="🚨"
                            )

                    key = f"{data_key}_{unit_key}_rebn_bin"
                    ini = None
                    set_ini(key, ini)
                    rebn_max_bin = st.text_input(
                        "Input rebining maximum bins: rebn_max_bin",
                        value=get_val(key),
                        placeholder="20 (defaults to None)",
                        key=key,
                    )
                    if rebn_max_bin == "":
                        rebn_max_bin = None
                    if rebn_max_bin is not None:
                        try:
                            rebn_max_bin = int(rebn_max_bin)
                        except (ValueError, TypeError):
                            st.error("The input value should be int!", icon="🚨")

                    if (
                        rebn_min_evt is None
                        and rebn_min_sigma is None
                        and rebn_max_bin is None
                    ):
                        rebn = None
                    else:
                        rebn = {
                            "min_evt": rebn_min_evt,
                            "min_sigma": rebn_min_sigma,
                            "max_bin": rebn_max_bin,
                        }

                    key = f"{data_key}_{unit_key}_time"
                    ini = None
                    set_ini(key, ini)
                    time = st.text_input(
                        "Input spectral time: time",
                        value=get_val(key),
                        placeholder="1.0 (defaults to None)",
                        key=key,
                    )
                    if time == "":
                        time = None
                    if time is not None:
                        try:
                            time = float(time)
                        except (ValueError, TypeError):
                            st.error(
                                "The input value should be int or float!", icon="🚨"
                            )

                    if src is not None:
                        dataunit = DataUnit(
                            src=src,
                            bkg=bkg,
                            rsp=rsp,
                            rmf=rmf,
                            arf=arf,
                            stat=stat,
                            notc=notc,
                            grpg=grpg,
                            rebn=rebn,
                            time=time,
                        )
                        if dataunit.completeness:
                            st.session_state.data[data_key][expr] = dataunit

                with info_col:
                    st.write("")
                    st.write("")

                    key = f"{data_key}_{unit_key}_info"
                    ini = False
                    set_ini(key, ini)
                    if st.checkbox("Show dataunit infomation", value=ini, key=key):
                        if src is None:
                            st.warning("dataunit does not exist!", icon="⚠️")
                        else:
                            properties = [
                                "src",
                                "bkg",
                                "rmf",
                                "arf",
                                "rsp",
                                "notc",
                                "stat",
                                "grpg",
                                "time",
                                "weight",
                                "grpg_min_evt",
                                "grpg_min_sigma",
                                "grpg_max_bin",
                                "rebn_min_evt",
                                "rebn_min_sigma",
                                "rebn_max_bin",
                            ]
                            raw_values = [
                                getattr(dataunit, "src_name", None),
                                getattr(dataunit, "bkg_name", None),
                                getattr(dataunit, "rmf_name", None),
                                getattr(dataunit, "arf_name", None),
                                getattr(dataunit, "rsp_name", None),
                                dataunit.notc,
                                dataunit.stat,
                                dataunit.grpg,
                                dataunit.time,
                                dataunit.weight,
                                grpg_min_evt,
                                grpg_min_sigma,
                                grpg_max_bin,
                                rebn_min_evt,
                                rebn_min_sigma,
                                rebn_max_bin,
                            ]
                            values = [
                                "" if v is None else str(v) for v in raw_values
                            ]
                            info = {"Property": properties, expr: values}
                            st.dataframe(
                                pd.DataFrame(info),
                                use_container_width=True,
                                hide_index=True,
                            )

                    with st.popover(
                        "📈  Display dataunit counts spectra", use_container_width=True
                    ):
                        if src is None:
                            st.warning("dataunit does not exist!", icon="⚠️")
                        else:
                            if not dataunit.completeness:
                                st.warning("dataunit is not complete!", icon="⚠️")
                            else:
                                fig = Plot.dataunit(dataunit, style="CE")

                                key = f"{data_key}_{unit_key}_fig"
                                st.plotly_chart(
                                    fig.fig,
                                    theme="streamlit",
                                    use_container_width=True,
                                    key=key,
                                )
