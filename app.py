import streamlit as st
from st_pages import add_page_title, get_nav_from_toml


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


st.set_page_config(layout="wide")

nav = get_nav_from_toml('.streamlit/pages.toml')

st.logo('.streamlit/logo.png')

pg = st.navigation(nav)

add_page_title(pg)

pg.run()

init_session_state()
