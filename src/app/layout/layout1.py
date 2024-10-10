import streamlit as st
from huggingface_hub import HfApi

from ..components.utils import initialize_hg_api_key
from ..features import research_assistant


def page_1():
    st.markdown('<div class="header">#1 Something_</div>', unsafe_allow_html=True)
    st.text("")
    st.text("Here is the page 2")
    st.markdown("---")

    api = HfApi()
    hg_api_key = initialize_hg_api_key()

    debug = st.sidebar.toggle("Debug mode")
    st.toast("Debug mode is on" if debug else "Debug mode is off")

    st.sidebar.write(f"## Connexion test")
    st.sidebar.write(api.whoami(hg_api_key))

    research_assistant.run_assistance(hg_api_key, debug=debug)
