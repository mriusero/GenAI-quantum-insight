import streamlit as st
import os

from ..components.utils import load_data

def page_0():
    st.markdown('<div class="header">#0 Project Overview_</div>', unsafe_allow_html=True)
    st.text("")
    st.text("Here is an overview of the project")
    st.markdown("___")

    if 'data' not in st.session_state:
        st.session_state['data'] = load_data()

    data = st.session_state['data']
    st.write(data)



