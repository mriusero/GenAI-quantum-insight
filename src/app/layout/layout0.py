import streamlit as st

from ..components.utils import load_data


def page_0():
    st.markdown('<div class="header">#0 Project Overview_</div>', unsafe_allow_html=True)
    st.text("")
    st.text("Here is an overview of the project")
    st.markdown("___")

    data = load_data()
    st.write(data)




