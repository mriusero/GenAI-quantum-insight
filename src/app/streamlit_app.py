import gc
import os

import src.app.features.arxiv_data_manager as arxiv        # Maintenance mode
import streamlit as st


def load_css():
    """Load custom CSS styles for the Streamlit app."""
    css_path = os.path.join(os.path.dirname(__file__), 'styles.css')
    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def main_layout():
    """Set up the main layout of the Streamlit application."""
    from .components import github_button
    from .layout import page_0, page_1, page_2, page_3, page_4, page_5, page_6

    st.set_page_config(
        page_title="Quantum Insights Agent",
        layout='wide',
        initial_sidebar_state="auto",
    )

    load_css()

    st.sidebar.markdown("# GenerativeAI \n"
                        " ## *Quantum Insights Agent*\n")

    page = st.sidebar.radio("Table of contents_", ["#0 Introduction_",
                                                   "#1 Something_",
                                                   "#2 Something_",
                                                   "#3 Something_",
                                                   "#4 Something_",
                                                   "#5 Something_",
                                                   "#6 Something_"
                                                   ])

    # -- LAYOUT --
    col1, col2 = st.columns([8, 4])
    with col1:
        global update_message
        st.markdown('<div class="title"> Quantum Insights</div>', unsafe_allow_html=True)

        colA, colB, colC = st.columns([2, 11, 1])

        with colA:
            github_button('https://github.com/mriusero/GenAI-quantum-insight')  # GitHub button

        with colB:
            st.text("")
            st.markdown("#### *Quantum Computing Vulgarisation Agent* ")

    with col2:
        st.text("")
        st.text("")
        arxiv.search_and_update(                       # Maintenance mode
            db_name="./database/arxiv_data.db",
            query="all:quantum",
            max_results=250,
            total_results_limit=1000
        )

    st.markdown('---')

    # -- PAGE RENDERING --
    if page == "#0 Introduction_":
        page_0()
    elif page == "#1 Something_":
        page_1()
    elif page == "#2 Something_":
        page_2()
    elif page == "#3 Something_":
        page_3()
    elif page == "#4 Something_":
        page_4()
    elif page == "#5 Something_":
        page_5()
    elif page == "#6 Something_":
        page_6()

    st.sidebar.markdown("&nbsp;")

    gc.collect()
