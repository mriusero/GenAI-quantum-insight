import gc
import os

import streamlit as st

#from ..visualization import DataVisualizer

update_message = 'Data loaded'
display = ""

def load_css():
    css_path = os.path.join(os.path.dirname(__file__), 'styles.css')
    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def main_layout():
    from .components import github_button
    from .layout import page_0, page_1, page_2, page_3, page_4, page_5, page_6

    st.set_page_config(
        page_title="Quantum Insights Agent",
        page_icon="/components/page_icon.png",
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
            # st.text("")
            github_button('https://github.com/mriusero/GenAI-quantum-insight')
        with colB:
            st.text("")
            st.markdown("#### *Quantum Computing Vulgarisation Agent* ")



        #with colC:
        #    # st.text("")
        #    st.text("")
        #    st.link_button('Link 2',
        #                   'https://www.something.com')


    with col2:
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        #st.markdown("##### Data Loading")
        #data = DataVisualizer()
        #st.session_state.data = data

        #st.markdown("##### Preprocessing")
        #preprocessor = Preprocessor()
        #st.session_state.processed_data = preprocessor.preprocess_data()



    st.markdown('---')

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