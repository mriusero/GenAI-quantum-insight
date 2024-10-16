import gc
import os

import streamlit as st
from huggingface_hub import HfApi

from .components import initialize_hg_api_key, load_data
from .features import arxiv_data_manager as arxiv
from .features import research_assistant as agent

global update_message, debug, document_processor, qa_system

def load_css():
    """Load custom CSS styles for the Streamlit app."""
    css_path = os.path.join(os.path.dirname(__file__), 'styles.css')
    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def main_layout():
    """Set up the main layout of the Streamlit application."""
    from .components import github_button
    from .layout import page_0, page_1, page_2, page_3

    st.set_page_config(
        page_title="Quantum Insights",
        layout='wide',
        initial_sidebar_state="auto",
    )
    load_css()
    st.sidebar.markdown("## *🤖 Generative AI* \n")
    page = st.sidebar.radio("Table of contents",  ["Overview_",
                                                   "Database_",
                                                   "Ask questions_",
                                                   "Latest chats_",
                                                   ])
    ## -- INIT -- ##
    if 'data' not in st.session_state:                                  # Data loading
        st.session_state['data'] = load_data()

    debug = st.sidebar.toggle("Debug mode")                             # Debug mode
    #st.toast("Debug mode is on" if debug else "Debug mode is off")
    os.system('cls' if os.name == 'nt' else 'clear')
    if debug:
        print(
"""
🔧 Debug mode is ON 
=====================
"""
        )
    else:
        print(
"""
🚫 Debug mode is OFF
=====================
"""
        )
    logo_url = "https://huggingface.co/front/assets/huggingface_logo.svg"    # Hugging Face API connexion test
    api = HfApi()
    hg_api_key = initialize_hg_api_key()
    try:
        result = api.whoami(hg_api_key)
        if result is not None:
            st.sidebar.markdown(
                f'<p style="color:silver; font-size:16px;">'
                f'<img src="{logo_url}" width="30" style="vertical-align:middle; margin-right:12px;"/>'
                f'API key found successfully !</p>',
                unsafe_allow_html=True
            )
        else:
            st.sidebar.markdown(
                f'<p style="color:red; font-size:16px;">'
                f'<img src="{logo_url}" width="30" style="vertical-align:middle; margin-right:10px;"/>'
                f'Invalid API key ❌</p>',
                unsafe_allow_html=True
            )
    except Exception as e:
        st.sidebar.markdown(
            f'<p style="color:red; font-size:16px;">'
            f'<img src="{logo_url}" width="30" style="vertical-align:middle; margin-right:10px;"/>'
            f'Error: {e} ❌</p>',
            unsafe_allow_html=True
        )
    document_processor, qa_system = agent.models_loading(hg_api_key, debug)    # Load models

    print("\n")
    # -- LAYOUT -- ##
    global update_message
    st.markdown('<div class="title"> Quantum Insights ⚡️</div>', unsafe_allow_html=True)    # Project title
    colA, colB= st.columns([8, 4])
    with colA:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.text("")                                                               # Project description
            st.markdown("#### *An AI agent specialized in quantum physics research, helping you to understand the latest sciences discoveries.*")
    with colB:
        col1, col2 = st.columns([2,1])
        with col2:
            github_button('https://github.com/mriusero/GenAI-quantum-insight')  # GitHub button
            st.write("***See on Github***")
    st.markdown('---')

    # -- PAGE RENDERING --
    if page == "Overview_":
        page_0()
    elif page == "Database_":
        page_1(debug, arxiv, document_processor)
    elif page == "Ask questions_":
        page_2(debug, agent, qa_system)
    elif page == "Latest chats_":
        page_3(debug, qa_system)

    st.sidebar.markdown("&nbsp;")
    gc.collect()

