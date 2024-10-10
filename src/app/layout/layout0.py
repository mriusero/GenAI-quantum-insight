import streamlit as st
import os
from environs import Env

from ..components.utils import load_data
from ..features.vectors_store.document_processor import ResearchAssistant

def page_0():
    st.markdown('<div class="header">#0 Project Overview_</div>', unsafe_allow_html=True)
    st.text("")
    st.text("Here is an overview of the project")
    st.markdown("___")

    data = load_data()
    st.write(data)

    env = Env()
    env.read_env('src/.env')
    hg_api_key = os.getenv("HG_API_KEY")

    # Liste des URLs de fichiers PDF
    pdf_urls = ["http://arxiv.org/pdf/2405.20113v2", "http://arxiv.org/pdf/2410.07099v1"]

    agent = ResearchAssistant(hg_api_key)
    agent.run_assistance(pdf_urls)

