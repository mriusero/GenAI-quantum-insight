import streamlit as st

from ..components.utils import initialize_hg_api_key
from ..features.vectors_store.document_processor import ResearchAssistant

def page_1():
    st.markdown('<div class="header">#1 Something_</div>', unsafe_allow_html=True)
    st.text("")
    st.text("Here is the page 2")
    st.markdown("---")

    hg_api_key = initialize_hg_api_key()
    pdf_urls = ["http://arxiv.org/pdf/2405.20113v2", "http://arxiv.org/pdf/2410.07099v1"]

    agent = ResearchAssistant(hg_api_key)
    agent.run_assistance(pdf_urls)