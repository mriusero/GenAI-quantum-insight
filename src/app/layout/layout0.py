import streamlit as st
import os

def page_0():
    st.markdown('<div class="header">#0 Project Overview_</div>', unsafe_allow_html=True)
    st.text("")
    st.text("Here is an overview of the project")
    st.markdown("---")

    # Récupérer la clé API depuis la variable d'environnement
    openai_api_key = os.getenv("API_KEY")

    if openai_api_key:
        st.success(f"Utilisation de la clé API: 'openai_api_key'")
    else:
        st.error("Pas de clé API fournie.")
