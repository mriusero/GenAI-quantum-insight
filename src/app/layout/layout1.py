import streamlit as st
import json
from src.app.features.research_assistant import store_management

def page_1(debug, arxiv, document_processor):
    st.markdown('<div class="header">Database_</div>', unsafe_allow_html=True)
    st.text("")

    col1, col2 = st.columns([4,3])

    with col1:
        st.write("#### Click on_")
        st.write("`Search release` pour rechercher des nouveaux papiers de recherches.")
        st.write("`Vectorize document` pour intégrer les nouveaux papiers dans le vector store.")

    with col2:
        st.text("")

    st.write("___")
    store_management(debug, arxiv, document_processor)

    st.write("---")

    st.write("## ArXiv papers database_")
    st.write("Cette table contient les métadonnées des papiers de recherches extraits de le base de données ArXiv.")
    st.write("")
    st.dataframe(st.session_state['data'])
