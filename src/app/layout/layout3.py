import streamlit as st

from ..features.research_assistant.qa_system.conversation_saving import display_conversations

def page_3(debug, qa_system):
    st.markdown('<div class="header">#3 Something_</div>', unsafe_allow_html=True)
    st.text("")
    st.text("Here is the page 3")
    st.markdown('---')

    display_conversations()
