import streamlit as st

from ..features.research_assistant.qa_system.conversation_saving import display_conversations

def page_3(debug, qa_system):
    st.markdown('<div class="header">Lastest chats_</div>', unsafe_allow_html=True)
    st.text("")
    #st.write("#### Choose a chat to display:")

    display_conversations()
