import streamlit as st

def page_2(debug, agent, qa_system):
    st.markdown('<div class="header">Ask questions_</div>', unsafe_allow_html=True)
    st.text("")

    agent.run_assistance(debug=debug, qa_system=qa_system)
