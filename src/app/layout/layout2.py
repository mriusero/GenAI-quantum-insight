import streamlit as st



def page_2(debug, agent, qa_system):
    st.markdown('<div class="header">#2 Something_</div>', unsafe_allow_html=True)
    st.text("")
    st.text("Here is the page 2")
    st.markdown('---')

    agent.run_assistance(debug=debug, qa_system=qa_system)
