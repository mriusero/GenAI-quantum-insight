import streamlit as st
from typing import List, Dict, Any

def user_interface(qasystem: Any) -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("user_expertise", "Débutant")
    st.session_state.setdefault("input_text", "")
    st.session_state.setdefault("loading", False)

    col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 2])
    st.write("___")
    with col1:
        st.image('src/app/layout/images/llama-icon.png', caption='Llama-2-7b-chat-hf', width=100)
    with col2:
        st.text("")
        st.text("")
        st.session_state.user_expertise = st.selectbox("Niveau d'expertise",
                                                       ["Beginner", "Intermediate", "Advanced", "Expert"])
    with col3:
        st.text("")
        st.text("")
        st.session_state.temperature = st.slider("Température", 0.0, 1.0, 0.1, step=0.1)
    with col4:
        st.text("")
        st.text("")
        st.session_state.max_tokens = st.slider("Max Tokens", 50, 500, 255)
    with col5:
        st.text("")
        st.text("")
        st.text("")
        if st.button("Reset conversation"):
            st.session_state.messages = []
            st.session_state.loading = False

    st.markdown("""
        <style>
        .user-bubble {background-color: #DCF8C6; color: black; padding: 10px; border-radius: 10px; margin: 10px 0; text-align: right;}
        .system-bubble {background-color: #F1F0F0; color: black; padding: 10px; border-radius: 10px; margin: 10px 0; text-align: left;}
        .chat-container {display: flex; flex-direction: column-reverse; max-height: 500px; overflow-y: auto;}
        </style>
    """, unsafe_allow_html=True)

    chat_container = st.container()

    def render_chat(chat_container) -> None:
        chat_container.empty()
        with chat_container:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            for message in st.session_state.messages:
                bubble_class = "user-bubble" if message["role"] == "user" else "system-bubble"
                st.markdown(f'<div class="{bubble_class}">{message["content"]}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    render_chat(chat_container)

    user_input = st.text_input("Votre question :", value=st.session_state.input_text)

    if st.session_state.loading:
        with st.spinner("Le modèle réfléchit..."):
            render_chat(chat_container)

    if st.button("Submit") and user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.input_text = ""
        st.session_state.loading = True

        llm_response = qasystem.ask_question(
            usr_question=user_input,
            usr_level=st.session_state.user_expertise,
            temperature=st.session_state.temperature,
            max_tokens=st.session_state.max_tokens
        )
        print(llm_response)
        st.session_state.messages.append({"role": "system", "content": llm_response})
        st.session_state.loading = False

        render_chat(chat_container)

    st.write("___")
