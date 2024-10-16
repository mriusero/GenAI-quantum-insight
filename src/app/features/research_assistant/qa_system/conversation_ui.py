import json
import os
from datetime import datetime
from typing import Any

import streamlit as st


def initialize_session_state():
    """Iniyialize the session variables if they are not already defined."""
    if "messages" not in st.session_state:
        st.session_state.messages = st.session_state.get("conversation_history", []) if "conversation_history" in st.session_state else []
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "user_expertise" not in st.session_state:
        st.session_state.user_expertise = "Débutant"
    if "input_text" not in st.session_state:
        st.session_state.input_text = ""
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.0
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = 320
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0


def add_empty_lines(n: int):
    """Add empty lines to space out the UI components."""
    for _ in range(n):
        st.text("")

def display_columns():
    """Display the columns for selecting expertise, temperature, etc."""

    col1, col2, col3, col4, col5, col6 = st.columns([1, 2, 2, 2, 2, 2])
    #st.write("___")
    with col1:
        st.image('src/app/layout/images/llama-icon.png', caption='Llama-2-7b-chat-hf', width=100)
    with col2:
        add_empty_lines(2)
        st.session_state.user_expertise = st.selectbox("Expertise level",
                                                       ["Beginner", "Intermediate", "Advanced", "Expert"])
    with col3:
        add_empty_lines(2)
        st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, 0.1, step=0.1)
    with col4:
        add_empty_lines(2)
        st.session_state.max_tokens = st.slider("Max Tokens", 50, 500, 320)

    with col5:
        add_empty_lines(2)
        if "tokens_count" not in st.session_state:
            st.session_state.tokens_count = st.empty()

        if "tokens_bar" not in st.session_state:
            st.session_state.tokens_bar = st.progress(0)
    with col6:
        add_empty_lines(3)
        if st.button("Reset conversation"):
            st.session_state.messages = []
            st.session_state.conversation_history = []
            st.session_state.user_expertise = "Débutant"
            st.session_state.input_text = ""
            st.session_state.temperature = 0.0
            st.session_state.max_tokens = 320
            st.session_state.total_tokens = 0
            st.toast("Conversation reinitialised !")
            initialize_session_state()

def display_chat_interface(chat_container):
    """Display the entire conversation between the user and the system."""
    with chat_container:
        st.empty()  # Réinitialise l'affichage
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for message in st.session_state.messages:
            bubble_class = "user-bubble" if message["role"] == "user" else "system-bubble"
            st.markdown(f'<div class="{bubble_class}">{message["content"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


def save_conversation():
    """Save the conversation to a JSON file with a timestamp."""
    SAVE_DIR = "database/.conversations/"

    if st.session_state.messages:
        col1, col2, col3 = st.columns([2, 1, 3])
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        conversation_data = {
            "timestamp": timestamp,
            "expertise_level": st.session_state.user_expertise,
            "messages": st.session_state.messages
        }
        json_data = json.dumps(conversation_data, indent=4)

        with col1:
            filename = st.text_input("Save conversation", value="Qbits-explanation...")
            filename = f"{filename}_{timestamp}"

            if st.button("Save"):
                if not os.path.exists(SAVE_DIR):
                    os.makedirs(SAVE_DIR)
                save_path = os.path.join(SAVE_DIR, f"{filename}.json")
                with open(save_path, "w") as f:
                    json.dump(conversation_data, f)
                    st.success(f"Conversation saved: {filename}.json")
        with col2:
            add_empty_lines(2)
            st.download_button(
                label="Download",
                file_name=filename,
                mime="application/json",
                data=json_data
            )



def user_interface(qasystem: Any) -> None:
    """Main function that handles the user interface."""
    initialize_session_state()
    display_columns()

    st.markdown("""
        <style>
        .user-bubble {background-color: #ABECAB; color: black; padding: 10px; border-radius: 10px; margin: 10px 0; text-align: right;}
        .system-bubble {background-color: #444D56; color: white; padding: 10px; border-radius: 10px; margin: 10px 0; text-align: left;}
        .chat-container {display: flex; flex-direction: column-reverse; max-height: 500px; overflow-y: auto;}
        h1 {color: #FF5733; }
        </style>
    """, unsafe_allow_html=True)

    chat_container = st.container()
    user_input = st.text_area("Query:", value=st.session_state.input_text)

    if st.button("Submit") and user_input != "":
        st.session_state.input_text = ""

        with st.spinner("Llama is thinking..."):

                first_question = user_input
                llm_response = qasystem.ask_question(
                    usr_question=first_question,
                    usr_level=st.session_state.user_expertise,
                    temperature=st.session_state.temperature,
                    max_tokens=st.session_state.max_tokens
                )
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.session_state.messages.append({"role": "system", "content": llm_response})

        display_chat_interface(chat_container)

    st.write("___")
    save_conversation()