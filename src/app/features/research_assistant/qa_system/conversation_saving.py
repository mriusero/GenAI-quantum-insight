import json
import os

import streamlit as st

SAVE_DIR = "database/.conversations/"

def display_conversations():

    conv_files = [f for f in os.listdir(SAVE_DIR) if f.endswith('.json')]
    selected_conv = st.selectbox("Currently display_", conv_files)

    if selected_conv:
        conv_path = os.path.join(SAVE_DIR, selected_conv)
        with open(conv_path, "r") as f:
            conversation = json.load(f)

        st.markdown("""
            <style>
            .user-bubble {background-color: #ABECAB; color: black; padding: 10px; border-radius: 10px; margin: 10px 0; text-align: right;}
            .system-bubble {background-color: #444D56; color: white; padding: 10px; border-radius: 10px; margin: 10px 0; text-align: left;}
            .chat-container {display: flex; flex-direction: column-reverse; max-height: 500px; overflow-y: auto;}
            </style>
        """, unsafe_allow_html=True)

        messages = conversation.get("messages", [])

        with st.container():
            for message in messages:
                if isinstance(message, dict):
                    if message.get("role") == "user":
                        st.markdown(f'<div class="user-bubble">{message.get("content", "")}</div>',
                                    unsafe_allow_html=True)
                    elif message.get("role") == "system":
                        st.markdown(f'<div class="system-bubble">{message.get("content", "")}</div>',
                                    unsafe_allow_html=True)
                else:
                    st.error("Error: Invalid message format")
