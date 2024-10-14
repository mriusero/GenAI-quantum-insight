import streamlit as st

def user_interface(qasystem):
    """Set up UI for the Q&A system using Streamlit."""
    st.title("Système de Questions-Réponses")
    st.write("Posez votre question ci-dessous.")

    usr_question = st.text_input("Votre question:")
    if st.button("Soumettre"):
        if usr_question:
            with st.spinner('Traitement...'):
                response = qasystem.ask_question(usr_question)
                st.markdown(f"## Réponse:\n\n  {response}\n")
        else:
            st.warning("Veuillez poser une question.")

    if qasystem.conversation_memory:
        st.write("### Historique de la conversation:")
        for entry in qasystem.conversation_memory:
            st.write(f"**Vous:** {entry['question']}")
            st.write(f"**Assistant:** {entry['response']}")

    if st.button("Réinitialiser la conversation"):
        qasystem.conversation_memory = []