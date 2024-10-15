import base64
import streamlit as st

def page_2(debug, agent, qa_system):
    st.markdown('<div class="header">Ask a question_</div>', unsafe_allow_html=True)
    st.text("")

    text = """
    Pour une réponse personnalisé, ajuster les paramètres du modèle **Llama-2-7b-chat-hf**
    
    * **Niveau d'expertise** : sélectionnez un niveau d'expertise. `(Beginner, Intermediate, Advanced, Expert)`
    * **Température** : ajustez la créativité des réponses. `0.0 - réponses déterministes et conservatrices ; 1.0 - réponses plus variées et créatives`
    * **Max Tokens** : limitez le nombre de tokens générés par le model.
    * **Réinitialiser** : commencer une nouvelle conversation.
    """
    st.write(text)
    st.write("___")

    agent.run_assistance(debug=debug, qa_system=qa_system)


