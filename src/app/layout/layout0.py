import streamlit as st


def page_0():
    st.markdown('<div class="header">Project Overview_</div>', unsafe_allow_html=True)
    text ="""
Dans les domaines en constante évolution tel que la technologie, il est essentiel de comprendre quels sont les principaux enjeux scientifiques du moment. Ce projet vise à explorer les larges langages models disponibles en open-source via Hugging Face pour ***synthétiser, résumer, traduire et vulgariser des travaux de recherches***, via la conception d'un modèle RAG augmented spécialisé dans les domaines de la recherche sur l'informatique quantique.

---

## Objectifs_

1. Synthèse et Résumé
   - Résumer des papiers de recherche issues de la base de données ArXiv pour en extraire des informations. 
   - Produire des résumés adaptés à différents niveaux d'expertise, de scientifique à débutant.

2. Explication des Concepts
   - Expliquer des concepts complexes de manière simple et compréhensible.
   - Fournir des exemples d'applications industrielles pour illustrer la pertinence des recherches.

3. Extraction de Code
   - Identifier et extraire les portions de code présents dans les papiers de recherche.
   - Présenter ce code dans un format clair et expliqué. 

4. Traduction
    - Traduire les papiers de recherche en plusieurs langues pour une diffusion internationale.
    
5. Vulgarisation
    - Rendre les informations sur l'informatique quantique et la blockchain accessibles à un large public.
    - Démocratiser l'accès à ces connaissances pour stimuler l'innovation dans ces domaines.
    
---

## Méthodologie_

**RAG (Retrieval-Augmented Generation)**
   - Utilisation d’une base de données contenant plus de 10 000 URL vers des fichiers PDF de papiers de recherche.
   - Intégration de techniques de recherche pour optimiser l'extraction et la génération de contenu.

---

## Public cible_

**Utilisateurs de tout niveau d'expertise**
   - **Scientifiques** : Accès à des synthèses techniques approfondies.
   - **Professionnels du secteur** : Exemples d'applications industrielles.
   - **Débutants** : Explications simples et accessibles pour comprendre les concepts clés.

---

## Conclusion_

Ce projet a pour ambition de créer un pont entre la recherche avancée et le grand public, rendant les informations sur l'informatique quantique et la blockchain accessibles à tous. Grâce à un modèle IA bien réglé, nous espérons démocratiser l'accès à ces connaissances et stimuler l'innovation dans ces domaines passionnants.

---

    """
    st.markdown(text)





