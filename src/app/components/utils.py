import streamlit as st
import pandas as pd
import sqlite3


@st.cache_data
def load_data(db_path="./database/arxiv_data.db"):
    """Load data from the SQLite database."""
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM arxiv_entries"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df