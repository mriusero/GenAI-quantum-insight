import sqlite3

import pandas as pd
import streamlit as st


@st.cache_data
def load_data(db_path="./database/arxiv_data.db"):
    """Load data from the SQLite database."""
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM arxiv_entries"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df