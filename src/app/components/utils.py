import json
import os
import sqlite3

import boto3
import pandas as pd
import streamlit as st
from botocore.exceptions import ClientError
from environs import Env, ErrorMapping


@st.cache_data
def load_data(db_path="./database/arxiv_data.db"):
    """Load data from the SQLite database."""
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM arxiv_entries"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_secret():
    """Retrieve the secret API key from AWS Secrets Manager."""
    secret_name = "HG_API_KEY_PRO_2"
    region_name = "eu-west-3"
    session = boto3.session.Session() # Create a Secrets Manager client
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )
    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e: # For a list of exceptions thrown, see https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e
    secret_dict = json.loads(get_secret_value_response['SecretString'])
    return secret_dict['HG_API_KEY_PRO_2']

def initialize_hg_api_key():
    """Initialize and return the Hugging Face API key."""
    env = Env()
    env.read_env('src/.env')
    hg_api_key = os.getenv("HG_API_KEY")  # Dev mode

    if hg_api_key is None:
        hg_api_key = get_secret()   # Prod mode AWS

    if hg_api_key is None:
        st.error("Hugging Face API key not found")
        raise ErrorMapping("Hugging Face API key not found.")

    return hg_api_key
