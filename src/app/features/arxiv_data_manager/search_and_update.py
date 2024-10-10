# search_and_update.py
import streamlit as st

from .arxiv_client import ArxivAPIClient
from .arxiv_db import ArxivDataLoader
from .arxiv_parser import ArxivXMLParser


def search_and_update(db_name, query="all:quantum", max_results=100, total_results_limit=100):
    """Search and update the database with Arxiv data."""
    col1, col2, col3 = st.columns([3, 6, 3])

    with col1:
        search_clicked = st.button("Search release")

    with col2:
        status_placeholder = st.empty()  # Info messages

    with col3:
        spinner_placeholder = st.empty()  # Spinner

    if search_clicked:
        client = ArxivAPIClient()
        parser = ArxivXMLParser()

        try:
            with spinner_placeholder:  # Spinner during processing
                with st.spinner(""):
                    status_placeholder.info("Fetching Arxiv data ...")
                    xml_data = client.fetch_arxiv_data(query=query, max_results=max_results,
                                                       total_results_limit=total_results_limit)

            if xml_data:
                status_placeholder.info("Parsing XML data...")
                entries = parser.parse_entries(xml_data)

                status_placeholder.info("Database update...")
                with ArxivDataLoader(db_name=db_name) as loader:
                    new_entries, updated_entries = loader.parse_and_insert(entries)

                if new_entries == 0 and updated_entries == 0:  # Status messages based on the update results
                    status_placeholder.info("Already up to date!")
                elif new_entries == 0:
                    status_placeholder.success(f"{updated_entries} update(s) found!")
                elif updated_entries == 0:
                    status_placeholder.success(f"{new_entries} new document(s) found!")
                else:
                    status_placeholder.success(f"{new_entries} new document(s) found & "
                                               f"{updated_entries} document(s) updated!")
            else:
                status_placeholder.error("API error: No data fetched")
        except Exception as e:
            status_placeholder.error(f"Error occurred: {e}")
        finally:
            spinner_placeholder.empty()