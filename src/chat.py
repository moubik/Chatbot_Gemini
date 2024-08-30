from src.retriever import get_response
import streamlit as st
from src.utils import add_fallback_url, remove_unwanted_phrases 
import os
from langchain_community.utilities import SQLDatabase
import time
from google.api_core.exceptions import ResourceExhausted


import time


def map_terms_to_tables(user_query, term_to_table_mapping):
    mapped_query = user_query
    for term, table in term_to_table_mapping.items():
        mapped_query = mapped_query.replace(term, table)
    return mapped_query

def log_query(user_query, mapped_query):
    """Log the original and mapped queries for debugging."""
    print(f"Original query: {user_query}")
    print(f"Mapped query: {mapped_query}")

def get_best_response(user_query):
    try:
        return get_response(user_query)
    except ResourceExhausted:
        time.sleep(60)  # Attendre 60 secondes avant de réessayer
        return "Le temps limite a été depassé. Veuillez recommencer."