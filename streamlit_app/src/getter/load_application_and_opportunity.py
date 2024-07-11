# src/getter/load_application_and_opportunity.py

# Importing libraries
import pandas as pd
import pickle
import os 
from pathlib import Path
import streamlit as st

def get_data(file_name):
    """
    Get data from the location listed in dataloc
    """
    dataloc = os.path.join(Path(os.path.realpath("")).resolve(), 'data', 'input')
    
    with open(os.path.join(dataloc, str(file_name) + ".pkl"), 'rb') as f:
        data = pickle.load(f)
        return data