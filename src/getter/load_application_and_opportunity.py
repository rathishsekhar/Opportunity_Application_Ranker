# src/getter/load_application_and_opportunity.py

# Importing libraries
import pandas as pd
import pickle
import os 
from pathlib import Path

def get_raw_data():
    """
    Get data from the location listed in dataloc
    """
    data_loc = os.path.join(Path(os.path.realpath("")).resolve(), 'inputs', 'data') 

    d = []
    for i in range(1, 7):
        data = pd.read_parquet(
            data_loc + "/raw/" + "split_" + str(i) + ".parquet", engine="pyarrow",
        )
        d.append(data)

    return pd.concat(d, ignore_index=True)


def get_interim_data(file_name):
    """
    Get data from the location listed in dataloc
    """
    dataloc = os.path.join(Path(os.path.realpath("")).resolve(), 'inputs', 'data')

    with open(dataloc + "/interim/" + str(file_name) + ".pkl", 'rb') as f:
        data = pickle.load(f) 
    
    # if isinstance(data, dict):
    #   return pd.DataFrame(data) flat
    
    # elif isinstance(data, pd.DataFrame):
    return data
    
    # else:
    # raise TypeError("Input must be a dictionary or pandas DataFrame obj")


def get_processed_data(file_name):
    """
    Fetches the file_name from the location listed in dataloc
    """

    dataloc = os.path.join(Path(os.path.realpath("")).resolve(), 'outputs', 'processed_data')

    with open(dataloc + '/' + str(file_name) + ".pkl", 'rb') as f:
        data = pickle.load(f)
    
    return data