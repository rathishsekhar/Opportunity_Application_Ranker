# src/getter/save_application_and_opportunity.py

# Importing libraries
import pandas as pd
import pickle
import os
from pathlib import Path


def save_data(file, file_name):
    """
    Save(pickle) the file as per the filename for the data in specific location
    """
    dataloc = os.path.join(Path(os.path.realpath("")).resolve(), 'data', 'output', 'saved_appdata')

    with open(os.path.join(dataloc,  str(file_name) + ".pkl"), 'wb') as f:
        pickle.dump(file, f)
        