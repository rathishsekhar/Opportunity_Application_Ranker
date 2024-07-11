# src/getter/load_application_and_opportunity.py

# Importing libraries
import pandas as pd
import pickle
import matplotlib.pyplot as pt
import os
from pathlib import Path


def save_interim_data(file, file_name):
    """
    Save(pickle) the file as per the filename for the data in specific location
    """
    dataloc = os.path.join(Path(os.path.realpath("")).resolve(), 'inputs', 'data')
    #"/Users/rathish/Documents/Projects/Opportunity_Application_Ranker/inputs/data"

    with open(dataloc + "/interim/" + str(file_name) + ".pkl", 'wb') as f:
        pickle.dump(file, f)


def save_processed_data(file, file_name):
    """
    Save(pickle) the file as per the filename for the data in specific location
    """

    dataloc = os.path.join(Path(os.path.realpath("")).resolve(), 'outputs', 'processed_data')
    
    with open(dataloc + "/processed/" + str(file_name) + ".pkl", 'wb') as f:
              pickle.dump(file, f)

def save_app_data(file, file_name):
    
    """
    Save(pickle) the file as per the filename for the data in specific location
    """
    
    dataloc = os.path.join(Path(os.path.realpath("")).resolve(), 'streamlit_app', 'data', 'input')

    with open(dataloc + str(file_name) + ".pkl", 'wb') as f:
         pickle.dump(file, f)

def save_plot(filename, foldername):
     '''
     Save the generated plot in the corresponding folder with the designated name
     '''

     dataloc =  os.path.join(Path(os.path.realpath("")).resolve(), 'outputs')

     pt.savefig(dataloc + "/visualizations/" + foldername + "/" + filename)