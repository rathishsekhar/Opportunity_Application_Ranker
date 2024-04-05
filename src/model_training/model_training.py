# src/module_training/module_training

# Importing libraries 

import os
from pathlib import Path
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm

# Generating function that creates arrays for calculating the cosine similarity 
def reduce_dimensionality(data, opp_uid_name, app_uid_name,  opportunity_stack_dict, candidate_stack_dict):
     """
    Reduce the dimensionality of vectors using PCA, if horizontally stacked 
    (ignored if vertically stacked) and create arrays from the data used 
    further for similarity calculations
    
    Parameters:
        candidate_vectors (pandas.DataFrame): Horizontally stacked Word2Vec 
        vectors for candidates
        opportunity_vectors (pandas.DataFrame): Horizontally stacked Word2Vec 
        vectors for opportunities
        
    Returns:
        numpy.ndarray: Transformed array of vectors with reduced dimensionality
        if horizontally stacked else transformed array of vectors with same 
        dimensionality
    """
     data__ = data[[opp_uid_name] + [app_uid_name]]
     data__['opportunity_vectors'] = data__[opp_uid_name].apply(lambda x : opportunity_stack_dict[x])
     data__['candidate_vectors'] = data__[app_uid_name].apply(lambda x : candidate_stack_dict[x])

     opportunity__ = np.array(
          [np.array(x) for x in data__['opportunity_vectors'].tolist()]
        )
    
     candidate__ = np.array(
          [np.array(x) for x in data__['candidate_vectors'].tolist()]
        )

    # Setting the number of dimensions as minimum shape
     no_of_dimensions = min(candidate__.shape[1], opportunity__.shape[1])

     if 'PCA' in dir():
        pca = PCA(n_components = no_of_dimensions, copy = False)
     else:
         from sklearn.decomposition import PCA 
         pca =  PCA(n_components = no_of_dimensions, copy = False)

     # Applying PCA 
     if candidate__.shape[1] >= opportunity__.shape[1]: 
         app_array, opp_array = pca.fit_transform(candidate__), opportunity__
     else:
         app_array, opp_array = candidate__, pca.fit_transform(opportunity__) 
    
     app_pca_dict, opp_pca_dict = {}, {}
     
     for uid, vector in zip(data__[app_uid_name], app_array):
         app_pca_dict[uid] = vector
     
     for uid, vector in zip(data__[opp_uid_name], opp_array):
         opp_pca_dict[uid] = vector
    
     return opp_pca_dict, app_pca_dict


# Getting top n-similar cosine vectors
def pairwise_cosine(data, opp_uid_name, app_uid_name, opp_pca_dict, app_pca_dict):
    '''
    Compute pairwise cosine similarity between opportunity and application 
    vectors in DataFrame

    Parameters:
        data(pandas.DataFrame): Input DataFrame containg opportunity and 
        application UIDs. 
        opp_upd_name(str): Columns name with oppotunity IDs
        app_uid_name(str): Columns name with applciation IDs
        opp_pca_dict (dict): Dictionary mapping opportunity uids to the vectors
        app_pca_dict (dict): Dictionary mapping application uids to the vectors
    
    Returns:
        pandas.DataFrame with columns "OpportunityId", "ApplciationId and 
        cosine similarity
    '''
    data__ = data[[opp_uid_name] + [app_uid_name]]
    data__['opportunity_vectors'] = data__[opp_uid_name].apply(lambda x : opp_pca_dict[x])
    data__['candidate_vectors'] = data__[app_uid_name].apply(lambda x : app_pca_dict[x])

    opportunity__ = np.array(
          [np.array(x) for x in data__['opportunity_vectors'].tolist()]
        )
    
    candidate__ = np.array(
          [np.array(x) for x in data__['candidate_vectors'].tolist()]
        )
    
    opportunity__ = (opportunity__/np.linalg.norm(opportunity__, axis = 1)[:, np.newaxis])
    candidate__ = (candidate__/np.linalg.norm(candidate__, axis = 1)[:, np.newaxis])
    
    data__['row_similarity'] = [np.dot(row1, row2) for row1, row2 in zip(opportunity__, candidate__)]
     
    return data__[[opp_uid_name] + [app_uid_name] + ['row_similarity']]

def topn_similar(opp_pca_dict, app_pca_dict, n = 3):
    """
    Calculates top n most similar application IDs for a given opportunity ID

    Parameters:
        opp_pca_dict (dict): Dictionary mapping opportunity IDs to their 
        dimensionally reduced vectors. 
        app_pca_dict (dict): Dictionary mapping application IDs to their 
        dimensionally reduced vectors. 
        n(int, optional - default = 3): Number of top similar application IDs 
        to retrieve
    
    Returns:
    dict: A dictionary with keys as opportunity IDs pointing to value which is 
    a dictionary in itself. This dictionary contaings application IDs as keys 
    and similarity scores as values
    """
    
    similarity_dict = {}

    for key_opp, val_opp in tqdm(opp_pca_dict.items(), desc = "Application IDs: ", total = len(opp_pca_dict)):
        
        temp = {}

        for key_app, val_app in app_pca_dict.items():
            temp_value = np.dot(val_opp/np.linalg.norm(val_opp), val_app/np.linalg.norm(val_app))
            temp[key_app] = temp_value
        
        sorteddict = sorted(temp.items(), key = lambda x: x[1], reverse = True)[:n]

        similarity_dict[key_opp] = sorteddict
    
    return similarity_dict

