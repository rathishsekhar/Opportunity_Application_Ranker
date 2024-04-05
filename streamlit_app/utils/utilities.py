# streamlit_app/Utilities/utilities.py

# Importing libraries
import numpy as np

# Function for encapsulating JSON like object columns

def json_column_encapsulator(key_column, values_list):

    if not key_column:
         return RuntimeError
    
    if len(key_column) != len(values_list):
        return RuntimeError 
    
    if all(len(lst) == len(values_list[0]) for lst in values_list):
            transposed_values = zip(*values_list)
            results = [{k:v for k,v in zip(key_column, sublist)} for sublist in transposed_values]
            return np.array(results)

def find_n_topmatches(a, dict_data, n):
    temp = {}
    for key_, val_ in dict_data.items():

        temp_value = np.dot(val_/np.linalg.norm(val_), a/np.linalg.norm(a))
        temp[key_] = temp_value
    
    sorteddict = (sorted(temp.items(), key = lambda x: x[1], reverse = True)[:n])

    return dict(sorteddict)