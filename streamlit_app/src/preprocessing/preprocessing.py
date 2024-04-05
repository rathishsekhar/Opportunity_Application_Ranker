# src/data_processing/preprocessing.py

# Importing libraries

import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re


def dataextractor(data, col_names):
    '''
    Extracts data from the JSON array like objects in the columns and 
    places string in a new column with name col_names__pp.

    Args: 
    data (pandas.DataFrame): Dataset containing the JSON array like objects
    col_names (str): Target column with JSON array like objects in each row 
    on whichthe operation needs to be performed

    Returns: None
    Creates pandas.DataFrame: Column with the name format 'columname'__'pp' 
    with string of all values extracted from JSON array like objects 
    '''
    def valuesextractor(cell):
        if isinstance(cell, np.ndarray):        
            lst = []
            for dctnry in cell:
                if not dctnry:
                    return str("")
                
                else:
                    for k, v in dctnry.items():
                        lst.append(k)
                        lst.append(v)
                    lst = [str(x) for x in lst]
                    return " ".join(lst)
        else:
            return cell

    #Adding '__' for unique identification of data extracted columns

    data[col_names + '__' + 'pp'] = data[col_names].apply(valuesextractor) 
    
    return 


def preprocessing_w2v(ls, stemming = True):
    """
    Preprocesses the text by doing the following: 
    1. Removes HTML and markup signs
    2. Converts text to lower case 
    3. Decontracts words for example : "won't" becomes "will not" etc
    4. Tokenizes the words 
    5. Removes stop words 
    6. Applies Porters stemmer if stemming = True
    
    Args:
    ls (text): Input text for preprocessing
    stemming (bool): Flag to enable stemming (default is False)
    input_islist (bool): Flag to enable preprocessing of text inside list. 
    If False, the input is treated as string

    Returns: 
    text (str): Preprocessed text for further use
    """

    # Taking care of values other than string that may come across
    text = str(ls)
    
    #removing html_text
    text = re.sub(r'<.*?>', '', text)

    #Lowering the case of the tokens 
    text = text.lower()

# Adding a few more functions for text processing

    #Code obtained from # https://stackoverflow.com/a/47091490/4084039
    #Removing decontracted words from text

    def decontracted(phrase):
        phrase = re.sub(r"won't", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)

        # general contractions
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)
        return phrase
    
    #Replacing decontracted words

    text = decontracted(text)

    #Removing special characters 

    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    
    #Removing stopwords

    tkns = word_tokenize(text)
    stpwrds = set(stopwords.words('english'))
    tkns = [e for e in tkns if e not in stpwrds]

    #Applying Stemmer 

    if stemming: 
        stemmer = PorterStemmer()
        sentence =  ' '.join([stemmer.stem(words) for words in tkns])
        
    return sentence

def preprocessing_transformermodels(ls):
    """
    Preprocesses the text by doing the following: 
    1. Removes HTML tags 
    2. Converts text to lower case 
  
    Args:
    ls (text): Input text for preprocessing
    
    Returns: 
    text (str): Preprocessed text for further use
    """

    # Taking care of values other than string that may come across

    text = str(ls)
    
    #removing html_text

    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[\n | \t]', " ", text)
    text = re.sub(r"  ", " ", text)

    return text