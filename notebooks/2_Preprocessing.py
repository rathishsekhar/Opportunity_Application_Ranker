#!/usr/bin/env python
# coding: utf-8

# # 2. Preprocessing

# In[1]:


#Importing libraries:

import os
import pickle
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import pyarrow as pa


# ### 2.1.1 Gathering data

# In[2]:


# Opening the data 

dataloc = '/Users/rathish/Documents/Project/data'

file_list = os.listdir(dataloc+"/raw_data")
d = []

for i in range(1,7):
    data = pd.read_parquet(dataloc+'/raw_data/'+'split_' + 
                           str(i)+'.parquet', engine= 'pyarrow')
    d.append(data)

pdata = pd.concat(d, ignore_index=True)


# In[3]:


# Reading a few lines of data

pdata.head(5)


# ### 2.1.2 Defining column names so that various preprocessing functions could be applied

# In[12]:


# Defining liststhat contains the names of the columns for easy access

job_column = [
    'ExternalBriefDescription',
    'ExternalDescription', 
    'Title', 
    'JobCategoryName'
]
uid_column = ['OpportunityId', 'ApplicationId']

# column - StepId has been removed on purpose, will be added later

can_column = [
    'IsCandidateInternal',
    'BehaviorCriteria', 
    'MotivationCriteria',
    'EducationCriteria', 
    'LicenseAndCertificationCriteria', 
    'SkillCriteria', 
    'WorkExperiences', 
    'Educations', 
    'LicenseAndCertifications', 
    'Skills', 'Motivations', 
    'Behaviors', 
    'StepName', 
    'Tag', 
    'StepGroup',
    'pass_first_step'
]
 
sel_column = ['IsRejected']

# Defining list of columns based on the type of contents

str_column = [
    'ExternalBriefDescription',
    'ExternalDescription', 
    'Title', 
    'JobCategoryName',
    'BehaviorCriteria', 
    'MotivationCriteria',
    'EducationCriteria', 
    'LicenseAndCertificationCriteria', 
    'SkillCriteria', 
    'WorkExperiences', 
    'Educations', 
    'LicenseAndCertifications', 
    'Skills', 
    'Motivations', 
    'Behaviors',
    'StepId', 
    'StepName', 
    'StepGroup'
]

bool_column = ['IsCandidateInternal', 'pass_first_step']


# ## 2.2 Preprocessing the data - TF-IDF weighted Word 2 Vec model, BERT and Universal Sentence Encoder

# ### 2.2.1 Creating functions that perform operations on text for different models

# In[5]:


# Defining functions for extracting information from the JSON like objects 
# Preprocessing the extracted information

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
    '''
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
    '''
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

def preprocessing_bert(ls):
    '''
    Preprocesses the text by doing the following: 
    1. Removes HTML tags 
    2. Converts text to lower case 
  
    Args:
    ls (text): Input text for preprocessing
    
    Returns: 
    text (str): Preprocessed text for further use
    '''
    # Taking care of values other than string that may come across

    text = str(ls)
    
    #removing html_text

    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[\n | \t]', " ", text)
    text = re.sub(r"  ", " ", text)

    #Lowering the case of the tokens 

    text = text.lower()

    return text


def preprocessing_use(ls):
    '''
    Preprocesses the text by doing the following: 
    1. Removes HTML tags 
    2. Removes markup tags eg: "\n"  
  
    Args:
    ls (text): Input text for preprocessing
    
    Returns: 
    text (str): Preprocessed text for further use
    '''
    # Taking care of values other than string that may come across

    text = str(ls)
    
    #removing html_text

    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[\n | \t]', " ", text)
    text = re.sub(r"  ", " ", text)

    return text


# ### 2.2.2 Applying the preprocessing functions to the datasets

# In[6]:


# Applying dataextractor function to data from defined columns

for colnames in str_column:
    dataextractor(pdata, colnames)


# In[7]:


# Applying preprocessing function to data from defined columns

for x in str_column:
    pdata[x+"__w2vpp"] = pdata[x+"__pp"].apply(preprocessing_w2v)

for x in str_column:
    pdata[x+"__bertpp"] = pdata[x+"__pp"].apply(preprocessing_bert)

for x in str_column:
    pdata[x+"__usepp"] = pdata[x+"__pp"].apply(preprocessing_use)


# Imputing NaN values in column - 'Tag'

# In[8]:


# Imputing NaN values with -1

pdata['Tag'].fillna(-1, inplace = True)


# In[9]:


pdata['Tag']


# ###  Exporting data for featurization

# In[10]:


# Exporting rdata to a pickle file

pdata.to_pickle(dataloc + "/cleaned_data/preprocesseddata.pkl")


# In[11]:


pdata.columns

