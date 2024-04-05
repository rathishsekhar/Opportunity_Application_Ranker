# src/featurization/featurization.py

import numpy as np 
import pandas as pd 
import gensim
import nltk
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import torch
from tqdm import tqdm

def w2vbased_embedder(data, uid_column_name, str_column, bool_column, float_column):
    """
    Embeds TF-IDF weighted Word2Vec for string columns and encodes and pads the
    boolean float columns to finally concatenate into horizontally and 
    vertically stacked vectors.

    Args:
        data (pandas.DataFrame): Input dataset.
        uid_column_name (str): Name of the user ID column.
        str_column (list): List of string column names.
        bool_column (list): List of boolean column names.
        float_column (list): List of float column names.
        vector_dim (int): Dimension of the word vectors.

    Returns:
        dict_hstack (dict): Dictionary with user ID as keys and hstacked 
        vectors as values.
        dict_vstack (dict): Dictionary with user ID as keys and vstacked
          vectors as values.

    """
    def tfidf_weighted_word2vec(data, colname, vector_dim = 768):
        """
        Function generates necessary components to derive TF-IDF weighted Word2Vec
        from an entire data column.    

        Args:
            data (pandas.DataFrame): Dataset containing columns text for converting 
            into word2vec
            col_names (str) : Name of the target column on which the operation 
            needs to be performed

        Returns: 
            w2v (dict) : Dictionary with keys as words and values as i.e. word2vec 
            model generated respective vectors
            word2weight (dict) : Dictionary with words and their corresponding 
            TF-IDF weights 
            vocab(dict) : Vocabulary dictionary with word indices
        """

        # Generating coldata
        c = data[colname].tolist()
        c = [str(x) for x in c]
        coldata = []
        
        # Creating model and tokenizing words for the moedl
        model = Word2Vec(
            window = 2, min_count = 3, sg = 1, vector_size = vector_dim
        )
        
        for x in c:
            coldata.append(gensim.utils.simple_preprocess(x))

        # Creating model vocabulary from the tokens and then training and 
        # later bundling into a dictionary
        
        model.build_vocab(coldata)
        model.train(corpus_iterable=coldata, total_examples= model.corpus_count, 
                    epochs=model.epochs)
        
        w2v = dict(zip(model.wv.index_to_key, model.wv.vectors.round(3)))
        
        # Creating TFIDF vectorizer model
        
        tfidfvectorizer = TfidfVectorizer()
        tfidfvectorizer.fit_transform(c)
        vocab = tfidfvectorizer.vocabulary_.items()

        # Generating word2weight dictionary of word and its TFIDF values
        word2weight = [(w, round(tfidfvectorizer.idf_[i], 3)) 
                    for w, i in tfidfvectorizer.vocabulary_.items()]
        word2weight  = dict(word2weight)

        return w2v, word2weight, vocab

    def tfidfw2v_vectorizer(text, w2v, word2weight, vector_dim = 768):
        """

        Perform TF-IDF weighted Word2Vec embdedding on a tet column in a DataFrame
        using the word2vec related components provided on the text. 

        Function calculates the TFIDF (from scikit-learn's TFIDFfVectorizer) 
        weighted word2vec (from gensim.Word2Vec) as per the following formulae:
        Tfidf w2v (w1,w2..) = 
        (tfidf(w1) * w2v(w1) + tfidf(w2) * w2v(w2) + …)/(tfidf(w1) + tfidf(w2) + …
        from various inputs. 

        Args:
            text (str): Input text for which to calculate the TF-IDF weighted 
            Word2Vec vector.
            w2v (dict): Dictionary with keys as words and values as their 
            respective vectors.
            word2weight (dict): Dictionary with words and their corresponding
            TF-IDF  weights.

        Returns:
            np.ndarray: TF-IDF weighted Word2Vec vector for the input text.

        """
        words = text.split() 

        if len(words) == 0:
            
            return np.zeros(vector_dim) 

        else:
            numerator_vector = np.zeros(vector_dim)
            denominator_value = 0.0
            
            for word in words:
                
                if word in w2v.keys() and word in word2weight.keys():
                    
                    numerator_val = words.count(word)*word2weight[word]*w2v[word]
                    numerator_vector += numerator_val
                
                    denominator_val = words.count(word)*word2weight[word]
                    denominator_value += denominator_val
            
            if denominator_value == 0.0:
                
                return np.zeros(vector_dim)
        
            else: 
                
                return np.round(numerator_vector/denominator_value, 3)
    
    # Defining functions that encode and pad boolean and float values

    def encode_and_pad_boolean_columns(fdata, bool_column, vector_dim = 768):
        """
        Encode bookean columns in a pandas DataFrame using OneHot Encoder
        
        Args:
            fdata (pandas DataFrame): upon whose boolean columns the encoding is to 
            executed

            bool_column (list): List containing the boolean columns names to be 
            encoded

            vector_dim (int): Dimension of the w2v_vectors
        
        Returns:
            None, modifies the DataFrame in place adding new columns with one hot 
            encoded data
        
        """
        onehotencoder = OneHotEncoder(sparse_output = False, handle_unknown = 'ignore')
        
        for colname in bool_column:
            fdata[colname + "__w2v"] = [
                np.pad(x,  
                    (0, vector_dim - (len(x) % vector_dim)), 
                    'constant') for x in onehotencoder.fit_transform(
                        np.reshape(np.array(fdata[colname]), (-1, 1))
                        )
                        ]

    def pad_float_columns(fdata, float_column, vector_dim = 768):
        """
        Pads the specified float columns in the fdata pandas DataFrame so that the
        final value has a length equal to vector_dim

        Args:
            fdata (pandas DataFrame): Data frame containing the float value
            float_column (list): List of column names containig the float data
            vector_dim (int): Dimension of the vector the columns will be padded

        Returns:
            None: Converts/ modifies the data and generates the new columns
        """
        for colname in float_column:
            fdata[colname + "__w2v"] = [np.pad(
                x, 
                (0, vector_dim - (len(x) % vector_dim)), 
                'constant'
            ) for x in (np.reshape(
                np.array(fdata[colname]), (-1, 1)
            ))]
            
    def hstacker(row_arrays):
        """
        Function that concatenates each of the column data for each row
        """
        return np.concatenate(row_arrays)

    def vstacker(row_arrays):
        """
        Gives the mean vector for the vectors in columns row-wise
        """
        return np.mean(row_arrays)
    
    # Gathering the data and dropping duplicates
    data__ = data[[uid_column_name]+ [x + "__w2vpp" for x in str_column] + bool_column + float_column]

    # Applying encode_pad_boolean_columns and pad_float_columns 

    encode_and_pad_boolean_columns(data__, bool_column)
    pad_float_columns(data__, float_column)

     # Gathering and applying BERT base embedded vector for opportunity columns
    
    dict_hstack = {}
    dict_vstack = {}

    # Gathering string data only along with uid_column_name

    for colname in  str_column:
        w2v, word2weight, vocab = tfidf_weighted_word2vec(data__, colname + "__w2vpp")
        data__[colname + "__w2v"] = data__[colname + "__w2vpp"].apply(lambda x: tfidfw2v_vectorizer(x, w2v,word2weight))
    
    data__[uid_column_name + "__w2v_hstack"] = data__[[m + "__w2v" for m in str_column + bool_column + float_column]].apply(hstacker, axis = 1)
    data__[uid_column_name + "__w2v_vstack"] = data__[[m + "__w2v" for m in str_column + bool_column + float_column]].apply(vstacker, axis = 1)
    
    for index, row in data__.iterrows():
        dict_hstack[data__.at[index, uid_column_name]] = data__.at[index, uid_column_name + "__w2v_hstack"]
        dict_vstack[data__.at[index, uid_column_name]] = data__.at[index, uid_column_name + "__w2v_vstack"]
    
    return dict_hstack, dict_vstack

def tfidf_weighted_word2vec(data, colname, vector_dim = 768):
    """
    Function generates necessary components to derive TF-IDF weighted Word2Vec
    from an entire data column.    

    Args:
        data (pandas.DataFrame): Dataset containing columns text for converting 
        into word2vec
        col_names (str) : Name of the target column on which the operation 
        needs to be performed

    Returns: 
        w2v (dict) : Dictionary with keys as words and values as i.e. word2vec 
        model generated respective vectors
        word2weight (dict) : Dictionary with words and their corresponding 
        TF-IDF weights 
        vocab(dict) : Vocabulary dictionary with word indices
    """

    # Generating coldata
    c = data[colname].tolist()
    c = [str(x) for x in c]
    coldata = []
    
    # Creating model and tokenizing words for the moedl
    model = Word2Vec(
        window = 2, min_count = 3, sg = 1, vector_size = vector_dim
    )
    
    for x in c:
        coldata.append(gensim.utils.simple_preprocess(x))

    # Creating model vocabulary from the tokens and then training and 
    # later bundling into a dictionary
    
    model.build_vocab(coldata)
    model.train(corpus_iterable=coldata, total_examples= model.corpus_count, 
                epochs=model.epochs)
    
    w2v = dict(zip(model.wv.index_to_key, model.wv.vectors.round(3)))
    
    # Creating TFIDF vectorizer model
    
    tfidfvectorizer = TfidfVectorizer()
    tfidfvectorizer.fit_transform(c)
    vocab = tfidfvectorizer.vocabulary_.items()

    # Generating word2weight dictionary of word and its TFIDF values
    word2weight = [(w, round(tfidfvectorizer.idf_[i], 3)) 
                for w, i in tfidfvectorizer.vocabulary_.items()]
    word2weight  = dict(word2weight)

    return w2v, word2weight, vocab

def tfidfw2v_vectorizer(text, w2v, word2weight, vector_dim = 768):
    """

    Perform TF-IDF weighted Word2Vec embdedding on a tet column in a DataFrame
    using the word2vec related components provided on the text. 

    Function calculates the TFIDF (from scikit-learn's TFIDFfVectorizer) 
    weighted word2vec (from gensim.Word2Vec) as per the following formulae:
    Tfidf w2v (w1,w2..) = 
    (tfidf(w1) * w2v(w1) + tfidf(w2) * w2v(w2) + …)/(tfidf(w1) + tfidf(w2) + …
    from various inputs. 

    Args:
        text (str): Input text for which to calculate the TF-IDF weighted 
        Word2Vec vector.
        w2v (dict): Dictionary with keys as words and values as their 
        respective vectors.
        word2weight (dict): Dictionary with words and their corresponding
        TF-IDF  weights.

    Returns:
        np.ndarray: TF-IDF weighted Word2Vec vector for the input text.

    """
    words = text.split() 

    if len(words) == 0:
        
        return np.zeros(vector_dim) 

    else:
        numerator_vector = np.zeros(vector_dim)
        denominator_value = 0.0
        
        for word in words:
            
            if word in w2v.keys() and word in word2weight.keys():
                
                numerator_val = words.count(word)*word2weight[word]*w2v[word]
                numerator_vector += numerator_val
            
                denominator_val = words.count(word)*word2weight[word]
                denominator_value += denominator_val
        
        if denominator_value == 0.0:
            
            return np.zeros(vector_dim)
    
        else: 
            
            return np.round(numerator_vector/denominator_value, 3)

# Defining functions that encode and pad boolean and float values

def encode_and_pad_boolean_columns(fdata, bool_column, vector_dim = 768):
    """
    Encode bookean columns in a pandas DataFrame using OneHot Encoder
    
    Args:
        fdata (pandas DataFrame): upon whose boolean columns the encoding is to 
        executed

        bool_column (list): List containing the boolean columns names to be 
        encoded

        vector_dim (int): Dimension of the w2v_vectors
    
    Returns:
        None, modifies the DataFrame in place adding new columns with one hot 
        encoded data
    
    """
    onehotencoder = OneHotEncoder(sparse_output = False, handle_unknown = 'ignore')
    
    for colname in bool_column:
        fdata[colname + "__w2v"] = [
            np.pad(x,  
                (0, vector_dim - (len(x) % vector_dim)), 
                'constant') for x in onehotencoder.fit_transform(
                    np.reshape(np.array(fdata[colname]), (-1, 1))
                    )
                    ]

def pad_float_columns(fdata, float_column, vector_dim = 768):
    """
    Pads the specified float columns in the fdata pandas DataFrame so that the
    final value has a length equal to vector_dim

    Args:
        fdata (pandas DataFrame): Data frame containing the float value
        float_column (list): List of column names containig the float data
        vector_dim (int): Dimension of the vector the columns will be padded

    Returns:
        None: Converts/ modifies the data and generates the new columns
    """
    for colname in float_column:
        fdata[colname + "__w2v"] = [np.pad(
            x, 
            (0, vector_dim - (len(x) % vector_dim)), 
            'constant'
        ) for x in (np.reshape(
            np.array(fdata[colname]), (-1, 1)
        ))]
        
def hstacker(row_arrays):
    """
    Function that concatenates each of the column data for each row
    """
    return np.concatenate(row_arrays)

def vstacker(row_arrays):
    """
    Gives the mean vector for the vectors in columns row-wise
    """
    return np.mean(row_arrays)

# Code for function:

def modelbased_embedder(data, uid_column_name, str_column, bool_column, float_column, hugging_face_model_name):
    """
    Embeds textual data using a pre-trained transformer model from Hugging Face.

    Parameters:
        data (pd.DataFrame): The input DataFrame containing textual data to be 
        embedded.
        uid_column_name (str): The name of the column in the DataFrame that 
        contains unique identifiers for each row.
        str_column (list): List containing the names(str) of columns onto which
        modelbased_embedder are to be performed.
        bool_column: List containing the names(str) of boolean columns
        float_column: List containing the names(str) of the float data
        hugging_face_model_name (str): The name or path of the pre-trained 
        transformer model from Hugging Face.

    Returns:
        dict_hstack (dict): A dictionary mapping unique identifiers to 
        horizontally stacked BERT embeddings.
        dict_vstack (dict): A dictionary mapping unique identifiers to 
        vertically stacked mean-pooled BERT embeddings.

    Notes:
        1. This function pads the flaot and boolean data to the transformer's 
        vector size (i.e. config.hidden_size) so as to perform vector/tensor
        mean(explained later). Therefore, in the horizontal 
        stacking/concatenating, the vector adds unnecessary dimensions. 
        2. This function uses the specified Hugging Face model to tokenize and 
        embed text data.
        3. The embeddings are computed for each row in the input DataFrame, and 
        the results are stored in dictionaries.
        4. Horizontal stacking (`job_opportunityid_bert_dict_hstack`) concatenates 
        embeddings for each column.
        5. Vertical stacking (`job_opportunityid_bert_dict_vstack`) computes the 
        mean-pooled embeddings across all columns.
        6. The resulting dictionaries can be used for downstream tasks such as 
        machine learning or similarity comparisons.
    """

    # Defining functions that encode and pad boolean and float values

    def encode_and_pad_boolean_columns(data, bool_column, vector_dim = 768):
        """
        Encode bookean columns in a pandas DataFrame using OneHot Encoder
        
        Args:
            data (pandas DataFrame): upon whose boolean columns the encoding is 
            to executed
            bool_column (list): List containing the boolean columns names to be 
            encoded
            vector_dim (int): Dimension of the w2v_vectors
        
        Returns:
            None, modifies the DataFrame in place adding new columns with one 
            hot encoded data
        
        """

        if 'OneHotEncoder' in dir():
            onehotencoder = OneHotEncoder(sparse_output = False, handle_unknown = 'ignore')
        else:
            from sklearn.preprocessing import OneHotEncoder
            onehotencoder = OneHotEncoder(sparse_output = False, handle_unknown = 'ignore')
        
        for colname in bool_column:
            data[colname + "__"] = [
                np.pad(x,  
                    (0, vector_dim - (len(x) % vector_dim)), 
                    'constant') for x in onehotencoder.fit_transform(
                        np.reshape(np.array(data[colname]), (-1, 1))
                        )
                    ]

    def pad_float_columns(data, float_column, vector_dim = 768):
        """
        Pads the specified float columns in the fdata pandas DataFrame so that 
        the final value has a length equal to vector_dim

        Args:
            fdata (pandas DataFrame): Data frame containing the float value
            float_column (list): List of column names containing the float data
            vector_dim (int): Dimension of the vector the columns will be 
            padded

        Returns:
            None: Converts/ modifies the data and generates the new columns
        """
        for colname in float_column:
            data[colname + "__"] = [np.pad(
                x, 
                (0, vector_dim - (len(x) % vector_dim)), 
                'constant'
            ) for x in (np.reshape(
                np.array(data[colname]), (-1, 1)
            ))]

    # Main function begins  
    # Applying hugging face model and tokenizer
    # Loading model and tokenizer
    '''Shifting to GPU for faster calculations - as applicable (user should 
    test if frequent transitioning between GPU and CPU, and viceversa 
    increases time-complexity)'''

    # Use the below code on higher config machines
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     device  = torch.device("mps")
    else:
        device = torch.device("cpu") 
    
    if 'AutoTokenizer' in dir() and 'AutoModel' in dir():
        tokenizer = AutoTokenizer.from_pretrained(hugging_face_model_name) 
        model = AutoModel.from_pretrained(hugging_face_model_name).to(device) 

    else:
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(hugging_face_model_name) 
        model = AutoModel.from_pretrained(hugging_face_model_name).to(device)   
    
    # Gathering data 
    data__ = data[
        [uid_column_name]+ 
        [x + "__trnsfrmrpp" for x in str_column] + 
        bool_column + 
        float_column
    ].drop_duplicates() 

    # Applying encode_pad_boolean_columns and pad_float_columns 

    encode_and_pad_boolean_columns(data__, bool_column, vector_dim = model.config.hidden_size)
    pad_float_columns(data__, float_column, vector_dim = model.config.hidden_size)
    
    # Gathering and applying BERT base embedded vector for opportunity columns
    
    dict_hstack = {}
    dict_vstack = {}
    str_data__ = data__[[x + "__trnsfrmrpp" for x in str_column]] # Gathering string data only

    for index, row in tqdm(
        str_data__.iterrows(), desc = "Processing rows", total = len(data__)
    ):
        
        embeddings_values = []
        
        for column in str_data__.columns:
            text = str_data__.at[index, column]

            # Taking care of empty text
            if not text:
                zero_embeddings = np.zeros((model.config.hidden_size,))
                embeddings_values.append(zero_embeddings)
                continue
            
            #Taking care of the blank [] obtained because of the lack of punctuation mark
            if text and text[-1] not in ".?!":
                text+= "."

        
            sentences = nltk.tokenize.sent_tokenize(text)
            
            inputs = tokenizer(
                sentences, 
                padding = True, 
                truncation = True, 
                return_tensors = "pt"
            ).to(device) 
            
            with torch.no_grad():
                output = model(**inputs)
                
            embeddings_values.append(
                np.mean(
                    output.last_hidden_state.mean(dim=1).cpu().numpy(), 
                    axis = 0
                )
            )
        
        # Now adding boolean and float values
        # Adding padded bool_column values to the embeddings_values
        for column in bool_column:
            embeddings_values.append(data__.at[index, column + "__"])
            
        # Adding padded float_column values to the embeddings_values
        for column in float_column:
            embeddings_values.append(data__.at[index, column + "__"])

        # Stacking the embeddings, boolean and float values 
        vector_h = np.hstack(tuple(embeddings_values))
        vector_v = np.mean((tuple(embeddings_values)), axis = 0)
        
        dict_hstack[data__.at[index, uid_column_name]] = vector_h
        dict_vstack[data__.at[index, uid_column_name]] = vector_v

    return dict_hstack, dict_vstack