from typing import Union, List
from copy import deepcopy 
from pdb import set_trace

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

class Standardization(BaseEstimator, TransformerMixin):
    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: pd.DataFrame = None):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return self
    
    def transform(self, X):
        return (X  - self.mean) / (self.std + self.epsilon)

class AddBias(BaseEstimator, TransformerMixin):
    def fit(self,
            X: pd.DataFrame, 
            y: pd.DataFrame = None) -> pd.DataFrame:
        return self
    
    def transform(self,
                  X: pd.DataFrame, 
                  y: pd.DataFrame = None) -> pd.DataFrame:
        X = X.copy()
        X.insert(0, 'bias', 1)
        return X

class ImageNormalization(BaseEstimator, TransformerMixin): 
    def fit(self,
            X: pd.DataFrame, 
            y: pd.DataFrame = None) -> pd.DataFrame:
        return self
    
    def transform(self,
                  X: pd.DataFrame, 
                  y: pd.DataFrame = None) -> pd.DataFrame:
        return (X/255).astype(np.float16)

class OneHotEncoding(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names='auto'):
        self.feature_names = feature_names
        self.encoder = OneHotEncoder(categories=feature_names, sparse=False)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        
        self.encoder.fit(X)
        
        # Store names of features
        try:
            self.feature_names = self.encoder.get_feature_names_out()
        except AttributeError:
            self.feature_names = self.encoder.get_feature_names()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        
        one_hot =  self.encoder.transform(X)

        return pd.DataFrame(one_hot, columns=self.feature_names)
    

def feature_label_split(df: pd.DataFrame, 
                        feature_names: List[str], 
                        label_name: str, 
                        return_df: bool = False) -> Union[np.ndarray, pd.DataFrame]:
    """ Splits DataFrame into features and labels
    
        Args:
            df: DataFrame which contains the features and label
            
            feature_name: Name of the columns to be used as features. Only the names given
                will be returned.
            
            label_feature: Name of the column which will be used as the label.
            
            return_df: If True then X and y will be returned as a DataFrame

    """
    if label_name in feature_names:
        err = f"Can not use your label {label_name} as a feature."
        raise ValueError(err)
    
    if isinstance(feature_names, (list, pd.Index, np.ndarray, tuple)):
        n = len(feature_names)
    else:
        n = 1

    X = df[feature_names]
    y = df[label_name]
    
    if return_df:
        return X, y
    
    return X.values.reshape((-1,n)), y.values.reshape((-1,1))

def split_data(df_trn: pd. DataFrame, 
               df_vld: pd.DataFrame, 
               use_features: List[str], 
               label_name: str, 
               return_df: bool = False):
    """
        Args:
            df_trn: Your training DataFrame
            
            df_vld: Your validation DataFrame
            
            use_features: List of features names you want to include 
                in your training and validation data.
            
            label_name: The name of the target/label.
            
            return_df: If True a DataFrame will be returned.
    """
    
    X_trn, y_trn = feature_label_split(df_trn, 
                                       feature_names=use_features, 
                                       label_name=label_name, 
                                       return_df=return_df)
    X_vld, y_vld = feature_label_split(df_vld, 
                                       feature_names=use_features, 
                                       label_name=label_name, 
                                       return_df=return_df)
    
    return X_trn, y_trn, X_vld, y_vld

def binarize_classes(
    X_df: pd.DataFrame,
    y_df: pd.DataFrame, 
    pos_class: List, 
    neg_class: List,
    neg_label = -1,
    pos_label = 1,
) -> pd.DataFrame:
    """ Converts a multi-class classification problem into a binary classification problem.
        Args:
            X_df: Pandas DataFrame containing input features.
            
            y_df: Pandas DataFrame continaing target/labels.
            
            pos_class: A list of unique labels/targets found in y_df that will be used
                to create the positive class.
                
            neg_class: A list of unique labels/targets found in y_df that will be used
                to create the negative class.
                           
            neg_label: Label used for the negative class.
            
            pos_label: Label used for the positive class.
    """
    
    
    if not isinstance(X_df, pd.DataFrame):
        err = f"X_df is of type {type(X_df)}: expected pd.DataFrame"
        raise TypeError(err)
    
    if not isinstance(y_df, pd.DataFrame):
        err = f"y_df is of type {type(y_df)}: expected pd.DataFrame"
        raise TypeError(err)
    
    pos_locs = y_df.isin(pos_class).values
    pos_X = X_df[pos_locs].copy()
    pos_X.reset_index(inplace=True, drop=True)
    pos_y = y_df[pos_locs].copy()
    pos_y.reset_index(inplace=True, drop=True)
    pos_y.loc[:] = pos_label
    
    neg_locs = y_df.isin(neg_class).values
    neg_X = X_df[neg_locs].copy()
    neg_X.reset_index(inplace=True, drop=True)
    neg_y = y_df[neg_locs].copy()
    neg_y.reset_index(inplace=True, drop=True)
    neg_y.loc[:] = neg_label

    new_X_df = pd.concat([pos_X, neg_X])
    new_y_df = pd.concat([pos_y, neg_y])
    
    return new_X_df, new_y_df

def dataframe_to_array(dfs: List[pd.DataFrame]):
    """ Converts any Pandas DataFrames into NumPy arrays
    
        Args:
            dfs: A list of Pandas DataFrames to be converted.
    
    """
    arrays = []
    for df in dfs:
        if isinstance(df, np.ndarray):
            arrays.append(df)
        else:
            arrays.append(df.values)
    return arrays

def copy_dataset(data: list):
    """ Deep copies all passed data.

        Args:
            data: A list of data objects such as NumPy arrays or Pandas DataFrames.
    """  
    return [deepcopy(d) for d in data]