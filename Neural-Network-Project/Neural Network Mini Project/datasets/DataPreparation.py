from abc import ABC, abstractmethod
from pdb import set_trace
import warnings
import numpy as np
import pandas as pd

from itcs4156.util.data import split_data, dataframe_to_array, binarize_classes
from itcs4156.datasets.HousingDataset import HousingDataset
from itcs4156.datasets.MNISTDataset import MNISTDataset

class DataPreparation():
    def __init__(self, target_pipe, feature_pipe):
        self.target_pipe = target_pipe
        self.feature_pipe = feature_pipe
        
    @abstractmethod
    def data_prep(self):
        pass
    
    def fit(self, X, y=None):
        if self.target_pipe  is not None:
            self.target_pipe.fit(y)
            
        if self.feature_pipe is not None:
            self.feature_pipe.fit(X)

    def transform(self, X, y=None):
        if self.target_pipe is not None:
            y = self.target_pipe.transform(y)
            
        if self.feature_pipe is not None:
            X = self.feature_pipe.transform(X)

        return X, y
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        X, y = self.transform(X, y)
        return X, y
    
class HousingDataPreparation(DataPreparation):
    def __init__(self, target_pipe, feature_pipe, use_features):
        super().__init__(target_pipe, feature_pipe)
        self.use_features = use_features
        
    def data_prep(self, return_array=False):
        
        if self.target_pipe is not None:
            warnings.warn("Target pipeline is not needed for the Boston House Price dataset. " \
                          "Even though you passed a Pipeline for `target_pipe`, " \
                          "`target_pipe` will be set to None.")
            self.target_pipe = None
        
        if return_array: 
            print("Returning data as NumPy array...")
            return_df = False
        
        print(f"Attempting to use the following features: {self.use_features}")
        housing_dataset = HousingDataset()
        house_df_trn, house_df_vld = housing_dataset.load()
        
        X_trn_df, y_trn_df, X_vld_df, y_vld_df = split_data(
            df_trn=house_df_trn,
            df_vld=house_df_vld,
            use_features=self.use_features,
            label_name='MEDV',
            return_df=return_df
        )

        X_trn_df, y_trn_df = self.fit_transform(X=X_trn_df, y=y_trn_df)
        X_vld_df, y_vld_df = self.transform(X=X_vld_df, y=y_vld_df)
        
        return X_trn_df, y_trn_df, X_vld_df, y_vld_df
    
class MNISTDataPreparation(DataPreparation):
    def __init__(self, target_pipe, feature_pipe):
        super().__init__(target_pipe, feature_pipe)
        
    def data_prep(self, binarize=False, return_array=False):
        mnist_dataset = MNISTDataset()
        X_trn_df, y_trn_df, X_vld_df, y_vld_df = mnist_dataset.load()
        
        # Converts MNIST problem to classifying ONLY 1s vs 0s
        if binarize:
            X_trn_df, y_trn_df = binarize_classes(
                X_trn_df, 
                y_trn_df, 
                pos_class=[1],
                neg_class=[0], 
            )
            
            X_vld_df, y_vld_df = binarize_classes(
                X_vld_df, 
                y_vld_df, 
                pos_class=[1], 
                neg_class=[0], 
            )

        X_trn_df, y_trn_df = self.fit_transform(X=X_trn_df, y=y_trn_df)
        X_vld_df, y_vld_df = self.transform(X=X_vld_df, y=y_vld_df)

        if return_array:
            print("Returning data as NumPy array...")
            return dataframe_to_array([X_trn_df, y_trn_df, X_vld_df, y_vld_df])
            
        return X_trn_df, y_trn_df, X_vld_df, y_vld_df
