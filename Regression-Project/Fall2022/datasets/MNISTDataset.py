import os
from pdb import set_trace

import pandas as pd
import numpy as np

from itcs4156.datasets.Dataset import Dataset

class MNISTDataset(Dataset):

    def __init__(self): 
        
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "MNIST")

        self.data = {
           
            "urls" : {
                "train" : "https://drive.google.com/uc?export=download&id=1PepMZ-2uHWf0HO-PG9we03jJ46BRHNUJ",
                "val"  : "https://drive.google.com/uc?export=download&id=1ER4qAUWncgZLSfGL_-hKmMqhFbUaImYt"
            },

            "paths" : {
                "X_trn" : os.path.join(self.data_dir, 'train_images.csv'),
                "y_trn" : os.path.join(self.data_dir, 'train_labels.csv'),
                "X_vld" : os.path.join(self.data_dir, 'val_images.csv'),
                "y_vld" : os.path.join(self.data_dir, 'val_labels.csv')
            }
        }

        self.init_download()

    def init_download(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        for key, url in self.data["urls"].items():
            data_path = self.download(url, self.data_dir, key + '.zip')
            self.extract_zip(data_path, location=self.data_dir)

    def load(self, return_array=False):
        print("Loading dataset with Pandas...")
        X_trn = pd.read_csv(self.data["paths"]['X_trn'], header=None)
        y_trn =  pd.read_csv(self.data["paths"]['y_trn'], header=None)
        y_trn.columns = ['class']
        X_vld =  pd.read_csv(self.data["paths"]['X_vld'], header=None)
        y_vld =  pd.read_csv(self.data["paths"]['y_vld'], header=None)
        y_vld.columns = ['class']
        if return_array:
            print("Returning NumPy arrays")
            X_trn, y_trn, X_vld, y_vld = (X_trn.values,
                                          y_trn.values, 
                                          X_vld.values, 
                                          y_vld.values)
            
            y_trn = y_trn.reshape(-1,1)
            y_vld = y_vld.reshape(-1,1)
        print("Done!")
        return X_trn, y_trn, X_vld, y_vld
      

if __name__ == "__main__":
    MNISTDataset()