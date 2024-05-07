import os
from pdb import set_trace

import pandas as pd
import numpy as np


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
module_path = os.path.abspath(os.path.join('..\\..\\..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import os
import pandas as pd
import wget
import gzip, zipfile
import shutil


class Dataset(object):

    def __getitem__(self, index):
        raise NotImplementedError

    def download(self, url, data_dir, file_name):
        data_path = os.path.join(data_dir, file_name)
        if not os.path.exists(data_path):
            print("Dowloading from url: ", url)
            print("Saving to directory: ", data_dir)
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            wget.download(url, out=data_path)
            print("Download complete.\n")
        else:
            print("Skipping download. File already exists: {}\n".format(data_path))
        return data_path

    def extract_gz(self, input_file_path, output_file_path):
        if not os.path.exists(output_file_path):
            print("Extracting: ", input_file_path)
            with gzip.open(input_file_path, 'rb') as f_in:
                with open(output_file_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            print("Skipping extraction. File already exists: {}\n".format(output_file_path))

    def extract_zip(self, file_path, location='.'):
        print('Unzipping: ' + file_path + '\n')
        zip_ref = zipfile.ZipFile(file_path, 'r')
        zip_ref.extractall(path=location)
        zip_ref.close()


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
