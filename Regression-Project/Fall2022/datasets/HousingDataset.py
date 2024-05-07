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

class HousingDataset(Dataset):

    def __init__(self): 
        
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

        self.data = {
           
            "urls" : {
                "train" : "https://drive.google.com/uc?export=download&id=1qbFX7dSCVdU8oj5uJVg5DOp-ckcZujGw",
                "val"   : "https://drive.google.com/uc?export=download&id=1k_0i6-wAZMPLFjPkk2VFj0P9ksceJvWN",
                "names" : "https://drive.google.com/uc?export=download&id=1zHWoQGrNByAh0yDCskNOAwQ7OARVHmGr"
            },

            "paths" : {

            },

            "columns" :  ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
        }

        self.init_download()

    def init_download(self):
        for key, url in self.data["urls"].items():
            data_path = self.download(url, self.data_dir, 'housing.' + key)
            self.data["paths"][key] = data_path

    def load(self):

        df_train = pd.read_csv(self.data["paths"]["train"], delim_whitespace=True, 
                        header=None, names=self.data["columns"])

        df_val = pd.read_csv(self.data["paths"]["val"], delim_whitespace=True, 
                        header=None, names=self.data["columns"])

        return df_train, df_val
      

if __name__ == "__main__":
    HousingDataset()