from itcs4156.datasets.Dataset import Dataset
import os
import pandas as pd
import numpy as np

class FMNISTDataset(Dataset):

    def __init__(self): 
        
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "FMNIST")

        self.data = {
           
            "urls" : {
                "train" : "https://drive.google.com/uc?export=download&id=1s2SdJYeDWzNvEX3DGsAtH3T076xZ8rwe",
                "val"  : "https://drive.google.com/uc?export=download&id=1A_zYOsFabjcQ0R4ZU0baVA_zxxL2dI5F"
            },

            "paths" : {
                "X_train" : os.path.join(self.data_dir, 'train_images.npy'),
                "Y_train" : os.path.join(self.data_dir, 'train_labels.npy'),
                "X_val" : os.path.join(self.data_dir, 'val_images.npy'),
                "Y_val" : os.path.join(self.data_dir, 'val_labels.npy')
            }
        }

        self.init_download()

    def init_download(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        for key, url in self.data["urls"].items():
            data_path = self.download(url, self.data_dir, key + '.zip')
            self.extract_zip(data_path, location=self.data_dir)

    def load(self):
        print("Loading dataset..")
        X_train = np.load(self.data["paths"]['X_train'])
        Y_train = np.load(self.data["paths"]['Y_train'])
        X_val = np.load(self.data["paths"]['X_val'])
        Y_val = np.load(self.data["paths"]['Y_val']) 
        print("Done!")
        Y_train = Y_train.reshape(-1,1)
        Y_val = Y_val.reshape(-1,1)
        return (X_train, Y_train), (X_val, Y_val)
      

if __name__ == "__main__":
    dataset = FMNISTDataset()
    dataset.load()