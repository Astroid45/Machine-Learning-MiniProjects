from datasets.Dataset import Dataset
import os
import pandas as pd

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
