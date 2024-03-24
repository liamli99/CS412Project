import os
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
class DataReader:
    def __init__(self, path, names=['exp', 'methy', 'mirna']):
        self.path = path
        self.cancer_type = path.split('/')[-1]
        self.names = names
        self.load_data()
    
    def load_omic(self, omic_path):
        df = pd.read_csv(omic_path, sep=' ')
        # import pdb; pdb.set_trace()
        # df.index: get rows name
        # df.columns: get columns name
        # df.T: transpose the matrix
        # df.sort_values(by="B") sort the matrix by column B
        # df.loc[row_name]: get the row by row_name
        # df[col_name]: get the column by col_name
        # df.iloc[row_index]: get the row by row_index
        # df.iloc[:, col_index]: get the column by col_index
        return df
    
    def load_data(self):
        for name in self.names:
            setattr(self, name, self.load_omic(f"{self.path}/{name}"))
        # self.exp = self.load_omic(f"{self.path}/exp")
        # self.methy = self.load_omic(f"{self.path}/methy")
        # self.mirna = self.load_omic(f"{self.path}/mirna")
    
    def save_data(self, path_to_save):
        os.makedirs(path_to_save, exist_ok=True)
        for name in self.names:
            getattr(self, name).to_csv(f"{path_to_save}/{name}", sep=' ',)
        # self.exp.to_csv(f"{path_to_save}/exp", sep=' ', index=False)
        # self.methy.to_csv(f"{path_to_save}/methy", sep=' ', index=False)
        # self.mirna.to_csv(f"{path_to_save}/mirna", sep=' ', index=False)
    

if __name__ == "__main__":
    path = "data/origin/aml"
    data = DataReader(path)