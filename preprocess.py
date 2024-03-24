import numpy as np
import pandas as pd
import os
from dataset import DataReader

# preprocess the data
class PreProcessor:
    def __init__(self, data_path_list):
        self.dr_list = []
        self.names = ['exp', 'methy', 'mirna']
        for path in data_path_list:
            self.dr_list.append(DataReader(path, self.names))
        
    def get_common_features(self, name):
        # get common features among all the cancer types
        # input: name of the omic
        common_features = getattr(self.dr_list[0], name).index
        for dr in self.dr_list[1:]:
            common_features = common_features.intersection(getattr(dr, name).index)
        return common_features

    def filter_common_features(self, ):
        # filter common features from all the omics
        # input: list of DataReader objects
        # output: list of DataReader objects
        for name in self.names:
            common_features_name = self.get_common_features(name)
            for i in range(len(self.dr_list)):
                condition = getattr(self.dr_list[i], name).index.isin(common_features_name)
                setattr(self.dr_list[i], name, getattr(self.dr_list[i], name)[condition])

    def save_data(self, path_to_save):
        os.makedirs(path_to_save, exist_ok=True)
        for dr in self.dr_list:
            dr.save_data(os.path.join(path_to_save, dr.cancer_type))

if __name__ == "__main__":
    data_path_list = ["data/origin/aml", "data/origin/sarcoma",]
    processor = PreProcessor(data_path_list)
    processor.filter_common_features()
    processor.save_data("data/filtered_common_features")