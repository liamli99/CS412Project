import numpy as np
import pandas as pd
import os
from sklearn.feature_selection import chi2
from scipy import stats
from dataset import DataReader

# process the data
# implement feature selection methods here
class Processor:
    def __init__(self, data_path_list):
        self.dr_list = []
        self.names = ['exp', 'methy', 'mirna']
        for path in data_path_list:
            self.dr_list.append(DataReader(path, self.names))
    
    # evaluate whether the data follows normal distribution
    def norm_evaluation(self, ):
        pass
    
    def Student_T_Test(self, name):
        # Chi-Square Test
        # input: name of the omic
        # output: p values of the features
        row_names = getattr(self.dr_list[0], name).index
        p_values = []
        for row in row_names:
            data = []
            for dr in self.dr_list:
                data.append(np.array(getattr(dr, name).loc[row].values))
            # import pdb; pdb.set_trace()
            # TODO: should evaluate the variance of the two groups first: stats.levene(A, B) 
            p_value = stats.ttest_ind(data[0],data[1],equal_var=True)
            p_values.append(p_value.pvalue)
        p_values = np.array(p_values)
        indexs = np.where(p_values < 0.05)
        print(f"number of features with p value less than 0.05: {len(indexs[0])}")
        return p_values
    
if __name__ == "__main__":
    data_path_list = ["data/filtered_common_features/aml", "data/filtered_common_features/sarcoma",]
    processor = Processor(data_path_list)
    p_values = processor.Student_T_Test("methy")
    # print(p_values, len(p_values))