import numpy as np
import pandas as pd
import os
from sklearn.feature_selection import chi2
from scipy import stats
from dataset import DataReader
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

# process the data
# implement feature selection methods here
class Processor:
    def __init__(self, data_path_list):
        self.dr_list = []
        self.names = ['exp', 'methy', 'mirna']
        for path in data_path_list:
            self.dr_list.append(DataReader(path, self.names))
        
    # evaluate whether the data follows normal distribution
    def norm_evaluation(self, name, alpha=0.05):

        for dr in self.dr_list:
            normality_results = {"Shapiro-Wilk": {"Pass": [], "Fail": []}}

            row_names = getattr(dr, name).index
            for row in row_names:
                data = np.array(getattr(dr, name).loc[row].values)

                stat, p_shapiro = stats.shapiro(data)
                if p_shapiro > alpha:
                    normality_results["Shapiro-Wilk"]["Pass"].append(row)
                else:
                    normality_results["Shapiro-Wilk"]["Fail"].append(row)
            print("Normality test for cancer", dr.cancer_type )
            print(f"Shapiro-Wilk Test - Pass: {len(normality_results['Shapiro-Wilk']['Pass'])}")
            print(f"Shapiro-Wilk Test - Fail: {len(normality_results['Shapiro-Wilk']['Fail'])} \n")
        return normality_results
    
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
        print(f"Number of features with p value less than 0.05: {len(indexs[0])}")
        print(f"Total number of features: {len(p_values)} \n")
        return p_values

    def Pearson_Correlation(self, name):
        for dr in self.dr_list:
            data_list = []
            row_names = getattr(dr, name).index
            for row in row_names:
                data_list.append(np.array(getattr(dr, name).loc[row].values))
            data = np.array(data_list).T
            correlation_matrix = np.corrcoef(data, rowvar=False)

            # plt.figure(figsize=(10, 6))
            # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
            # plt.title("Correlation Matrix")
            # plt.show()
            print(f"Correlation Matrix for cancer {dr.cancer_type} is: {correlation_matrix}")
            print(f"Correlation Matrix's shape for cancer {dr.cancer_type} is: {correlation_matrix.shape}\n")
    
if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    data_path_list = ["data/filtered_common_features/aml", "data/filtered_common_features/sarcoma",]
    processor = Processor(data_path_list)
    for name in ['exp', 'mirna']:
        print("************************************************")
        print("Analysis for category:", name)
        print("************************************************\n")

        normality_results = processor.norm_evaluation(name)

        p_values = processor.Student_T_Test(name)
        processor.Pearson_Correlation("exp")

