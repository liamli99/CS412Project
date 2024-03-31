import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2
from scipy import stats
from dataset import DataReader
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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

    
    def perform_pca_for_each_omic(self):
        for omic_type in self.names:
            for data_reader in self.dr_list:
                omic_data = getattr(data_reader, omic_type)
                if omic_data.shape[0] == 0:
                    continue  # Skip if data is missing or empty
                pca = PCA(n_components=2)  # Specify the number of components
                principal_components = pca.fit_transform(omic_data)
                self.visualize_pca(principal_components, omic_type, data_reader)
                
    def visualize_pca(self, principal_components, omic_type, data_reader):
        # Visualize PCA results
        plt.figure(figsize=(8, 6))
        plt.scatter(principal_components[:, 0], principal_components[:, 1], marker='o', edgecolors='k')
        plt.title(f'PCA Visualization for {omic_type} Data ({data_reader.cancer_type})')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        
        # Add DataReader name (disease name) as annotation
        plt.text(0.05, 0.95, f'Disease: {data_reader.cancer_type}', transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        plt.grid(True)
        plt.show()
    
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
    processor.perform_pca_for_each_omic()
