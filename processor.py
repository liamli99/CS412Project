import numpy as np
import pandas as pd
import os
from sklearn.feature_selection import chi2
from scipy import stats
from dataset import DataReader
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# process the data
# implement feature selection methods here
class Processor:
    def __init__(self, data_path_list):
        self.dr_list = []
        self.names = ['clinical', 'exp', 'methy', 'mirna', ]
        for path in data_path_list:
            self.dr_list.append(DataReader(path, self.names))
    
    def normalize(self, name, row_names=None):
        if row_names is not None:
            dr_combined = pd.concat([getattr(dr, name).loc[row_names] for dr in self.dr_list], axis=1).T
            mean = dr_combined.mean()
            std = dr_combined.std()
            for dr in self.dr_list:
                data = getattr(dr, name).loc[row_names].T
                data = (data - mean) / std
                exec(f"dr.{name}.loc[{row_names}] = data.T")
        else:
            dr_combined = pd.concat([getattr(dr, name) for dr in self.dr_list], axis=1).T
            mean = dr_combined.mean()
            std = dr_combined.std()
            for dr in self.dr_list:
                data = getattr(dr, name).T
                data = (data - mean) / std
                setattr(dr, name, data.T)
    
    # evaluate whether the data follows normal distribution
    def norm_evaluation(self, name, alpha=0.01):
        for dr in self.dr_list:
            normality_results = {"Shapiro-Wilk": {"Pass": [], "Fail": []}}

            row_names = getattr(dr, name).index
            for row in row_names:
                data = np.array(getattr(dr, name).loc[row].values)
                # import pdb; pdb.set_trace()
                stat, p_shapiro = stats.shapiro(data)
                if p_shapiro > alpha:
                    normality_results["Shapiro-Wilk"]["Pass"].append(row)
                else:
                    normality_results["Shapiro-Wilk"]["Fail"].append(row)
            print("Normality test for cancer", dr.cancer_type )
            print(f"Shapiro-Wilk Test - Pass: {len(normality_results['Shapiro-Wilk']['Pass'])}")
            print(f"Shapiro-Wilk Test - Fail: {len(normality_results['Shapiro-Wilk']['Fail'])} \n")
        return normality_results
    
    def Student_T_Test(self, name, row_names=None, filter=False):
        # Chi-Square Test
        # input: name of the omic
        # output: p values of the features
        if row_names is None:
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
        if filter:
            for dr in self.dr_list:
                dr.filter_rows(name, row_names=indexs[0])
        print(f"Student T Test for {name}:")
        print(f"Number of features with p value less than 0.05: {len(indexs[0])}")
        print(f"Total number of features: {len(p_values)} \n")
        return p_values
    
    def Chi_Square_Test(self, name, row_names, filter=False):
        # Chi-Square Test
        # input: name of the omic
        # output: p values of the features
        p_values = []
        for row in row_names:
            data = np.zeros((len(self.dr_list), 2))
            for i, dr in enumerate(self.dr_list):
                data[0, i] = sum(getattr(dr, name).loc[row].values == 0)
                data[1, i] = sum(getattr(dr, name).loc[row].values == 1)
            chi2, p, dof, expected = chi2_contingency(data)
            p_values.append(p)
        p_values = np.array(p_values)
        indexs = np.where(p_values < 0.05)
        print(f"Chi_Square_Test for {name}:")
        print(f"Number of features with p value less than 0.05: {len(indexs[0])}")
        print(f"Total number of features: {len(p_values)} \n")
        print(f"Features with p value less than 0.05: {getattr(self.dr_list[0], name).index[indexs[0]]}")
        if filter:
            for dr in self.dr_list:
                dr.filter_rows(name, row_names=indexs[0])
        
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
    data_path_list = ["data/filtered_common_features/aml", "data/filtered_common_features/sarcoma",]
    processor = Processor(data_path_list)
    
    for name in ['clinical', ]:
        print("************************************************")
        print("Analysis for category:", name)
        print("************************************************\n")

        # normality_results = processor.norm_evaluation(name)
        processor.normalize(name, row_names=['age_at_initial_pathologic_diagnosis'])
        processor.Student_T_Test(name, row_names=['age_at_initial_pathologic_diagnosis'])
        p_values = processor.Chi_Square_Test(name, row_names=['gender_FEMALE', 'gender_MALE', \
                                    'history_of_neoadjuvant_treatment_No', 'history_of_neoadjuvant_treatment_Yes', \
                                    'vital_status_DECEASED', 'vital_status_LIVING'])
        
        # processor.Pearson_Correlation("exp")

