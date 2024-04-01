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
    
    def average_correlation(self, correlation_matrices):
        avg_correlations = []
        for cancer_type, correlation_matrix in correlation_matrices:
            avg_corr = np.nanmean(correlation_matrix, axis=1)
            avg_correlations.append((cancer_type, avg_corr))
        return avg_correlations

    def plot_comparison(self, avg_correlations, name):
    # Plot comparison of average correlations
        plt.figure(figsize=(10, 6))
        for cancer_type, avg_corr in avg_correlations:
            plt.plot(avg_corr, label=cancer_type)
        plt.xlabel(name)
        plt.ylabel('Average Correlation')
        plt.title('Average Correlation by Cancer Type')
        plt.legend()
        plt.grid(True)
        plt.show()

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
        correlation_matrices = []
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
            #print(f"Correlation Matrix for cancer {dr.cancer_type} is: {correlation_matrix}")
            print(f"Correlation Matrix's shape for cancer {dr.cancer_type} is: {correlation_matrix.shape}\n")
            correlation_matrices.append((dr.cancer_type, correlation_matrix))
        return correlation_matrices

    def perform_pca_for_each_omic(self):
        for omic_type in self.names:
            for data_reader in self.dr_list:
                omic_data = getattr(data_reader, omic_type)
                #print(omic_data)
                if omic_data.shape[0] == 0:
                    continue  # Skip if data is missing or empty
                # Normalize the data
                scaler = StandardScaler()
                omic_data_normalized = scaler.fit_transform(omic_data)
                pca = PCA(n_components=2)  # Specify the number of components
                principal_components = pca.fit_transform(omic_data_normalized)
                self.visualize_pca(principal_components, omic_type, data_reader)
                #self.visualize_pca_3d(principal_components, omic_type, data_reader)
                
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

    def visualize_pca_3d(self, principal_components, omic_type, data_reader):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(principal_components[:, 0], principal_components[:, 1], principal_components[:, 2], alpha=0.5)
        ax.set_title(f'PCA 3D Visualization for {omic_type}')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        plt.show()

    
if __name__ == "__main__":
    data_path_list = ["data/filtered_common_features/aml", "data/filtered_common_features/sarcoma",]
    processor = Processor(data_path_list)
    
    for name in [ 'mirna', 'exp']:
        print("************************************************")
        print("Analysis for category:", name)
        print("************************************************\n")

        normality_results = processor.norm_evaluation(name)

        p_values = processor.Student_T_Test(name)
        #print("P-values:", p_values)

        correlation_matrices = processor.Pearson_Correlation(name)
        
        avg_correlations = processor.average_correlation(correlation_matrices)
       
        processor.plot_comparison(avg_correlations, name)
    processor.perform_pca_for_each_omic()
