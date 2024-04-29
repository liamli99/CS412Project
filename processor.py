import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2
from scipy import stats
from dataset import DataReader
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import seaborn as sns
from scipy.stats import chi2_contingency
import warnings
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# process the data
# implement feature selection methods here
class Processor:
    def __init__(self, data_path_list):
        self.dr_list = []
        self.names = ['exp', 'methy', 'mirna', ] # 'clinical', 
        for path in data_path_list:
            self.dr_list.append(DataReader(path, self.names))
        self.get_common_patient(filter=True)
        
    def normalize(self, name, row_names=None):
        # import pdb; pdb.set_trace()
        if row_names is not None:
            dr_combined = pd.concat([getattr(dr, name).loc[row_names] for dr in self.dr_list], axis=1).T
            mean = dr_combined.mean()
            std = dr_combined.std()
            for dr in self.dr_list:
                data = getattr(dr, name).loc[row_names].T
                data = (data - mean) / (std+1e-6)
                exec(f"dr.{name}.loc[{row_names}] = data.T")
        else:
            dr_combined = pd.concat([getattr(dr, name) for dr in self.dr_list], axis=1).T
            mean = dr_combined.mean()
            std = dr_combined.std()
            for dr in self.dr_list:
                data = getattr(dr, name).T
                data = (data - mean) / (std+1e-6)
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
    
    def box_cox_transformation(self, name, row_names=None):
        if row_names is not None:
            for dr in self.dr_list:
                data = getattr(dr, name).loc[row_names].T
                data, lamb = stats.boxcox(data)
                print(f"Lambda for {dr.cancer_type} is: {lamb}")
                exec(f"dr.{name}.loc[{row_names}] = data.T")
        else:
            for dr in self.dr_list:
                data = getattr(dr, name).T
                data, lamb = stats.boxcox(data)
                print(f"Lambda for {dr.cancer_type} is: {lamb}")
                setattr(dr, name, data.T)
    
    def ANOVA_Test(self, name, row_names=None, filter=False):
        if row_names is None:
            row_names = getattr(self.dr_list[0], name).index
        p_values = []
        for row in row_names:
            data = []
            for dr in self.dr_list:
                data.append(np.array(getattr(dr, name).loc[row].values))
            # import pdb; pdb.set_trace()
            stat, p_value = stats.f_oneway(data[0],data[1])
            p_values.append(p_value)
        p_values = np.array(p_values)
        indexs = np.where(p_values < 0.05)
        if filter:
            for dr in self.dr_list:
                dr.filter_rows(name, row_names=row_names[indexs[0]])
        
        print(f"ANOVA for {name}:")
        print(f"Number of features with p value less than 0.05: {len(indexs[0])}")
        
        return p_values
    
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
                dr.filter_rows(name, row_names=row_names[indexs[0]])
        
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
                dr.filter_rows(name, row_names=row_names[indexs[0]])
        return p_values
        
    def Pearson_Correlation(self, name):
        correlation_matrices = []
        df_combined = pd.concat([getattr(dr, name) for dr in self.dr_list], axis=1)
        
        data_list = []
        row_names = df_combined.index
        
        for row in row_names:
            data_list.append(np.array(df_combined.loc[row].values))
        data = np.array(data_list).T
        correlation_matrix = np.corrcoef(data, rowvar=False)

        # plt.figure(figsize=(10, 6))
        # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        # plt.title("Correlation Matrix")
        # plt.show()
        #print(f"Correlation Matrix for cancer {dr.cancer_type} is: {correlation_matrix}")
        print(f"Correlation Matrix's shape for feature {name} is: {correlation_matrix.shape}\n")
        correlation_matrices.append((name, correlation_matrix))
        return correlation_matrices
    
    def filter_feature_by_correlation(self, name, threshold=0.5, filter=False):
        correlation_matrices = self.Pearson_Correlation(name)
        for name, correlation_matrix in correlation_matrices:
            # Filter features based on correlation matrix
            # Get the upper triangle of the correlation matrix
            
            upper_triangle = np.triu(correlation_matrix, k=1)
            # Get the indices of the features that are highly correlated
            correlated_features = np.where(np.abs(upper_triangle) > threshold)
            # Get the names of the features
            feature_names = getattr(self.dr_list[0], name).index
            # Filter the features
            features_to_remove = set()
            for i, j in zip(*correlated_features):
                if (feature_names[i] not in features_to_remove) and (feature_names[j] not in features_to_remove):
                    features_to_remove.add(feature_names[i])
            print(f"Features left after correlation for {name} are: {len(feature_names) - len(features_to_remove)}\n")
            if filter:
                for dr in self.dr_list:
                    dr.filter_rows(name, row_names=feature_names.difference(features_to_remove))
    
    def filter_feature_by_average(self, name, top_k=0.5, filter=False):
        row_names = getattr(self.dr_list[0], name).index
        filtered_row_names = []
        avg_diff = []
        # import pdb; pdb.set_trace()
        for row in row_names:
            data = []
            for dr in self.dr_list:
                data.append(np.array(getattr(dr, name).loc[row].values))
            avg_diff.append(np.abs(np.mean(data[0]) - np.mean(data[1])))
        avg_diff = np.array(avg_diff)
        indexs = np.argsort(avg_diff)
        filtered_row_names = row_names[indexs[:int(top_k*len(row_names))]]
        
        print(f"Total number of features after filter_feature_by_average: {len(filtered_row_names)} \n")
        if filter:
            for dr in self.dr_list:
                dr.filter_rows(name, row_names=filtered_row_names)
    
    def filter_feature_by_variance(self, name, top_k=0.5, filter=False):
        df_combined = pd.concat([getattr(dr, name) for dr in self.dr_list], axis=1).T
        # calculate the variance of each feature
        variances = df_combined.var(axis=0)
        # sort the features based on variance
        sorted_variances = variances.sort_values(ascending=False)
        # get the top k features
        top_k_features = sorted_variances.index[:int(top_k*len(sorted_variances))]
        print(f"Total number of features after filter_feature_by_variance: {len(top_k_features)} \n")
        if filter:
            for dr in self.dr_list:
                dr.filter_rows(name, row_names=top_k_features)
        
    def pca(self, name, n_components=30):
        df_combined = pd.concat([getattr(dr, name) for dr in self.dr_list], axis=1).T
        labels = []
        for dr in self.dr_list:
            labels.extend([dr.cancer_type] * len(getattr(dr, name).columns))
        # Normalize the data
        scaler = StandardScaler()
        omic_data_normalized = scaler.fit_transform(df_combined)
        import pdb; pdb.set_trace()
        pca = PCA(n_components=n_components)  # Specify the number of components
        principal_components = pca.fit_transform(omic_data_normalized)
        print(f"pc: {principal_components.shape}")
        # df_pca = pd.DataFrame(data=principal_components,) # N_samples, N_features(30)
        self.tsne_visualization(principal_components, name, labels=labels)
        #self.visualize_pca_3d(principal_components, omic_type, data_reader)
    
    # multi variable analysis
    def LASSO_regression(self, name, filter=False):
        row_names = getattr(self.dr_list[0], name).index
        X_train, X_test, y_train, y_test = self.vanilla_train_test_split(name)
        lasso_logistic = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=42)
        
        lasso_logistic.fit(X_train, y_train)

        y_pred = lasso_logistic.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        index = np.where(np.sum(lasso_logistic.coef_, axis=0) != 0)
        print(f"LASSO Regression Accuracy: {accuracy}")
        print(f"Num of features used in LASSO: {len(index[0])}")
        filtered_rows = row_names[index] 
        # import pdb; pdb.set_trace()
        if filter:
            for dr in self.dr_list:
                dr.filter_rows(name, row_names=filtered_rows)
    
    # multi variable analysis
    def RandomForest(self, name, filter=False):
        row_names = getattr(self.dr_list[0], name).index
        X_train, X_test, y_train, y_test = self.vanilla_train_test_split(name)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)

        rf.fit(X_train, y_train)

        feature_importances = rf.feature_importances_
        sorted_feature_importances = np.sort(feature_importances, )[::-1]
        sorted_feature_importances_index = np.argsort(feature_importances)[::-1]
        sum = 0.0
        filtered_rows = []
        for idx, importance in zip(sorted_feature_importances_index, sorted_feature_importances):
            sum += importance
            filtered_rows.append(row_names[idx])
            if sum > 0.9:
                break
        
        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        # import pdb; pdb.set_trace()
        
        print(f"RandomForest Accuracy: {accuracy}")
        print(f"Num of features used in RandomForest: {len(filtered_rows)}")
        # import pdb; pdb.set_trace()
        if filter:
            for dr in self.dr_list:
                dr.filter_rows(name, row_names=filtered_rows)
    
    def get_common_patient(self, filter=False):
        df_combined = pd.concat([getattr(dr, self.names[0]) for dr in self.dr_list], axis=1).T
        patient_ids = set(df_combined.index)
        for name in self.names:
            df_name_combined = pd.concat([getattr(dr, name) for dr in self.dr_list], axis=1).T
            patient_ids = patient_ids.intersection(set(df_name_combined.index))
        patient_ids = list(patient_ids)
        if filter:
            for dr in self.dr_list:
                for name in self.names:
                    dr.filter_cols(name, col_names=patient_ids)
    
    def combine_omics(self,):
        for dr in self.dr_list:
            df_combined = pd.concat([getattr(dr, name) for name in self.names], axis=0)
            setattr(dr, 'combined', df_combined)
    
    # data: N_samples, N_features
    def tsne_visualization(self, data, name, labels):
        COLORS = ['r', 'b', 'g', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        tsne = TSNE(n_components=2, random_state=42)
        # tsne transform
        tsne_data = tsne.fit_transform(data_scaled)
        unique_labels = np.unique(labels)
        plt.figure(figsize=(10, 8))
        for idx in range(data.shape[0]):
            # import pdb; pdb.set_trace()
            color = COLORS[int(np.where(labels[idx] == unique_labels)[0])]
            if idx == 0 or labels[idx] != labels[idx-1]:
                plt.scatter(tsne_data[idx, 0], tsne_data[idx, 1], label=labels[idx], c=color, cmap='viridis', edgecolor='k', s=50)
            else:
                plt.scatter(tsne_data[idx, 0], tsne_data[idx, 1], c=color, cmap='viridis', edgecolor='k', s=50)
        plt.legend()
        plt.title(f't-SNE visualization of {name} data')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.savefig(f'{name}_tsne.png')
    
    def vanilla_train_test_split(self, name):
        X = []
        y = []
        for i, dr in enumerate(self.dr_list):
            num_samples = np.array(getattr(dr, name).T).shape[0]
            X.append(np.array(getattr(dr, name).T))
            y.append(np.array(num_samples*[i]))
        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        return X_train, X_test, y_train, y_test
    
    def kfold_cross_validation(self, name, k=5):
        # implement kfold cross validation train-test split here
        X = []
        y = []
        for i, dr in enumerate(self.dr_list):
            num_samples = np.array(getattr(dr, name).T).shape[0]
            X.append(np.array(getattr(dr, name).T))
            y.append(np.array(num_samples*[i]))
        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        # how to use kf.split(X)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            yield X_train, X_test, y_train, y_test
        
    
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
        plt.savefig(f'{name}_correlation.png', )
        # plt.show()

    
if __name__ == "__main__":
    data_path_list = ["data/filtered_common_features/aml", "data/filtered_common_features/liver", \
        "data/filtered_common_features/melanoma", "data/filtered_common_features/sarcoma",]
    # data_path_list = ["data/origin/breast", "data/origin/colon", "data/origin/lung", ]
    processor = Processor(data_path_list)
    
    for name in ['methy', 'exp',  'mirna',   ]: # 'methy', 'mirna'
        print("************************************************")
        print("Analysis for category:", name)
        print("************************************************\n")
        # import pdb; pdb.set_trace()
        processor.normalize(name)
        # normality_results = processor.norm_evaluation(name, alpha=0.05)
        p_values = processor.ANOVA_Test(name, filter=True)
        processor.filter_feature_by_average(name, top_k=0.5, filter=True)
        processor.filter_feature_by_variance(name, top_k=0.5, filter=True)
        # p_values = processor.Student_T_Test(name, filter=True)
        # print("P-values:", p_values)
        processor.filter_feature_by_correlation(name,\
                                            threshold=0.75, filter=True)
        processor.RandomForest(name, filter=True)
    processor.combine_omics()
        # correlation_matrices = processor.Pearson_Correlation(name)
        # avg_correlations = processor.average_correlation(correlation_matrices)
        # processor.pca(name, n_components=0.9)
        
        # processor.plot_comparison(avg_correlations, name)
    # processor.perform_pca_for_each_omic()
