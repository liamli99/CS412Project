import numpy as np
import pandas as pd
import os
from sklearn.feature_selection import chi2
from scipy import stats
from dataset import DataReader
from processor import Processor
import matplotlib.pyplot as plt
import seaborn as sns
from classifier import Classifier
import warnings

class Pipeline:
    def __init__(self, data_path_list):
        self.processor = Processor(data_path_list)
        self.names = ['exp', 'methy', 'mirna', ] # 'clinical', 
        self.models = ['RandomForest', 'LogisticRegression', 'SVM', 'KNN', 'XGBoost']
        self.pipeline = {}
        self.pipeline['clinical'] = []
        self.pipeline['exp'] = [
            {'func':self.processor.normalize, 'params':{}}, \
            {'func':self.processor.filter_feature_by_average, 'params':{'top_k':0.1, 'filter':True}}, \
            {'func':self.processor.filter_feature_by_variance, 'params':{'top_k':0.1, 'filter':True}}, \
            # {'func':self.processor.ANOVA_Test, 'params':{'filter':True}}, \
            {'func':self.processor.filter_feature_by_correlation, 'params':{'threshold':0.6, 'filter':True}}, \
            {'func':self.processor.LASSO_regression, 'params':{'filter':True}}
        ]
        self.pipeline['methy'] = [
            ]
        # {'func':self.processor.normalize, 'params':{}}, \
        #     {'func':self.processor.norm_evaluation, 'params':{'alpha':0.05}}, \
        #     {'func':self.processor.Student_T_Test, 'params':{'filter':True}}
        self.pipeline['mirna'] = []
    
    def feature_engineering(self,):
        for name in self.names:
            print(f"Processing {name} data...")
            for step in self.pipeline[name]:
                step['func'](name, **step['params'])
        print(f"Processing combined data...")
        self.processor.combine_omics()
        self.processor.LASSO_regression('combined', filter=True)
    
    def train_test_split(self, name):
        spliter = self.processor.kfold_cross_validation(name)
        return spliter
    
    def evaluate(self, ):
        for name in self.names:
            print(f"{'-'*10}")
            for model in self.models:
                print(f"\nEvaluating {name} data use {model}...")
                kfold_spliter = self.train_test_split(name)
                metric_dict = {}
                for X_train, X_test, y_train, y_test in kfold_spliter:
                    classifier = Classifier(model_name=model)
                    classifier.train(X_train, y_train)
                    metric_dict_fold = classifier.predict(X_test, y_test)
                    for key, value in metric_dict_fold.items():
                        if key not in metric_dict:
                            metric_dict[key] = []
                        metric_dict[key].append(value)
                for key, value in metric_dict.items():
                    print(f"{key}: {np.mean(value)} +- {np.std(value)}")
                    # classifier.draw_roc_curve(X_test, y_test)
    
    def evaluate_multi_omics(self,):
        for model in self.models:
            kfold_spliter = self.train_test_split('combined')
            metric_dict = {}
            for X_train, X_test, y_train, y_test in kfold_spliter:
                classifier = Classifier(model_name=model)
                classifier.train(X_train, y_train)
                metric_dict_fold = classifier.predict(X_test, y_test)
                for key, value in metric_dict_fold.items():
                    if key not in metric_dict:
                        metric_dict[key] = []
                    metric_dict[key].append(value)
            
                # classifier.draw_roc_curve(X_test, y_test)

if __name__ == "__main__":
    data_path_list = ["data/filtered_common_features/aml", "data/filtered_common_features/liver", \
        "data/filtered_common_features/melanoma", "data/filtered_common_features/sarcoma",]
    pip = Pipeline(data_path_list)
    pip.feature_engineering()
    pip.evaluate()
    
    # pip.evaluate_multi_omics()