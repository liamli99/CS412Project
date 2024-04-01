import numpy as np
import pandas as pd
import os
from sklearn.feature_selection import chi2
from scipy import stats
from dataset import DataReader
from processor import Processor
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

class Pipeline:
    def __init__(self, data_path_list):
        self.processor = Processor(data_path_list)
        self.names = ['clinical', 'exp', 'methy', 'mirna', ]
        self.pipeline = {}
        self.pipeline['clinical'] = []
        self.pipeline['exp'] = []
        self.pipeline['methy'] = [
            {'func':self.processor.normalize, 'params':{}}, \
            {'func':self.processor.norm_evaluation, 'params':{'alpha':0.05}}, \
            {'func':self.processor.Student_T_Test, 'params':{'filter':True}}]
        self.pipeline['mirna'] = []
    
    def run(self,):
        for name in self.names:
            for step in self.pipeline[name]:
                step['func'](name, **step['params'])

if __name__ == "__main__":
    data_path_list = ["data/filtered_common_features/aml", "data/filtered_common_features/sarcoma",]
    pip = Pipeline(data_path_list)
    pip.run()