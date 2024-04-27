from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

class Classifier:
    def __init__(self, model_name='RandomForest'):
        if model_name == 'RandomForest':
            self.model = RandomForestRegressor()
        self.model = None

    def train(self, X, y):
        pass

    def predict(self, X):
        pass