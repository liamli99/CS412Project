from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
class Classifier:
    def __init__(self, model_name='RandomForest'):
        if model_name == 'RandomForest':
            self.model = RandomForestClassifier()
        elif model_name == 'LogisticRegression':
            self.model = LogisticRegression()
        elif model_name == 'SVM':
            self.model = SVC(probability=True)
        elif model_name == 'KNN':
            self.model = KNeighborsClassifier()
        elif model_name == 'XGBoost':
            self.model = XGBClassifier()
        else:
            raise ValueError('Invalid model name')

    def train(self, X, y):
        self.model.fit(X, y)
        return self.model

    def predict(self, X, y_label):
        y_pred = self.model.predict(X)
        # compute all kinds of metrics
        # accuracy, precision, recall, f1, auc, etc.
        accuracy = accuracy_score(y_label, y_pred)
        # print(f"accuracy: {accuracy}")
        # precision = precision_score(y_label, y_pred)
        # recall = recall_score(y_label, y_pred)
        # f1 = f1_score(y_label, y_pred)
        # auc = roc_auc_score(y_label, y_pred)
        metric_dict = classification_report(y_label, y_pred, output_dict=True)
        accuracy = metric_dict['accuracy']
        precision = metric_dict['macro avg']['precision']
        recall = metric_dict['macro avg']['recall']
        f1 = metric_dict['macro avg']['f1-score']
        y_prob = self.model.predict_proba(X)

        # auc for multi-class classification
        auc = roc_auc_score(y_label, y_prob, multi_class='ovr')
        # print(f"auc: {auc}")
        
        # You can print or return these metrics as per your requirement
        # print(f"Accuracy: {accuracy}")
        # print(f"Precision: {precision}")
        # print(f"Recall: {recall}")
        # print(f"F1 Score: {f1}")
        # print(f"AUC: {auc}")
        return {"accuracy": accuracy, \
            "precision": precision, \
            "recall": recall, \
            "f1": f1, \
            "auc": auc}
    
    # for binary classification
    # TODO: for multi-class classification
    def draw_roc_curve(self, X, y_label):
        # draw roc curve
        y_pred_prob = self.model.predict_proba(X)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_label, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('FP Rate')
        plt.ylabel('TP Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.show()