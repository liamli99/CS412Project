import numpy as np
import pandas as pd
import os
from dataset import DataReader

# preprocess the data
class PreProcessor:
    def __init__(self, data_path_list):
        self.dr_list = []
        self.names = ['clinical', 'exp', 'methy', 'mirna', ]
        self.IRRELEVANT_FEATURES = {'clinical':['_EVENT', '_TIME_TO_EVENT', '_TIME_TO_EVENT_UNIT', 'patient_id', '_INTEGRATION', '_OS', '_OS_IND', '_OS_UNIT', '_PATIENT', 'year_of_initial_pathologic_diagnosis', \
            'days_to_birth', 'days_to_death', 'days_to_last_followup', 'days_to_initial_pathologic_diagnosis', 'form_completion_date', 'sample_type', 'sample_type_id', 'tumor_tissue_site', '_cohort', '_primary_site', \
            '_primary_disease', 'informed_consent_verified', 'is_ffpe', 'vial_number', 'tissue_source_site'],
            'exp':[], 'methy':[], 'mirna':[],}
        for path in data_path_list:
            self.dr_list.append(DataReader(path, self.names))
    
    def transform_clinical_data(self, ):
        # transpose the clinical data
        # still need to replace '-' with '.'
        for dr in self.dr_list:
            print(f"dr.cancer_type: {dr.cancer_type}")
            # import pdb; pdb.set_trace()
            dr.clinical = pd.get_dummies(dr.clinical, columns=['gender', 'history_of_neoadjuvant_treatment', 'vital_status'])
            dr.clinical.index = dr.clinical.index.str.replace("-", ".", regex=False)
            dr.clinical = dr.clinical.T
    
    def get_common_features(self, name):
        # get common features among all the cancer types
        # input: name of the omic
        common_features = getattr(self.dr_list[0], name).index
        for dr in self.dr_list[1:]:
            common_features = common_features.intersection(getattr(dr, name).index)
        return common_features

    def filter_common_features(self, ):
        # filter common features from all the omics
        # input: list of DataReader objects
        # output: list of DataReader objects
        for name in self.names:
            common_features_name = self.get_common_features(name)
            for i in range(len(self.dr_list)):
                self.dr_list[i].filter_rows(name, common_features_name)
    
    def filter_irrelevant_features(self, ):
        # filter irrelevant features from all the omics
        # input: list of DataReader objects
        # output: list of DataReader objects
        for name in self.names:
            irrelevant_feature_name = self.IRRELEVANT_FEATURES[name]
            for i in range(len(self.dr_list)):
                getattr(self.dr_list[i], name).drop(index=irrelevant_feature_name, inplace=True)

    def save_data(self, path_to_save):
        os.makedirs(path_to_save, exist_ok=True)
        for dr in self.dr_list:
            dr.save_data(os.path.join(path_to_save, dr.cancer_type))

if __name__ == "__main__":
    data_path_list = ["data/origin/aml", "data/origin/sarcoma", "data/origin/liver", "data/origin/melanoma",]
    
    # "data/origin/overian", 
    # "data/origin/gbm", 
    
    # "data/origin/aml", "data/origin/sarcoma", "data/origin/liver", "data/origin/melanoma",
    # "data/origin/kidney", 
    
    # "data/origin/breast", "data/origin/colon", "data/origin/lung",
    processor = PreProcessor(data_path_list)
    # import pdb; pdb.set_trace()
    processor.transform_clinical_data()
    # import pdb; pdb.set_trace()
    processor.filter_common_features()
    processor.filter_irrelevant_features()
    # import pdb; pdb.set_trace()
    processor.save_data("data/filtered_common_features")